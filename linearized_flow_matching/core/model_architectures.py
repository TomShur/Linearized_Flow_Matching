import torch
import torch.nn as nn
import torch.nn.functional as F
from linearized_flow_matching.core.losses import IsometryLoss
from piq import LPIPS
from linearizer.one_step.modules.linear_network import OneStepLinearModule
from linearizer.linearizer import Linearizer
# from utils import pair_batch
from linearized_flow_matching.core.pair_batch import pair_batch
from linearized_flow_matching.configs.config import CONFIG_DICT

# Unpack config
INIT_FACTOR_A = CONFIG_DICT['init_factor_A']
# DATASET = CONFIG_DICT['dataset']
IMG_SIZE = CONFIG_DICT['img_size']
IN_CHANNELS = CONFIG_DICT['in_channels']
BATCH_SIZE = CONFIG_DICT['batch_size']
VAL_BATCH_SIZE = CONFIG_DICT['val_batch_size']
MODEL_CHANNELS = CONFIG_DICT['model_channels']
NUM_LAYERS_G = CONFIG_DICT['num_layers_g']
INT_T_CONF = CONFIG_DICT['int_t_conf']
LAMBDAS_DICT = CONFIG_DICT['lambdas']
NUM_SAMPLING_STEPS = CONFIG_DICT['num_sampling_steps']
NUM_SAMPLES = CONFIG_DICT['num_samples']



# --- 1. The Fixed Linear Operator A ---
class FixedLinearMatrix(OneStepLinearModule):
    def __init__(self, dim, init_factor=INIT_FACTOR_A):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim) * init_factor)
        self.register_buffer('exp_A_cached', None)

    def forward(self, x, **kwargs):
        original_shape = x.shape
        if x.dim() > 2: x = x.view(x.shape[0], -1)
        out = x @ self.weight.t()
        return out.view(original_shape)

    def cache_exponential(self):
        with torch.no_grad():
            self.exp_A_cached = torch.linalg.matrix_exp(self.weight.t())

    def forward_exponential(self, x):
        if self.exp_A_cached is None: self.cache_exponential()
        original_shape = x.shape
        if x.dim() > 2: x = x.view(x.shape[0], -1)
        out = x @ self.exp_A_cached
        return out.view(original_shape)

    def get_lin_t(self, t):
        return self.weight.unsqueeze(0)
    






# ---- The Time-Varying Wrapper ---
class TimeVaryingGLinearizer(Linearizer):
    def __init__(self, g_net, fixed_A_net):
        # We pass g_net for both gx and gy because it handles time internally
        super().__init__(gx=g_net, linear_network=fixed_A_net, gy=g_net)

    def g(self, x, t):
        return self.net_gx(x, t=t)

    def g_inverse(self, z, t):
        return self.net_gx.inverse(z, t=t)

    def A(self, z):
        return self.linear_network(z)
    




class TimeG_FlowMatcher:
    def __init__(
            self,
            linearizer,
            wandb_run,
            inv_t_conf=INT_T_CONF,
            lambdas_dict=LAMBDAS_DICT,
            ):
        self.linearizer = linearizer
        self.iso_loss_fn = IsometryLoss()

        self.wandb_run = wandb_run

        # self.lpips_fn = LPIPS(replace_pooling=True, reduction="mean")

        device = next(linearizer.parameters()).device
        self.lpips_fn = LPIPS(replace_pooling=True, reduction="mean").to(device)

        self.inv_t_conf = inv_t_conf



        self.lambda_FM_L2 = lambdas_dict['FM_L2']
        self.lambda_FM_LPIPS = lambdas_dict['FM_LPIPS']

        self.lambda_target_L2 = lambdas_dict['TARGET_L2']
        self.lambda_target_LPIPS = lambdas_dict['TARGET_LPIPS']

        self.lambda_iso = lambdas_dict['ISO']
        self.lambda_frob = lambdas_dict['FROB']
        self.lambda_spec = lambdas_dict['SPEC']
        self.lambda_vel = lambdas_dict['VEL']


    def _calc_fm_loss(self, velocity_pred, velocity_target, t, batch_size, device):
        if self.lambda_FM_L2 <= 0:
            return torch.tensor(0.0, device=device)

        if self.inv_t_conf == 'False':
            return ((velocity_pred - velocity_target) ** 2).mean()

        inv_t = None
        if self.inv_t_conf == '1':
            inv_t = torch.ones(batch_size, device=device)
        elif self.inv_t_conf == 'Random':
            inv_t = torch.rand(batch_size, device=device)
        else: # 'T'
            inv_t = t.clone()

        inv_velocity_pred = self.linearizer.g_inverse(velocity_pred, t=inv_t)
        inv_velocity_target = self.linearizer.g_inverse(velocity_target, t=inv_t)
        return ((inv_velocity_pred - inv_velocity_target) ** 2).mean()

    def _calc_lpips_fm_loss(self, x0, x1, velocity_pred, device):
        # Currently placeholder based on provided code
        return torch.tensor(0.0, device=device)
        # if self.lpips_fn is not None: # if self.lambda_FM_LPIPS > 0:
        #     lpips_fm_loss = torch.tensor(0.0, device=x1.device)
        #     if self.lambda_FM_LPIPS > 0:
        #         # Step A: Extrapolate the predicted x1 from the velocity
        #         # If your velocity is defined as v = x1 - x0, then:
        #         # x1_pred = x0 + inv_velocity_pred
        #         # (Adjust this formula if your velocity definition is different!)
        #         x1_pred = x0 + inv_velocity_pred

        #         # Step B: LPIPS expects 3-channel RGB images.
        #         # If using MNIST (1 channel), we must duplicate the channels.
        #         if x1.shape[1] == 1:
        #             x1_pred_rgb = x1_pred.repeat(1, 3, 1, 1)
        #             x1_target_rgb = x1.repeat(1, 3, 1, 1)
        #         else:
        #             x1_pred_rgb = x1_pred
        #             x1_target_rgb = x1

        #         # Step C: LPIPS expects values roughly in the [0, 1] range.
        #         # We clamp to prevent the VGG network from seeing wild extrapolated values
        #         lpips_fm_loss = self.lpips_fn(
        #             x1_pred_rgb.clamp(0, 1),
        #             x1_target_rgb.clamp(0, 1)
        #         )

    def _predict_target_x1(self, x0, t_start, batch_size, device):
        """Helper to generate hat_x_1 for target losses"""
        # t_0 = torch.zeros(batch_size, device=device)

        t_start = t_start.clone()
        z_0 = self.linearizer.g(x0, t_start)
        if isinstance(z_0, tuple): z_0 = z_0[0]

        # Matrix Exponential Jump
        A = next(self.linearizer.linear_network.parameters())
        exp_A = torch.matrix_exp(A)

        z_0_flat = z_0.view(batch_size, -1)
        z_1_flat = torch.matmul(exp_A, z_0_flat.T).T
        z_1_predicted = z_1_flat.view_as(x0)

        t_1 = torch.ones(batch_size, device=device)
        hat_x_1 = self.linearizer.g_inverse(z_1_predicted, t_1)
        if isinstance(hat_x_1, tuple): hat_x_1 = hat_x_1[0]

        return hat_x_1

    def _calc_target_l2_loss(self, hat_x_1, x1, device):
        if self.lambda_target_L2 > 0:
            return F.mse_loss(hat_x_1, x1)
        return torch.tensor(0.0, device=device)

    def _calc_target_lpips_loss(self, hat_x_1, x1, device):
        if self.lambda_target_LPIPS > 0:
            if x1.shape[1] == 1:
                hat_x_1_rgb = hat_x_1.repeat(1, 3, 1, 1)
                x1_rgb = x1.repeat(1, 3, 1, 1)
            else:
                hat_x_1_rgb = hat_x_1
                x1_rgb = x1

            return self.lpips_fn(
                hat_x_1_rgb.clamp(-1.0, 1.0),
                x1_rgb.clamp(-1.0, 1.0)
            )
        return torch.tensor(0.0, device=device)

    def _calc_frob_loss(self, matrix_A, device):
        if self.lambda_frob > 0:
            return torch.sum(matrix_A ** 2)
        return torch.tensor(0.0, device=device)

    def _calc_spectral_loss(self, matrix_A, device):
        if self.lambda_spec > 0:
            return torch.linalg.matrix_norm(matrix_A, ord=2)
        return torch.tensor(0.0, device=device)

    def _calc_vel_loss(self, velocity_pred, device):
        if self.lambda_vel > 0:
            return torch.sum(velocity_pred ** 2)
        return torch.tensor(0.0, device=device)

    def _calc_iso_loss(self, x_t, g_t_x_t, t, x1_shape, device):
        if self.lambda_iso > 0:
            zeros = torch.zeros(x1_shape, device=device)
            g_t_0 = self.linearizer.g(zeros, t=t)
            return self.iso_loss_fn(x_t, g_t_x_t, g_t_0)
        return torch.tensor(0.0, device=device)

    def training_losses(self, x1):
        batch_size = x1.shape[0]
        device = x1.device

        # --- Compute needed values for loss calculation ---
        x0 = torch.randn_like(x1) # sample gaussian noise ~ N(0,1)
        x0 = pair_batch(x0, x1) # re-pair the batch
        t = torch.rand(batch_size, device=device) # sample random values for time t, uniform distribution
        t0 = torch.zeros(batch_size, device=device) # t0 = 0
        t1 = torch.ones(batch_size, device=device) # t1 = 1
        g0_x0 = self.linearizer.g(x0, t=t0) # g0(x0)
        g1_x1 = self.linearizer.g(x1, t=t1) # g1(x1)

        # We pass xt through gt
        zt = (1 - t[:, None, None, None]) * g0_x0 + t[:, None, None, None] * g1_x1 # interpolation: zt = (1-t)*g0(x0) + t*g1(x1)
        xt = self.linearizer.g_inverse(zt, t=t) # xt = gt^-1(zt)
        gt_xt = self.linearizer.g(xt, t=t) # gt(xt) (should be equal to zt)

        velocity_pred = self.linearizer.A(gt_xt) # A*gt(xt)
        velocity_target = g1_x1 - g0_x0 # g1(x1) - g0(x0)

        matrix_A = self.linearizer.linear_network.weight # A

        # --- Calculate Loss Terms ---

        # FM Losses
        fm_loss = self._calc_fm_loss(velocity_pred, velocity_target, t, batch_size, device)
        lpips_fm_loss = self._calc_lpips_fm_loss(x0, x1, velocity_pred, device)

        # Target Losses
        target_L2_loss = torch.tensor(0.0, device=device)
        target_LPIPS_loss = torch.tensor(0.0, device=device)

        if self.lambda_target_L2 > 0 or self.lambda_target_LPIPS > 0:
            # hat_x_1 = self._predict_target_x1(x0, batch_size, device)
            hat_x_1 = self._predict_target_x1(x0, t0, batch_size, device)

            target_L2_loss = self._calc_target_l2_loss(hat_x_1, x1, t_start=t0, device=device)
            target_LPIPS_loss = self._calc_target_lpips_loss(hat_x_1, x1, t_start=t0, device=device)

        # Regularization Losses
        frob_loss = self._calc_frob_loss(matrix_A, device)
        spectral_loss = self._calc_spectral_loss(matrix_A, device)
        vel_loss = self._calc_vel_loss(velocity_pred, device)
        iso_loss = self._calc_iso_loss(xt, gt_xt, t, x1.shape, device)

        # --- Total Loss ---
        total_loss = (
            self.lambda_FM_L2 * fm_loss
            + (self.lambda_FM_LPIPS * lpips_fm_loss)
            + (self.lambda_target_L2 * target_L2_loss)
            + (self.lambda_target_LPIPS * target_LPIPS_loss)
            + (self.lambda_frob * frob_loss)
            + (self.lambda_spec * spectral_loss)
            + (self.lambda_vel * vel_loss)
            + (self.lambda_iso * iso_loss)
        )

        return {
            "loss/total": total_loss,
            "loss/fm": fm_loss.item(),
            "loss/lpips": lpips_fm_loss.item(),
            "loss/target_l2": target_L2_loss.item(),
            "loss/target_lpips": target_LPIPS_loss.item(),
            "loss/iso": iso_loss.item(),
            "loss/spectral": spectral_loss.item(),
            "loss/frob": frob_loss.item(),
            "loss/velocity": vel_loss.item()
        }



    @torch.no_grad()
    def _test_invertibility(self, x, t):
        """
        Test 1: Round-trip Invertibility (x -> z -> x_rec)
        """
        # Forward
        z = self.linearizer.g(x, t) # z = gt(x)
        if isinstance(z, tuple): z = z[0]

        # Inverse
        x_rec = self.linearizer.g_inverse(z, t) # x_rec =  gt^-1(z) = gt^-1(gt(x))
        if isinstance(x_rec, tuple): x_rec = x_rec[0]

        # Metrics
        mse = F.mse_loss(x_rec, x).item()
        max_err = torch.max(torch.abs(x_rec - x)).item()

        return {"round_trip_mse": mse, "round_trip_max_error": max_err}

    @torch.no_grad()
    def _test_latent_space(self, z):
        """
        Test 2: Latent Space Statistics (Check for exploding values)
        """
        return {
            "latent_min": z.min().item(),
            "latent_max": z.max().item(),
            "latent_std": z.std().item()
        }

    @torch.no_grad()
    def _test_time_sensitivity(self, x):
        """
        Test 3: Time Embedding Sensitivity
        Checks if g(x, t=0) is significantly different from g(x, t=1)
        """
        batch_size = x.shape[0]
        device = x.device

        t0 = torch.zeros(batch_size, device=device)
        t1 = torch.ones(batch_size, device=device)

        z0 = self.linearizer.g(x, t0)
        z1 = self.linearizer.g(x, t1)

        if isinstance(z0, tuple): z0 = z0[0]
        if isinstance(z1, tuple): z1 = z1[0]

        diff = torch.mean(torch.abs(z0 - z1)).item()
        return {"time_sensitivity_diff": diff}

    @torch.no_grad()
    def _test_matrix_stability(self, x_noise, device):
        """
        Test 4: Matrix Exponential Stability
        Checks if e^A * z0 produces reasonable values compared to a standard normal
        """
        batch_size = x_noise.shape[0]
        t0 = torch.zeros(batch_size, device=device)

        # Encode noise at t=0
        z0 = self.linearizer.g(x_noise, t0)
        if isinstance(z0, tuple): z0 = z0[0]

        # Calculate e^A
        A = next(self.linearizer.linear_network.parameters())
        # exp_A = torch.matrix_exp(A)
        exp_A = torch.linalg.matrix_exp(A)

        # Jump forward
        z0_flat = z0.view(batch_size, -1)
        z1_pred_flat = torch.matmul(exp_A, z0_flat.T).T
        z1_pred = z1_pred_flat.view_as(x_noise)

        return {
            "jump_pred_std": z1_pred.std().item(),
            "jump_pred_max": z1_pred.max().item()
        }

    @torch.no_grad()
    def run_diagnostics(self, x_real, step):
        """
        Master method: Runs all tests and logs to wandb
        """
        self.linearizer.eval()
        device = x_real.device
        batch_size = x_real.shape[0]

        # Prepare Inputs
        t_rand = torch.rand(batch_size, device=device)
        x_noise = torch.randn_like(x_real)

        # 1. Invertibility
        inv_metrics = self._test_invertibility(x_real, t_rand)

        # 2. Latent Stats (using the z from the invertibility test)
        z_rand = self.linearizer.g(x_real, t_rand)
        if isinstance(z_rand, tuple): z_rand = z_rand[0]
        lat_metrics = self._test_latent_space(z_rand)

        # 3. Time Sensitivity
        time_metrics = self._test_time_sensitivity(x_real)

        # 4. Matrix Stability
        mat_metrics = self._test_matrix_stability(x_noise, device)

        # Log to WandB
        if self.wandb_run.run is not None:
            log_dict = {}
            for k, v in {**inv_metrics, **lat_metrics, **time_metrics, **mat_metrics}.items():
                log_dict[f"diagnostics/{k}"] = v
            self.wandb_run.log(log_dict, step=step)

        # Print summary to console for sanity check
        print(f"\n[Step {step}] Diagnostics:")
        print(f"  MSE: {inv_metrics['round_trip_mse']:.6f} | Max Err: {inv_metrics['round_trip_max_error']:.6f}")
        print(f"  Latent Std: {lat_metrics['latent_std']:.3f} | Jump Std: {mat_metrics['jump_pred_std']:.3f}")

        self.linearizer.train()





    @torch.no_grad()
    def _sample_exp(self, z0, t1):
        z1 = self.linearizer.linear_network.forward_exponential(z0)
        x_pred = self.linearizer.g_inverse(z1, t=t1)
        return x_pred

    @torch.no_grad()
    def _sample_discrete_collapsed(self, z0_flat, t1, step_matrix, num_steps, x_noise):

        collapsed_matrix = torch.linalg.matrix_power(step_matrix, num_steps)
        z1_collapsed_flat = torch.matmul(collapsed_matrix, z0_flat.T).T
        z1_collapsed = z1_collapsed_flat.view_as(x_noise)
        x_pred = self.linearizer.g_inverse(z1_collapsed, t=t1)
        if isinstance(x_pred, tuple): x_pred = x_pred[0]
        return x_pred

    @torch.no_grad()
    def _sample_iterative(self, z0_flat, t1, step_matrix, num_steps, x_noise):
        z_curr_flat = z0_flat.clone()
        for i in range(num_steps):
            z_curr_flat = torch.matmul(step_matrix, z_curr_flat.T).T

        z1_iter = z_curr_flat.view_as(x_noise)

        x_pred = self.linearizer.g_inverse(z1_iter, t=t1)
        if isinstance(x_pred, tuple): x_pred = x_pred[0]


        return x_pred

    @torch.no_grad()
    def sample(self, device, num_samples=16, img_size=IMG_SIZE, channels=IN_CHANNELS, num_steps=NUM_SAMPLING_STEPS):
        self.linearizer.eval()

        # Base Setup & Encoding
        x_noise = torch.randn(num_samples, channels, img_size, img_size, device=device)
        t0 = torch.zeros(num_samples, device=device)
        t1 = torch.ones(num_samples, device=device)

        z0 = self.linearizer.g(x_noise, t=t0)
        if isinstance(z0, tuple): z0 = z0[0]
        z0_flat = z0.view(num_samples, -1)

        # Linear Matrix Setup
        A = next(self.linearizer.linear_network.parameters())
        I = torch.eye(A.shape[0], device=device)
        delta_t = 1.0 / num_steps
        step_matrix = (I + delta_t*A)

        x_pred_exp = self._sample_exp(z0, t1) # exp sampling
        x_pred_discrete_collapsed = self._sample_discrete_collapsed(z0_flat, t1, step_matrix, num_steps, x_noise) # discrete collapsed matrix sampling
        x_pred_iterative = self._sample_iterative(z0_flat, t1, step_matrix, num_steps, x_noise) # iterative sampling

        self.linearizer.train()
        return x_pred_exp, x_pred_discrete_collapsed, x_pred_iterative
        # return {
        #     "exp": x_pred_exp,
        #     "discrete_collapsed": x_pred_discrete_collapsed,
        #     "iterative": x_pred_iterative
        # }
