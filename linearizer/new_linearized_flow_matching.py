import torch
import torch.nn as nn
import torch.linalg
from one_step.modules.linear_network import OneStepLinearModule

from linearizer import Linearizer


class FixedLinearMatrix(OneStepLinearModule):
    """
    A fixed, learnable linear operator A with caching for e^A.
    """

    def __init__(self, dim):
        super().__init__()
        # Initialize A close to zero/identity for stability
        self.weight = nn.Parameter(torch.randn(dim, dim) * 0.01)

        # Register a buffer for the cached exponential.
        # Buffers are saved in state_dict but not trained by the optimizer.
        self.register_buffer('exp_A_cached', None)

    def forward(self, x, **kwargs):
        """Standard forward pass for Training (uses dynamic A)"""
        # x: [B, D] or [B, C, H, W]
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)

        # Linear transform: y = xA^T
        out = x @ self.weight.t()

        return out.view(original_shape)

    def cache_exponential(self):
        """Computes and stores e^(A^T). Call this once before inference."""
        print("Caching Matrix Exponential...")
        with torch.no_grad():
            # We compute exp(A^T) because the operation is x @ A^T.
            # So the evolution is x_t = x_0 @ exp(A^T * t)
            self.exp_A_cached = torch.linalg.matrix_exp(self.weight.t())

    def forward_exponential(self, x):
        """
        Fast Inference forward pass.
        Uses the cached exponential matrix: x_1 = x_0 @ e^(A^T)
        """
        if self.exp_A_cached is None:
            self.cache_exponential()

        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)

        # Apply pre-calculated exponential
        out = x @ self.exp_A_cached

        return out.view(original_shape)

    def get_lin_t(self, t):
        # Used if we ever want to run Runge-Kutta manually (sanity check)
        return self.weight.unsqueeze(0)



class TimeVaryingGLinearizer(Linearizer):
    def __init__(self, g_net, fixed_A_net):
        # We pass the same network for gx and gy because it's a single g(x,t)
        super().__init__(gx=g_net, linear_network=fixed_A_net, gy=g_net)

    def g(self, x, t):
        return self.net_gx(x, t=t)

    def g_inverse(self, z, t):
        return self.net_gx.inverse(z, t=t)

    def A(self, z):
        return self.linear_network(z)

    def prepare_for_inference(self):
        """Helper to trigger caching on the underlying linear network"""
        if hasattr(self.linear_network, 'cache_exponential'):
            self.linearizer.linear_network.cache_exponential()


class TimeG_FlowMatcher:
    def __init__(self, linearizer: TimeVaryingGLinearizer):
        self.linearizer = linearizer

    def training_losses(self, x1, x0=None):
        # ... (Same as previous implementation) ...
        # (Included here for completeness context, uses standard self.linearizer.A)
        batch_size = x1.shape[0]
        device = x1.device
        if x0 is None: x0 = torch.randn_like(x1)

        t = torch.rand(batch_size, device=device)
        t0 = torch.zeros(batch_size, device=device)
        t1 = torch.ones(batch_size, device=device)

        g0_x0 = self.linearizer.g(x0, t=t0)
        g1_x1 = self.linearizer.g(x1, t=t1)

        z_t = (1 - t[:, None, None, None]) * g0_x0 + t[:, None, None, None] * g1_x1
        x_t = self.linearizer.g_inverse(z_t, t=t)

        # Enforce consistency and learn A
        g_t_x_t = self.linearizer.g(x_t, t=t)
        velocity_pred = self.linearizer.A(g_t_x_t)
        velocity_target = g1_x1 - g0_x0  # Target is the total displacement

        return ((velocity_pred - velocity_target) ** 2).mean()

    @torch.no_grad()
    def sample_exponential(self, x_noise, device):
        """
        Optimized One-Step Sampling.
        Formula: x1 = g_1^-1( g_0(x0) * e^(A^T) )
        """
        self.linearizer.eval()
        b = x_noise.shape[0]

        # 1. Encode noise at t=0
        t0 = torch.zeros(b, device=device)
        z0 = self.linearizer.g(x_noise, t=t0)

        # 2. Apply Cached Matrix Exponential
        # The linear_network handles the caching logic internally
        z1 = self.linearizer.linear_network.forward_exponential(z0)

        # 3. Decode at t=1
        t1 = torch.ones(b, device=device)
        x_pred = self.linearizer.g_inverse(z1, t=t1)

        return x_pred





# # 1. Setup
# dim = 3 * 32 * 32 # for CIFAR/MNIST size
# fixed_A = FixedLinearMatrix(dim)
# g_net = ... # Your InvUnet
# linearizer = TimeVaryingGLinearizer(g_net, fixed_A)
# matcher = TimeG_FlowMatcher(linearizer)
#
# # 2. Training Loop (Standard)
# # ... optimizer.step() ...
#
# # 3. Inference
# # You can explicitly cache before the loop, or let the first sample call do it.
# linearizer.linear_network.cache_exponential()
#
# # noise = torch.randn(16, 3, 32, 32).cuda()
# # images = matcher.sample_exponential(noise, device='cuda')









