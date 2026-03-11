import torch
import torch.optim as optim
from tqdm import tqdm
import os
from linearizer.one_step.modules.invertable_network_new import InvUnet
from linearizer.common.song__unet import creat_song_unet
from linearized_flow_matching.core.model_architectures import TimeVaryingGLinearizer, FixedLinearMatrix, TimeG_FlowMatcher
from linearized_flow_matching.core.ema import EMA
from linearized_flow_matching.utils.sampling import sample_and_show
from linearized_flow_matching.utils.checkpointing import save_checkpoint
from linearized_flow_matching.configs.config import CONFIG_DICT

# Unpack config dictionary into variables for easier access
DATASET = CONFIG_DICT['dataset']
IMG_SIZE = CONFIG_DICT['img_size']
IN_CHANNELS = CONFIG_DICT['in_channels']
BATCH_SIZE = CONFIG_DICT['batch_size']
VAL_BATCH_SIZE = CONFIG_DICT['val_batch_size']
MODEL_CHANNELS = CONFIG_DICT['model_channels']
NUM_LAYERS_G = CONFIG_DICT['num_layers_g']
EMA_DECAY = CONFIG_DICT['ema_decay']
LAMBDAS_DICT = CONFIG_DICT['lambdas']
LR = CONFIG_DICT['lr']
EPOCHS = CONFIG_DICT['epochs']
EVAL_INTERVAL = CONFIG_DICT['eval_interval']
GRADIENT_CLIP_THRESOLD = CONFIG_DICT['gradient_clip_threshold']
SAVE_DIR_BASE = CONFIG_DICT['saved_models_base_dir']
SAVE_DIR_RAW = os.path.join(SAVE_DIR_BASE, 'raw')
SAVE_DIR_EMA = os.path.join(SAVE_DIR_BASE, 'ema')


def training_init(
        # model_name,
        device,
        wandb_run=None,
        # img_size=IMG_SIZE,
        # in_channels=IN_CHANNELS,
        # num_layers_g=NUM_LAYERS_G,
        # model_channels=MODEL_CHANNELS,
        # ema_decay=EMA_DECAY,
        lambdas_dict=LAMBDAS_DICT,
        learning_rate=LR,
    ):
    """
    Initializes the Model, EMA, Flow Matcher, and Optimizer.
    """
    # Define Architecture
    flat_dim = IN_CHANNELS * IMG_SIZE * IMG_SIZE
    fixed_A = FixedLinearMatrix(flat_dim).to(device)
    g_net = InvUnet(NUM_LAYERS_G, IN_CHANNELS, IMG_SIZE, creat_song_unet, MODEL_CHANNELS).to(device)

    # Create Linearized Model
    model = TimeVaryingGLinearizer(g_net, fixed_A).to(device)

    # Create EMA Wrapper
    ema = EMA(model, decay=EMA_DECAY)

    # Create Flow Matcher Loss Module
    fm = TimeG_FlowMatcher(
        linearizer=model,
        wandb_run=wandb_run,
        lambdas_dict=lambdas_dict
    )

    # Create Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return model, ema, fm, optimizer



def train_model(
        device,
        model,
        ema,
        fm,
        optimizer,
        train_loader,
        save_dir_raw=SAVE_DIR_RAW,
        save_dir_ema=SAVE_DIR_EMA,
        wandb_run=None,
        num_epochs=EPOCHS,
        gradiend_clip_threshold=GRADIENT_CLIP_THRESOLD
):
    print("Starting Training")

    for epoch in range(num_epochs):
        model.train()

        num_batches = len(train_loader)
        milestones = {
            num_batches // 4: "25%",
            num_batches // 2: "50%",
            (3 * num_batches) // 4: "75%",
            num_batches : "100%"
        }


        # with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
        with tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch", mininterval=10) as pbar:
            for batch_idx, (data, _) in enumerate(pbar):
                data = data.to(device)
                optimizer.zero_grad()

                # loss, fm_loss, lpips_fm_loss, target_L2_loss, target_LPIPS_loss, iso_loss, spec_loss, frob_loss, vel_loss = fm.training_losses(data)
                losses_dict = fm.training_losses(data)

                loss = losses_dict["loss/total"]
                loss.backward()

                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradiend_clip_threshold) # We do clipping, but the returned gradient norm is before clipping

                optimizer.step()

                # Update EMA
                ema.update(model)

                # W&B logging
                if wandb_run is not None:
                    wandb_run.log({
                        "train/gradient_norm": gradient_norm,
                        "epoch": epoch + 1,
                        "batch": batch_idx + (epoch * len(train_loader)), # Absolute step count
                        **losses_dict
                    })


                # if reached 25% batches
                if batch_idx in milestones:
                    batch_percent_str = milestones[batch_idx]

                    samples_exp_raw, samples_discrete_collapsed_raw, samples_iter_raw = sample_and_show(model, epoch, batch_percent_str, wandb_run, model_type_string="RAW", device=device)
                    samples_exp_ema, samples_discrete_collapsed_ema, samples_iter_ema = sample_and_show(ema.get_model(), epoch, batch_percent_str, wandb_run, model_type_string="EMA", device=device)

                    save_checkpoint(model, ema, optimizer, epoch, save_dir_raw, save_dir_ema)

                    # model.run_diagnostics(samples_exp_raw, epoch)
                    # ema.run_diagnostics(samples_exp_ema, epoch)

                    fm.run_diagnostics(data, epoch)

                # pbar.set_postfix(loss=f"{loss.item():.4f}", fm=f"{fm_loss:.4f}", lpips_fm_loss=f"{lpips_fm_loss:.4f}", target_L2_loss=f"{target_L2_loss:.4f}", target_LPIPS_loss=f"{target_LPIPS_loss:.4f}", iso=f"{iso_loss:.4f}", spec=f"{spec_loss:.4f}", frob=f"{frob_loss:.4f}", vel=f"{vel_loss:.4f}", grad_norm = f"{gradient_norm:.4f}")
                pbar.set_postfix({k.replace("loss/", ""): f"{v:.4f}" for k, v in losses_dict.items()}, grad=f"{gradient_norm:.4f}")



        # print(f"Saving models at Epoch {epoch+1}...")
        # raw_model_path = os.path.join(SAVE_DIR_RAW, f'model_{epoch+1}.pth')
        # ema_model_path = os.path.join(SAVE_DIR_EMA, f'model_ema_{epoch+1}.pth')
        # torch.save(model.state_dict(), raw_model_path)
        # ema_inference_model = ema.get_model()
        # torch.save(ema_inference_model.state_dict(), ema_model_path)

        # # save optimizer too
        # checkpoint = {
        #         'epoch': epoch + 1,                            # Save the next epoch number
        #         'model_state_dict': model.state_dict(),        # Save the model weights
        #         'optimizer_state_dict': optimizer.state_dict() # Save the optimizer momentum/velocity
        #     }
        # raw_model_path = os.path.join(SAVE_DIR_RAW, f'model_{epoch+1}_with_optimizer.pth')
        # # Save the dictionary to a file
        # torch.save(checkpoint, raw_model_path)
        # print(f"Models saved at Epoch {epoch+1}.")


        # if (epoch + 1) % EVAL_INTERVAL == 0:
        #         # --------------------------------------------------
        #         # 1. RAW MODEL SAMPLING & LOGGING
        #         # --------------------------------------------------
        #         print(f"--- Epoch {epoch+1}: Sampling with RAW Model ---")
        #         model.eval()
        #         model_matcher = TimeG_FlowMatcher(model)
        #         model_matcher.linearizer.linear_network.cache_exponential()

        #         with torch.no_grad(): # Don't forget this to save memory during sampling!
        #             samples_raw = model_matcher.sample_exponential(16, IMG_SIZE, IN_CHANNELS, device)
        #         show_samples(samples_raw, title=f"Epoch {epoch+1} (Original)")

        #         # Convert to W&B Images and log
        #         if wandb_run is not None:
        #             wandb_images_raw = [wandb_run.Image(img) for img in samples_raw]
        #             wandb_run.log({"samples/generated_images_RAW": wandb_images_raw, "epoch": epoch + 1})

        #         # --------------------------------------------------
        #         # 2. EMA MODEL SAMPLING & LOGGING
        #         # --------------------------------------------------
        #         print(f"--- Epoch {epoch+1}: Sampling with EMA Model ---")
        #         ema_model = ema.get_model()
        #         ema_matcher = TimeG_FlowMatcher(ema_model)
        #         ema_model.linear_network.cache_exponential()

        #         with torch.no_grad(): # Don't forget this!
        #             samples_ema = ema_matcher.sample_exponential(16, IMG_SIZE, IN_CHANNELS, device)
        #         show_samples(samples_ema, title=f"Epoch {epoch+1} (EMA)")

        #         # Convert to W&B Images and log
        #         if wandb_run is not None:
        #             wandb_images_ema = [wandb_run.Image(img) for img in samples_ema]
        #             wandb_run.log({"samples/generated_images_EMA": wandb_images_ema, "epoch": epoch + 1})

    if wandb_run is not None:
        wandb_run.finish()



# model, ema, fm, optimizer = training_init()

# train_model(
#     model=model,
#     ema=ema,
#     fm=fm,
#     optimizer=optimizer,
#     train_loader=train_loader,
#     wandb_run=wandb_run
# )