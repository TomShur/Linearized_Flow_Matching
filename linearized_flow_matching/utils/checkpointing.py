import os
import torch
from configs.config import SAVE_DIR_RAW, SAVE_DIR_EMA




def save_checkpoint(model, ema, optimizer, epoch, save_dir_raw=SAVE_DIR_RAW, save_dir_ema=SAVE_DIR_EMA):
    """
    Saves model weights, EMA weights, and a full training checkpoint.
    """
    # Ensure directories exist
    os.makedirs(save_dir_raw, exist_ok=True)
    os.makedirs(save_dir_ema, exist_ok=True)

    epoch_num = epoch + 1
    print(f"Saving models at Epoch {epoch_num}...")

    # 1. Save Raw Model Weights (Lightweight - for inference/transfer)
    raw_path = os.path.join(save_dir_raw, f'model_{epoch_num}.pth')
    torch.save(model.state_dict(), raw_path)

    # 2. Save EMA Model Weights (Best for sampling/generation)
    ema_path = os.path.join(save_dir_ema, f'model_ema_{epoch_num}.pth')
    torch.save(ema.get_model().state_dict(), ema_path)

    # 3. Save Full Checkpoint (For resuming training)
    # This contains optimizer state, allowing you to resume exactly where you left off
    checkpoint = {
        'epoch': epoch_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # Optional: Save EMA internal state if your EMA class has buffers (shadow params)
        # 'ema_state_dict': ema.state_dict()
    }
    checkpoint_path = os.path.join(save_dir_raw, f'model_{epoch_num}_optimizer.pth')
    torch.save(checkpoint, checkpoint_path)

    print(f"✅ Models saved at Epoch {epoch_num}.")

# save_checkpoint(model, ema, optimizer, epoch, SAVE_DIR_RAW, SAVE_DIR_EMA)