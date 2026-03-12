import wandb
import torch
from dotenv import load_dotenv
import os
from linearized_flow_matching.core.training_functions import training_init, train_model
from linearized_flow_matching.configs.config import CONFIG_DICT
from linearized_flow_matching.utils.WandB_logger import wandb_init
from linearized_flow_matching.utils.setup import seed_everything
from linearized_flow_matching.data.mnist import get_optimized_loader

def main(device, model_index):
    seed_everything()

    MODEL_NAME = f'model{model_index}'

    if os.getenv("WANDB_API_KEY"):
        wandb.login(key=os.getenv("WANDB_API_KEY"))
    else:
        wandb.login()
    wandb_run = wandb_init(model_name=MODEL_NAME) # Instantiate

    train_loader = get_optimized_loader()

    model, ema, fm, optimizer = training_init(device=device, wandb_run=wandb_run)
    train_model(
        device=device,
        model=model,
        ema=ema,
        fm=fm,
        optimizer=optimizer,
        train_loader=train_loader,
        wandb_run=wandb_run
    )

load_dotenv()

MODEL_INDEX = CONFIG_DICT['model_index']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

main(device=device, model_index=MODEL_INDEX)