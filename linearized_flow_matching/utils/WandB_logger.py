import wandb
from linearized_flow_matching.configs.config import CONFIG_DICT

# --- Unpack Configurations ---
DATASET = CONFIG_DICT['dataset']
IMG_SIZE = CONFIG_DICT['img_size']
IN_CHANNELS = CONFIG_DICT['in_channels']
BATCH_SIZE = CONFIG_DICT['batch_size']
VAL_BATCH_SIZE = CONFIG_DICT['val_batch_size']
MODEL_CHANNELS = CONFIG_DICT['model_channels']
NUM_LAYERS_G = CONFIG_DICT['num_layers_g']
INIT_FACTOR_A = CONFIG_DICT['init_factor_A']
LR = CONFIG_DICT['lr']
EPOCHS = CONFIG_DICT['epochs']
EVAL_INTERVAL = CONFIG_DICT['eval_interval']
EMA_DECAY = CONFIG_DICT['ema_decay']
GRADIENT_CLIP_THRESOLD = CONFIG_DICT['gradient_clip_threshold']
INT_T_CONF = CONFIG_DICT['int_t_conf']
LAMBDAS_DICT = CONFIG_DICT['lambdas']

def wandb_init(
    model_name,
        project_name="linearized-flow-matching",
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        gradient_clip=GRADIENT_CLIP_THRESOLD,
        optimizer="AdamW",
        model_channels=MODEL_CHANNELS,
        num_layers_g=NUM_LAYERS_G,
        img_size=IMG_SIZE,
        in_channels=IN_CHANNELS,
        inv_t_conf=INT_T_CONF,
        ema_decay=EMA_DECAY,
        lambdas_dict=LAMBDAS_DICT,
    ):
    wandb_run = wandb.init(
        project=project_name,
        name=model_name,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "gradient_clip": gradient_clip,
            "optimizer": optimizer,
            "model_channels": model_channels,
            "num_layers_g": num_layers_g,
            "img_size": img_size,
            "in_channels": in_channels,
            "inv_t_conf": inv_t_conf,
            "ema_decay": ema_decay,
            **lambdas_dict
        }
    )

    return wandb_run
