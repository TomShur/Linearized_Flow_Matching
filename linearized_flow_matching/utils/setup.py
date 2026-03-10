import os
import torch
from linearized_flow_matching.configs.config import CONFIG_DICT

SEED=CONFIG_DICT['random_seed']

def seed_everything(seed=SEED):
    """
    Locks the random seed for reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # These two lines ensure the GPU operations are deterministic
    # (might slightly slow down training, but worth it for debugging)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Global Seed set to {seed}")

# seed_everything()