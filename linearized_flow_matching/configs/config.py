import os

CONFIG_DICT = {

    'model_index': 47,

    'saved_models_base_dir': 'linearized_flow_matching/saved_models',
    'random_seed': 42,

    'spec_loss_fast_calculation' : True, # Whether to use the fast approximation for the spectral loss, or the exact calculation. 

    # --- Data & Dimensions ---
    'dataset': 'mnist',
    'img_size': 32,
    'in_channels': 1,
    'batch_size': 64,    
    'val_batch_size': 64,

    # --- Model Architecture ---
    'model_channels': 16,        # U-Net capacity (Alternatives: 32)
    'num_layers_g': 3,           # Depth of g (Alternatives: 4)
    'init_factor_A': 1e-2,       # Initialization factor for the noise A is initialized with

    # --- Training Hyperparameters ---
    'lr': 1e-4,                  # Alternatives: 1e-3, 1e-2
    'epochs': 10,                # Alternatives: 20
    'eval_interval': 1,          # Print samples every N epochs
    'ema_decay': 0.9999,         # Alternatives: 0.999
    'gradient_clip_threshold': 3.0, # Alternatives: 2.0, 1.0, 10.0

    # Parameter for whether we use the inv_t, and what t will be - the t of the gt, randomly sampled, constant like 0/1/0.5
    'int_t_conf': 'False',       # Alternatives: 'T', 'Random', '1'

    # --- Loss Weights (Lambdas) ---
    'lambdas': {
        'FM_L2': 0.0,            # 1.0, 2.5
        'FM_LPIPS': 0.0,         # 1e-1, 1e-2
        'ISO': 0.0,              # 1e-4, 1e-3, 1e-1, 1e-2
        'FROB': 0.0,             # 1e-3
        'SPEC': 0.0,             # 1e-3, 1e-1
        'VEL': 0.0,              # 1e-3
        'TARGET_L2': 1.0,        # 0.5, 1e-1, 1e-3
        'TARGET_LPIPS': 1e-1     # 1e-2, 1e-3, 1e-5
    },

    # --- Sampling Configuration ---
    'num_sampling_steps': 100,   # For discrete and iterative sampling
    'num_samples': 16            # How many samples (per type) to show each time during training
    
    # Future hyper-parameters to maybe add:
    # 'dropout': 0.0,
    # 'batch_norm': True,
    # 'weight_init': '...',
    # 'bipartite_p': 1,
}

SAVE_DIR_BASE = CONFIG_DICT['saved_models_base_dir']

SAVE_DIR_RAW = os.path.join(SAVE_DIR_BASE, 'raw')
if not os.path.exists(SAVE_DIR_RAW):
    os.makedirs(SAVE_DIR_RAW)

SAVE_DIR_EMA = os.path.join(SAVE_DIR_BASE, 'ema')
if not os.path.exists(SAVE_DIR_EMA):
    os.makedirs(SAVE_DIR_EMA)





# DATASET = 'mnist'
# IMG_SIZE = 32
# IN_CHANNELS = 1
# BATCH_SIZE = 64 #32 #64 #32 #64
# VAL_BATCH_SIZE = 64 #32 #64 #32 #64

# MODEL_CHANNELS = 16 #32    # U-Net capacity
# NUM_LAYERS_G = 3 #4 #3 #4       # Depth of g

# LR = 1e-4 #1e-3 #1e-4 #1e-2 #1e-4
# EPOCHS = 10 #20
# EVAL_INTERVAL = 1   # Print samples every N epochs
# EMA_DECAY = 0.9999 #0.999

# # IS_GRADIENT_CLIP = True
# GRADIENT_CLIP_THRESOLD = 3.0 #2.0 #1.0 #10.0

# # Parameter for whether we use the inv_t, and what t will be - the t of the gt, randomly sampled, constant like 0/1/0.5
# INT_T_CONF = 'False' #['False', 'T', 'Random', '1']


# LAMBDA_FM_L2 = 0.0 #1.0 #2.5 #1.0
# LAMBDA_FM_LPIPS = 0.0 #1e-1 #1e-2
# LAMBDA_ISO = 0.0 #1e-4 #1e-3 #1e-1 #1e-3 #0.0 #1e-3 #1e-4 #1e-3 #1e-4 #1e-2
# LAMBDA_FROB = 0.0 #1e-3 #0.0 #1e-3
# LAMBDA_SPEC = 0.0 #1e-3 #1e-1 #1e-3
# LAMBDA_VEL = 0.0 #1e-3
# LAMBDA_TARGET_L2 = 1.0 #0.5 #1e-1 #1e-3 #0.0 #1e-3
# LAMBDA_TARGET_LPIPS = 1e-1 #1e-2 #1e-3 #1e-1 #1e-5

# LAMBDAS_DICT = {
#     'FM_L2': LAMBDA_FM_L2,
#     'FM_LPIPS': LAMBDA_FM_LPIPS,
#     'ISO': LAMBDA_ISO,
#     'FROB': LAMBDA_FROB,
#     'SPEC': LAMBDA_SPEC,
#     'VEL': LAMBDA_VEL,
#     'TARGET_L2': LAMBDA_TARGET_L2,
#     'TARGET_LPIPS': LAMBDA_TARGET_LPIPS
# }

# NUM_SAMPLING_STEPS = 100 # for dicrete and iterative sampling

# NUM_SAMPLES = 16 # how many samples (per type) to show each time during training

# INIT_FACTOR_A = 1e-2 # initialization factor for the noise A is initialized with

# # maybe set name according to configuration

# # hyper-parameters to maybe add:
# # dropout, batch norm, weight init (for A and also for gt)
# # p for the bipitite matching (though try other matching techniques)



