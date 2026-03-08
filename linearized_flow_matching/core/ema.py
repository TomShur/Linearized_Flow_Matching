import copy
import torch
from linearized_flow_matching.configs.config import CONFIG_DICT

EMA_DECAY = CONFIG_DICT['ema_decay']

class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval() # EMA model is always in eval mode

        for param in self.model.parameters(): # We don't want to optimize the EMA parameters
            param.requires_grad = False

    def update(self, model): # Updates the EMA model weights based on the current model.
        with torch.no_grad():
            for ema_param, param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def get_model(self):
        return self.model