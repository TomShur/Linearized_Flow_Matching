import torch
import torch.nn as nn

class IsometryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, g_x, g_0):
        """
        Calculates L_isometry = | ||g(x) - g(0)||^2 - ||x||^2 |

        Args:
            x: Input data [B, C, H, W]
            g_x: Latent representation of x [B, C, H, W]
            g_0: Latent representation of zero [B, C, H, W] (or [1, C, H, W])
        """
        # Flatten to [B, D] for norm calculation
        x_flat = x.view(x.shape[0], -1)
        g_x_flat = g_x.view(x.shape[0], -1)
        g_0_flat = g_0.view(g_0.shape[0], -1) # Handle broadcasting if g_0 is single sample

        # Calculate squared norms
        # Norm of input x
        norm_x_sq = torch.sum(x_flat ** 2, dim=1)

        # Norm of displacement in latent space (g(x) - g(0))
        # Note: g_0 might need broadcasting if it's calculated once per batch or once globally
        diff = g_x_flat - g_0_flat
        norm_g_diff_sq = torch.sum(diff ** 2, dim=1)

        # The Loss: Absolute difference between the squared norms
        loss = torch.abs(norm_g_diff_sq - norm_x_sq).mean()

        return loss
