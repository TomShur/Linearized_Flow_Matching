import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from linearized_flow_matching.core.model_architectures import TimeG_FlowMatcher
from linearized_flow_matching.configs.config import config

NUM_SAMPLING_STEPS = config['NUM_SAMPLING_STEPS']
NUM_SAMPLES = config['NUM_SAMPLES']
IMG_SIZE = config['IMG_SIZE']
IN_CHANNELS = config['IN_CHANNELS']


def show_samples(samples, title="Generated Samples"):
    samples = samples.detach().cpu().clamp(0, 1)
    grid_size = int(len(samples)**0.5)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))
    fig.suptitle(title)

    for i, ax in enumerate(axes.flat):
        if i < len(samples):
            ax.imshow(samples[i].permute(1, 2, 0).squeeze(), cmap='gray')
            ax.axis('off')
    plt.show()





def sample_and_show(model, epoch, batch_precent, wandb_run, device, model_type_string="RAW", num_steps=NUM_SAMPLING_STEPS, num_samples=NUM_SAMPLES, img_size=IMG_SIZE, channels=IN_CHANNELS):
    print(f'Epoch: {epoch+1} - Batch Percent: {batch_precent} - Model Type: {model_type_string}')

    model.eval()
    model_matcher = TimeG_FlowMatcher(model, wandb_run=wandb_run)

    with torch.no_grad(): # Generate samples from each of the 3 sampling types
        samples_exp, samples_discrete_collapsed, samples_iter = model_matcher.sample(
            num_samples=num_samples,
            img_size=img_size,
            channels=channels,
            num_steps=num_steps,
            device=device
        )

    title_base = f'Epoch {epoch+1} - {model_type_string}'

    show_samples(samples_exp, title=f'{title_base} - Exact Matrix Exp')
    show_samples(samples_discrete_collapsed, title=f'{title_base} - Discrete Collapsed (Power)')
    show_samples(samples_iter, title=f'{title_base} - Iterative Euler (Loop)')

    # Log to W&B
    # We create grids for cleaner logging, but can also log individual lists
    if wandb_run is not None:
        # Create normalized grids
        grid_exp = vutils.make_grid(samples_exp.clamp(-1, 1), normalize=True, nrow=4)
        grid_col = vutils.make_grid(samples_discrete_collapsed.clamp(-1, 1), normalize=True, nrow=4)
        grid_iter = vutils.make_grid(samples_iter.clamp(-1, 1), normalize=True, nrow=4)

        wandb_run.log({
            f"samples/{model_type_string}_1_Exact_Exp": wandb_run.Image(grid_exp, caption=f"{model_type_string} EXP"),
            f"samples/{model_type_string}_2_Discrete_Collapsed": wandb_run.Image(grid_col, caption=f"{model_type_string} Collapsed"),
            f"samples/{model_type_string}_3_Iterative_Euler": wandb_run.Image(grid_iter, caption=f"{model_type_string} Iterative"),
            "epoch": epoch + 1
        }, step=epoch+1)

    model.train()

    return samples_exp, samples_discrete_collapsed, samples_iter