import torch
import torchbearer as tb
import state_keys as keys
from torchvision.utils import save_image


# Callback saves a number of images from input batch
def save_recon(save_file='imgs/model_recons.png', row_size=12):
    @tb.callbacks.on_step_validation
    @tb.callbacks.once_per_epoch
    def save_recon_1(state):
        data = state[tb.X]
        target = state[tb.Y_TRUE]
        recon_batch = state[tb.Y_PRED]
        n = min(data.size(0), row_size)
        comparison = torch.cat([data[:n], target[:n],
                                recon_batch.view(data.shape)[:n]])
        state[keys.RECON] = comparison
        save_image(comparison.cpu() ,save_file, nrow=n, pad_value=1)
    return save_recon_1


# KL-Divergence loss callback with multiplicative scaling beta, acting on mu and logvar given as state keys
def key_kl_loss(beta, mu_key, logvar_key):
    @tb.callbacks.add_to_loss
    def kl_loss_1(state):
        mu = state[mu_key]
        logvar = state[logvar_key]

        KLD = -0.5 * torch.sum((1 + logvar - mu.pow(2) - logvar.exp()) * beta)

        return KLD
    return kl_loss_1


# Callback to visualise motional latent space with linear sample of each dimension
def visualise_motional_linspace(size_motion, row_size, file='imgs/vis_motion_space.png', channels=1):
    @tb.callbacks.only_if(lambda state: state[tb.BATCH] == 0)
    @tb.callbacks.on_step_validation
    def test_movement_digit_grid_1(state):
        try:
            img, _ = next(state[keys.VL2I])
        except Exception:
            state[keys.VL2I] = iter(state[keys.VL2])
            img, _ = next(state[keys.VL2I])

        num_images = row_size*size_motion

        # Get first image and repeat to batch
        img = img[0].unsqueeze(0)
        img = img.repeat(num_images,channels,1,1).view(num_images, channels, img.shape[-2], img.shape[-1])
        img = img.to(state[tb.DEVICE])

        # Create set of v vectors
        v = torch.zeros(num_images, size_motion, device=state[tb.DEVICE])
        lin = torch.linspace(-2, 2, row_size, device=state[tb.DEVICE])
        for i in range(size_motion):
            v[(i)*row_size:(i+1)*row_size, i] = lin

        # Get output
        model = state[tb.MODEL]
        out = model.eval_img(img, v)
        comparison = torch.cat([img[:row_size*1], out[:num_images]])
        state[keys.V] = v
        state[keys.VIS] = comparison

        save_image(comparison.cpu(),file, nrow=row_size, pad_value=1)
    return test_movement_digit_grid_1

