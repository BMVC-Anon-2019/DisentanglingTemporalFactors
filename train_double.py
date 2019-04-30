import torch.nn.functional as F
import torchbearer as tb
from torch.utils.data import DataLoader
from torchbearer import Trial
from torchbearer.callbacks import *
from torchvision import datasets, transforms
from torchvision.utils import save_image

import state_keys as keys
import torchbearer_modules as tbm
from networks import Double_MLP_Model

row_size = 6
lr = 3e-3
batch_size = 128
nepoch = 300
datapath = './datasets/mnist'

n_size = 4
n_shape = 80

@tb.callbacks.on_sample
@tb.callbacks.on_sample_validation
def double_mnist_target(state):
    def move_image(img, v):
        theta = torch.eye(2, 3).repeat(data.shape[0], 1, 1).to(state[tb.DEVICE])
        theta[:, :, 2] = v[:, :2]
        grid = F.affine_grid(theta, data.shape)
        y = F.grid_sample(img, grid)
        return y

    data = state[tb.X]
    v = (torch.rand(data.shape[0], 4, device=state[tb.DEVICE])*2-1)*0.5

    ####### Image 1
    data = data.repeat(1, 3, 1, 1)
    state[keys.V] = v
    im1 = move_image(data, v[:, :2])

    ####### Image 2
    try:
        data2, _ = next(state[keys.TL2I])
    except Exception:
        state[keys.TL2I] = iter(state[keys.TL2])
        data2, _ = next(state[keys.TL2I])

    data2 = data2.to(data.device)
    data2 = data2.repeat(1, 3, 1, 1)
    im2 = move_image(data2, v[:, 2:])

    ####### Final
    img = torch.zeros(data.shape[0], 3, data.shape[-2], 2*data.shape[-1]).to(data.device)
    target = torch.zeros(data.shape[0], 3, data.shape[-2], 2 * data.shape[-1]).to(data.device)
    img[:,:,:,:data.shape[-1]] = data
    img[:, :, :, data.shape[-1]:] = data2
    target[:,:,:,:data.shape[-1]] = im1
    target[:, :, :, data.shape[-1]:] = im2
    state[tb.X] = img.to(data.device)
    state[tb.Y_TRUE] = target.to(data.device)


def visualise_linspace(n_size, row_size, file='imgs/linspace.png', channels=1):
    @tb.callbacks.only_if(lambda state: state[tb.BATCH] == 0)
    @tb.callbacks.on_step_validation
    def test_movement_digit_grid_1(state):
        try:
            img1, _ = next(state[keys.VL2I])
            img2, _ = next(state[keys.VL2I])
        except Exception:
            state[keys.VL2I] = iter(state[keys.VL2])
            img1, _ = next(state[keys.VL2I])
            img2, _ = next(state[keys.VL2I])

        num_imgs = row_size*n_size

        # Repeat image 1 and 2 for batch
        img1 = img1[0].unsqueeze(0)
        img1 = img1.repeat(num_imgs,channels,1,1).view(num_imgs, channels, img1.shape[-2], img1.shape[-1])
        img1 = img1.to(state[tb.DEVICE])
        img2 = img2[0].unsqueeze(0)
        img2 = img2.repeat(num_imgs,channels,1,1).view(num_imgs, channels, img1.shape[-2], img1.shape[-1])
        img2 = img2.to(state[tb.DEVICE])

        # Spatially concat images
        img = torch.zeros(num_imgs, 3, img1.shape[-2], 2 * img1.shape[-1]).to(img1.device)
        img[:, :, :, :img1.shape[-1]] = img1
        img[:, :, :, img1.shape[-1]:] = img2

        # Create v for each batch image
        v = torch.zeros(num_imgs, n_size, device=state[tb.DEVICE])
        lin = torch.linspace(-1, 1, row_size, device=state[tb.DEVICE])
        for i in range(n_size):
            v[(i)*row_size:(i+1)*row_size, i] = lin

        # Output
        state[keys.V] = v
        model = state[tb.MODEL]
        out = model.eval_img(img, v)
        data = img
        comparison = torch.cat([data[:row_size*1],
                                out[:n_size*row_size]])
        state[keys.VIS] = comparison
        save_image(comparison.cpu(),file, nrow=row_size, pad_value=1)
    return test_movement_digit_grid_1


model = Double_MLP_Model(n_feat=n_size, n_shape=n_shape, n_channel=3)
optimiser = torch.optim.Adam(model.parameters(), lr=lr)

callback_list = [double_mnist_target,
                 tbm.key_kl_loss(1.0, keys.MU, keys.LOGVAR), tbm.key_kl_loss(10.0, keys.MUV, keys.LOGVARV),
                 tbm.save_recon('imgs/recon_double.png', row_size=row_size),
                 visualise_linspace(n_size, file='imgs/linspace_double.png', row_size=row_size, channels=3),
                 MultiStepLR([200, 250, 280, 290])
                 ]

trainloader = DataLoader(
    datasets.MNIST(datapath, train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
validationloader = DataLoader(
    datasets.MNIST(datapath, train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

trainloader_2 = DataLoader(
    datasets.MNIST(datapath, train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
validationloader_2 = DataLoader(
    datasets.MNIST(datapath, train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)


trial = Trial(model, optimiser, torch.nn.BCELoss(),
              metrics=['loss', 'lr'], callbacks=callback_list)
trial.to('cuda')
trial.state[keys.TL2] = trainloader_2
trial.state[keys.VL2] = validationloader_2
trial.with_train_generator(trainloader).with_val_generator(validationloader)
trial.run(nepoch, verbose=1)

