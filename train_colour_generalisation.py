import colorsys
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torchbearer as tb
from torchbearer import Trial
from torchbearer.callbacks import *

import state_keys as keys
import torchbearer_modules as tbm
from networks import MLP_Model

nepoch = 50
row_size = 12
datapath = './datasets/fashion'
lr = 3e-4
batch_size = 256

n_size_colour = 2
n_size_motion = 2
n_shape = 40


@tb.callbacks.on_sample
@tb.callbacks.on_sample_validation
def store_class(state):
    state[keys.CLASS] = state[tb.Y_TRUE]


def class_to_rand_rgbs(class_, rand_offset=0.3, device='cuda'):
    h = 1/10*class_
    offset = (torch.rand(h.shape[0])*2-1)/10
    h = h + offset
    target_offset = (torch.rand(h.shape[0])*2-1)*rand_offset
    h_target = h + target_offset

    rgb_data = torch.zeros(list(class_.shape)+[3])
    rgb_targ = torch.zeros(list(class_.shape)+[3])

    for i in range(class_.shape[0]):
        rgb_data[i] = torch.tensor(colorsys.hsv_to_rgb(h[i], 1, 255))
        rgb_targ[i] = torch.tensor(colorsys.hsv_to_rgb(h_target[i], 1, 255))

    return rgb_data.to(device), rgb_targ.to(device)


@tb.callbacks.on_sample
@tb.callbacks.on_sample_validation
def gen_colours(state):
    def colour_img(img, colour):
        img[:, 0] *= colour[:, 0]/255
        img[:, 1] *= colour[:, 1] / 255
        img[:, 2] *= colour[:, 2] / 255
        return img

    data = state[tb.X]
    classes = state[keys.CLASS]

    data = data.repeat(1,3,1,1)
    rgb_data, rgb_target = class_to_rand_rgbs(classes.to(torch.float32).cpu(), device=state[tb.DEVICE])
    rgb_data, rgb_target = rgb_data.view(data.shape[0], 3, 1, 1).repeat(1,1,28,28), rgb_target.view(data.shape[0], 3, 1, 1).repeat(1,1,28,28)

    X = data.clone()
    X = colour_img(X, rgb_data)

    target = data.clone()
    target = colour_img(target, rgb_target)

    state[tb.X] = X
    state[tb.Y_TRUE] = target


def visualise_colour_grid(file='imgs/colour_grid.png'):
    @tb.callbacks.once_per_epoch
    @tb.callbacks.on_step_training
    def visualise_colour_grid_1(state):
        img = state[tb.X]
        num_imgs = row_size + row_size**2

        img = img[0].unsqueeze(0)
        img = img.repeat(num_imgs,1,1,1).view(num_imgs, 3, img.shape[-2], img.shape[-1])
        img = img.to(state[tb.DEVICE])

        lin = torch.linspace(-2, 2, row_size, device=state[tb.DEVICE])
        gx, gy = torch.meshgrid([lin, lin])
        gx, gy = gx.contiguous().view(-1), gy.contiguous().view(-1)
        vnew = torch.stack([gx, gy], 1)
        c = torch.zeros(num_imgs, 2, device=state[tb.DEVICE])
        c[:vnew.shape[0], :] = vnew

        model = state[tb.MODEL]
        state[keys.V] = c
        out = model.eval_img(img, c)
        comparison = torch.cat([img[:row_size*1],
                                out[:row_size*row_size]])
        state[keys.VISC] = comparison
        tbm.save_image(comparison.cpu(), file, nrow=row_size, pad_value=1)
    return visualise_colour_grid_1


model = MLP_Model(n_motion=n_size_motion, n_shape=n_shape, n_channel=3)
optimiser = torch.optim.Adam(model.parameters(), lr=lr)


trainloader = DataLoader(
    datasets.FashionMNIST(datapath, train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
validationloader = DataLoader(
    datasets.FashionMNIST(datapath, train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

trainloader_2 = DataLoader(
    datasets.FashionMNIST(datapath, train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
validationloader_2 = DataLoader(
    datasets.FashionMNIST(datapath, train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)


callback_list = [
                store_class,
                gen_colours,
                tbm.key_kl_loss(1.0, keys.MU, keys.LOGVAR), tbm.key_kl_loss(3.0, keys.MUV, keys.LOGVARV),
                MultiStepLR([40,45]),
                tbm.visualise_motional_linspace(n_size_motion, row_size=8, channels=3),
                tbm.save_recon(),
                visualise_colour_grid(),
                ]

trial = Trial(model, optimiser, torch.nn.MSELoss(reduction='sum'),
                  metrics=['loss', 'lr'], callbacks=callback_list)
trial.to('cuda')
trial.state[keys.TL2] = trainloader_2
trial.state[keys.VL2] = validationloader_2
trial.with_train_generator(trainloader).with_val_generator(validationloader)
trial.run(nepoch, verbose=1)

