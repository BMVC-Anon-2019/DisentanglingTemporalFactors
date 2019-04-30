import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import torchbearer_modules as tbm
from networks import MLP_Model
import torchbearer as tb
from torchbearer import Trial
from torchbearer.callbacks import MultiStepLR
import math
import state_keys as keys

size_motion = 7
size_shape = 40
nepoch = 300
datapath = './datasets/mnist'
row_size = 12
batch_size = 128
lr = 3e-4

@tb.callbacks.on_sample
@tb.callbacks.on_sample_validation
def gen_target_colour(state):
    def colour_image(img, colour):
        img[:, 0] *= colour[:, 0]
        img[:, 1] *= colour[:, 1]
        img[:, 2] *= colour[:, 2]
        return img

    F1 = state[tb.X].repeat(1,3,1,1)
    target = state[keys.MOVED].repeat(1,3,1,1)

    # Colour target
    colour = torch.rand(F1.shape[0], 3, device=state[tb.DEVICE]).view(F1.shape[0], 3, 1, 1).repeat(1,1,F1.shape[-2],F1.shape[-1])
    target_colour = target.clone()
    target_colour = colour_image(target_colour, colour)
    state[tb.Y_TRUE] = target_colour
    state[tb.Y_TRUE] = torch.clamp(state[tb.Y_TRUE], 0, 1)

    # Colour frame 1
    colour = torch.rand(F1.shape[0], 3, device=state[tb.DEVICE]).view(F1.shape[0], 3, 1, 1).repeat(1,1,F1.shape[-2],F1.shape[-1])
    F1 = colour_image(F1, colour)
    state[tb.X] = F1

@tb.callbacks.on_sample
@tb.callbacks.on_sample_validation
def gen_target_motion(state):
    data = state[tb.X]
    def rot_mat(r):
        mat = torch.eye(3, 3, device=state[tb.DEVICE]).repeat(data.shape[0], 1, 1)
        cos_r = torch.cos(r).squeeze()
        sin_r = torch.sin(r).squeeze()
        mat[:, 0, 0], mat[:, 0, 1], mat[:, 0, 2] = cos_r, sin_r, 1
        mat[:, 1, 0], mat[:, 1, 1], mat[:, 1, 2] = -sin_r, cos_r, 1
        return mat
    def translation_mat(x,y):
        mat = torch.eye(3, 3, device=state[tb.DEVICE]).repeat(data.shape[0], 1, 1)
        mat[:, :, 2] = torch.stack([x,y,torch.ones_like(x)], 1)
        return mat
    def scale_mat(s):
        mat = torch.eye(3, 3, device=state[tb.DEVICE]).repeat(data.shape[0], 1, 1)
        mat[:,0,0], mat[:,1,1] = s[:].squeeze(), s[:].squeeze()
        return mat

    v = (torch.rand(data.shape[0], 2, device=state[tb.DEVICE])*2-1)*0.3
    s = 1+(torch.rand(data.shape[0], 1, device=state[tb.DEVICE])*2-1)*0.3
    r = (torch.rand(data.shape[0], 1, device=state[tb.DEVICE])*2-1)*2*math.pi*0.5*0.5*0.5

    theta = torch.matmul(torch.matmul(torch.transpose(rot_mat(r),1, 2), translation_mat(v[:,0], v[:,1])), scale_mat(s))
    theta = theta[:,:2,:3]
    grid = F.affine_grid(theta, data.shape)
    y = F.grid_sample(data, grid)
    state[keys.MOVED] = y


model = MLP_Model(n_motion=size_motion, n_shape=size_shape, n_channel=3)
optimiser = torch.optim.Adam(model.parameters(), lr=lr)

####### Metrics and Callbacks #######

callback_list = [gen_target_motion, gen_target_colour,
                 tbm.key_kl_loss(1.0, keys.MU, keys.LOGVAR),
                 tbm.key_kl_loss(10.0, keys.MUV, keys.LOGVARV),
                 tbm.visualise_motional_linspace(size_motion, row_size, channels=3),
                 tbm.save_recon(),
                 MultiStepLR([200, 230, 250, 270, 280, 290])
                 ]

####### Datasets #######

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

####### Trainer #######

trial = Trial(model, optimiser, torch.nn.BCELoss(reduction='sum'),
                  metrics=['loss', 'lr'], callbacks=callback_list)
trial.to('cuda')
trial.state[keys.TL2] = trainloader_2
trial.state[keys.VL2] = validationloader_2
trial.with_train_generator(trainloader).with_val_generator(validationloader)
trial.run(nepoch, verbose=2)

