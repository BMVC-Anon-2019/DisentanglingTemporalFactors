import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchbearer import Trial
from torchbearer.callbacks import MultiStepLR, MostRecent
from torchvision import transforms

import state_keys as keys
import torchbearer_modules as tbm
from networks import ChairsCNNModel


class Chairs3D(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform

        self.classes_path = os.path.join(self.root, 'jpgs')
        self.latent_values = {'p': [20, 30], 't': np.floor(np.linspace(0, 360, 32))[:-1]}

        self.names = self.get_names()

    def get_names(self):
        names = os.listdir(self.classes_path)
        try:
            names.remove('all_chair_names.mat')
        except:
            print('not found')
        return names

    def get_image_pair(self, chair_name):
        chair_path = os.path.join(self.classes_path, chair_name, 'renders')
        imgs = os.listdir(chair_path)
        im1_id, im2_id = np.random.randint(0, len(imgs)), np.random.randint(0, len(imgs))
        im1_path, im2_path = os.path.join(chair_path, imgs[im1_id]), os.path.join(chair_path, imgs[im2_id])
        im1, im2 = Image.open(im1_path), Image.open(im2_path)
        return im1, im2

    def __getitem__(self, index):
        chair_name = self.names[index]
        im1, im2 = self.get_image_pair(chair_name)

        if self.transform is not None:
            im1 = self.transform(im1)
            im2 = self.transform(im2)

        return im1, im2

    def __len__(self):
        return len(self.names)


size_motion = 1
size_shape = 80
nepoch = 300
datapath = './datasets/3dchairs'
row_size = 12
batch_size = 8
lr = 3e-4


model = ChairsCNNModel(n_motion=size_motion, n_shape=size_shape, n_channel=1)
optimiser = torch.optim.Adam(model.parameters(), lr=lr)

####### Metrics and Callbacks #######

callback_list = [
                    tbm.key_kl_loss(1.0, keys.MU, keys.LOGVAR),
                    tbm.key_kl_loss(50.0, keys.MUV, keys.LOGVARV),
                    tbm.visualise_motional_linspace(size_motion, row_size, file='imgs/chairs_vis.png', channels=1),
                    tbm.save_recon('imgs/chairs_recons.png'),
                    MultiStepLR([100,250,295]),
                    MostRecent('saved_models/chairs_model.pt'),
                ]

####### Datasets #######

from torchbearer.cv_utils import DatasetValidationSplitter
data = Chairs3D(datapath, transform=transforms.Compose([transforms.Grayscale(1), transforms.ToTensor(),]))
splitter = DatasetValidationSplitter(len(data), 0.1)
trainset, valset = splitter.get_train_dataset(data), splitter.get_val_dataset(data)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
validationloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)
trainloader_2 = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
validationloader_2 = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)

####### Trainer #######

trial = Trial(model, optimiser, torch.nn.BCELoss(reduction='sum'), metrics=['loss', 'lr'], callbacks=callback_list)
trial.to('cuda')
trial.state[keys.TL2] = trainloader_2
trial.state[keys.VL2] = validationloader_2
trial.with_train_generator(trainloader).with_val_generator(validationloader)
trial.with_inf_train_loader()
trial.run(nepoch, verbose=2)



