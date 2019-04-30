import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchbearer import Trial
from torchbearer.callbacks import StepLR, MostRecent
from torchbearer.variational.datasets import dSprites
from torchvision import transforms

import state_keys as keys
import torchbearer_modules as tbm
from networks import dSprites_MLP_Model

size_motion = 5
size_shape = 5
nepoch = 200 
datapath = './datasets/dsprites'
row_size = 12
batch_size = 256
lr = 3e-4
beta = 8.0


class TempDSprites(dSprites):
    def set_diff(self, max_diff):
        self.max_diff = max_diff

    def __getitem__(self, index):
        img1 = super().__getitem__(index)[0]
        img_class = self.latents_classes[index][1]
        new_img_latents = [0]

        for i, ol in enumerate(self.latents_classes[index]):
            if i == 0:
                continue
            if i == 2:
                diff = torch.randint(2, (1,))
            else:
                diff = torch.randint(self.max_diff, (1,))

            if torch.rand(1) < 0.5:
                nl = (ol - diff).abs().item()
            else:
                nl = ((ol + diff).clamp(0, self.latents_sizes[i]-1)).abs().item()
            new_img_latents.append(nl)

        new_img_latents[1] = int(img_class)
        img2_id = np.dot(new_img_latents, self.latents_bases).astype(int)
        img2 = super().__getitem__(img2_id)[0]
        return img1, img2


model = dSprites_MLP_Model(n_motion=size_motion, n_shape=size_shape, n_channel=1)
optimiser = torch.optim.Adam(model.parameters(), lr=lr)


####### Datasets #######

from torchbearer.cv_utils import DatasetValidationSplitter
data = TempDSprites(datapath, download=True, transform=transforms.ToTensor())
data.set_diff(10)
splitter = DatasetValidationSplitter(len(data), 0.1)
trainset, valset = splitter.get_train_dataset(data), splitter.get_val_dataset(data)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
validationloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
validationloader_2 = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)


####### Metrics and Callbacks #######

callback_list = [
    tbm.key_kl_loss(1.0, keys.MU, keys.LOGVAR),
    tbm.key_kl_loss(beta, keys.MUV, keys.LOGVARV),
    tbm.visualise_motional_linspace(size_motion, 8, file='imgs/sprites_vis.png', channels=1),
    tbm.save_recon('imgs/model_recons_b{}.png'.format(int(beta))),
    StepLR(75, 0.5),
    MostRecent('saved_models/sprites_disentangle_model_b{}.pt'.format(int(beta))),
]


####### Trainer #######

trial = Trial(model, optimiser, torch.nn.BCELoss(reduction='sum'), metrics=['loss', 'lr'], callbacks=callback_list)
trial.to('cuda')
trial.state[keys.VL2] = validationloader_2
trial.with_train_generator(trainloader).with_val_generator(validationloader)
trial.run(nepoch, verbose=1)


import disentangle_metric as dm
metrics = dm.DissentangleMetric(data)
print('Higgins Metric')
metrics.get_higgins(size_motion+size_shape, model, keys.COMB)
print('Factor Metric')
metrics.get_factor(size_motion+size_shape, model, keys.COMB)