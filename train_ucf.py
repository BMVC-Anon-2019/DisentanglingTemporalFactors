import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchbearer import Trial
from torchbearer.callbacks import StepLR, MostRecent
from torchvision import transforms

import state_keys as keys
import torchbearer_modules as tbm
from networks import UcfModel
from ucf import ucf_ds

size_motion = 8
size_shape = 120
nepoch = 300
datapath = './datasets/ucf101'
row_size = 12
batch_size = 8
lr = 3e-4


class BasicUCF(ucf_ds):
    def __getitem__(self, index):
        (img1, img2), class_id = super().__getitem__(index)
        return img1, img2


model = UcfModel(n_motion=size_motion, n_shape=size_shape, n_channel=3)

optimiser = torch.optim.Adam(model.parameters(), lr=lr)

####### Metrics and Callbacks #######

callback_list = [
                 tbm.key_kl_loss(1.0, keys.MU, keys.LOGVAR),
                 tbm.key_kl_loss(4.0, keys.MUV, keys.LOGVARV),
                 tbm.save_recon('imgs/ucf_recon_b4.png'),
                 StepLR(75, 0.5),
                 MostRecent('saved_models/ucf_model.pt'),
                 tbm.visualise_motional_linspace(size_motion, row_size, file='imgs/ucf_vis.png', channels=1)
]


####### Datasets #######

trans = transforms.Compose([
    transforms.CenterCrop((240, 320)),
    transforms.ToTensor()
])

from torchbearer.cv_utils import DatasetValidationSplitter
data = BasicUCF(datapath, transform=trans, target_transform=trans)
splitter = DatasetValidationSplitter(len(data), 0.1)
trainset, valset = splitter.get_train_dataset(data), splitter.get_val_dataset(data)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
validationloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
validationloader_2 = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

####### Trainer #######

trial = Trial(model, optimiser, torch.nn.BCELoss(reduction='sum'),
                  metrics=['loss', 'lr'], callbacks=callback_list)
trial.to('cuda')
trial.state[keys.VL2] = validationloader_2
trial.with_train_generator(trainloader).with_val_generator(validationloader)
trial.run(350, verbose=2)

