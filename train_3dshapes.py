import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchbearer import Trial
from torchbearer.callbacks import StepLR, MostRecent, TensorBoard
from torchvision import transforms

import state_keys as keys
import torchbearer_modules as tbm
from networks import Shapes3dModel
from shapes3d import Shapes3D

size_motion = 15
size_shape = 5
nepoch = 200
datapath = './datasets/shapes3d/3dshapes.h5'
row_size = 12
batch_size = 32
lr = 3e-4
beta = 10.0


class new_shapes(Shapes3D):
    def __getitem__(self, index):
        img1 = super().__getitem__(index)
        img_class = self.latent_to_int(self.labels[index])[4]
        new_img_latents = []

        for i, ol in enumerate(self.latent_to_int(self.labels[index])):
            diff = torch.randint(0, 3, (1,)).item()
            nl = abs(ol - diff)
            new_img_latents.append(nl)

        new_img_latents[4] = int(img_class)
        try:
            img2_id = self.get_index(new_img_latents)
            img2 = super().__getitem__(img2_id)
        except:
            print(new_img_latents)
            img2 = img1
        return img1, img2


model = Shapes3dModel(n_motion=size_motion, n_shape=size_shape, n_channel=3)
optimiser = torch.optim.Adam(model.parameters(), lr=lr)

####### Metrics and Callbacks #######
saver = tbm.save_recon('imgs/3dshapes_model_recons_15.png')
saver.on_step_training = saver.on_step_validation

viser = tbm.visualise_motional_linspace(size_motion, row_size, file='imgs/vis_motion_15_b{}.png'.format(int(beta)), channels=1)
viser.on_step_training = viser.on_step_validation

callback_list = [
                 tbm.key_kl_loss(1.0, keys.MU, keys.LOGVAR),
                 tbm.key_kl_loss(beta, keys.MUV, keys.LOGVARV),
                 StepLR(75, 0.1),
                 saver, 
                 viser, 
                 MostRecent('saved_models/3dshapes_disentangle_model_15_b{}.pt'.format(int(beta))),
                 ]

####### Datasets #######

from torchbearer.cv_utils import DatasetValidationSplitter
data = new_shapes(datapath, transform=transforms.ToTensor(), in_mem=False)
splitter = DatasetValidationSplitter(len(data), 0.1)
trainset = splitter.get_train_dataset(data)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=7, drop_last=True)
trainloader_2 = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=7, drop_last=True)

####### Trainer #######

trial = Trial(model, optimiser, torch.nn.BCELoss(reduction='sum'),
                  metrics=['loss', 'lr'], callbacks=callback_list)
trial.to('cuda')
trial.state[keys.VL2] = trainloader_2
trial.with_train_generator(trainloader)
trial.run(nepoch, verbose=2)


import disentangle_metric as dm
metrics = dm.DissentangleMetric(data)
print('Higgins Metric')
metrics.get_higgins(size_motion+size_shape, model, mu_key=keys.COMB)
print('Factor Metric')
metrics.get_factor(size_motion+size_shape, model, mu_key=keys.COMB)
