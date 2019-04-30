import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import torchbearer as tb
from torchbearer import Trial
from torchbearer.callbacks import MultiStepLR

import torchbearer_modules as tbm
import state_keys as keys
from networks import MLP_Model

import math
import sys

b = float(sys.argv[1])

n_size = 4
n_shape = 40
lr = 3e-4
batch_size = 128
n_runs = 5
n_epoch_vae = 300
n_epoch_classifier = 50
datapath = './datasets/mnist'


@tb.callbacks.on_sample
@tb.callbacks.on_sample_validation
def gen_target(state):
    data = state[tb.X]
    state[keys.CLASS] = state[tb.Y_TRUE]
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

    v = (torch.rand(data.shape[0], 2, device=state[tb.DEVICE])*2-1)*0.5
    s = 1+(torch.rand(data.shape[0], 1, device=state[tb.DEVICE])*2-1)*0.3
    r = (torch.rand(data.shape[0], 1, device=state[tb.DEVICE])*2-1)*2*math.pi*0.5*0.5*0.5

    state[keys.V] = v

    theta = torch.matmul(torch.matmul(torch.transpose(rot_mat(r),1, 2), translation_mat(v[:,0], v[:,1])), scale_mat(s))
    theta = theta[:,:2,:3]
    grid = F.affine_grid(theta, data.shape)
    y = F.grid_sample(data, grid)
    state[tb.Y_TRUE] = y.clamp(0,1)

@tb.callbacks.on_criterion
@tb.callbacks.on_criterion_validation
def celoss(state):
    state[tb.LOSS] = state[tb.LOSS] + torch.nn.CrossEntropyLoss()(state[tb.Y_PRED], state[keys.CLASS])


class ClassifierNoV(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.classify = torch.nn.Sequential(
            torch.nn.Linear(n_shape, 10),
        )

    def forward(self, x, state):
        mu, logvar = self.model.encode_shape(x.view(x.shape[0], -1))
        x = self.model.reparameterize(mu, logvar)
        return self.classify(x)


class ClassifierV(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.classify = torch.nn.Sequential(
            torch.nn.Linear(n_size, 10),
        )

    def forward(self, x, state):
        F1, F2 = x, state[tb.Y_TRUE]
        in_motion = torch.cat([F1, F2], 1).view(F1.shape[0], -1)
        mu, lv = self.model.encode_motion(in_motion)
        v = self.model.reparameterize(mu, lv)
        return self.classify(v)


class ClassifierCombined(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.classify = torch.nn.Sequential(
            torch.nn.Linear(n_size+n_shape, 10),
        )

    def forward(self, x, state):
        F1, F2 = x, state[tb.Y_TRUE]
        in_motion = torch.cat([F1, F2], 1).view(F1.shape[0], -1)
        mu, logvar = self.model.encode_motion(in_motion)
        v = self.model.reparameterize(mu, logvar)

        mu, logvar = self.model.encode_shape(x.view(x.shape[0], -1))
        z = self.model.reparameterize(mu, logvar)
        tangle = torch.cat([z,v], 1)
        return self.classify(tangle)


@tb.metrics.default_for_key('cat_acc')
@tb.metrics.running_mean
@tb.metrics.std
@tb.metrics.mean
class CategoricalAccuracy(tb.metrics.Metric):
    """Categorical accuracy metric. Uses torch.max to determine predictions and compares to targets. Decorated with a
    mean, running_mean and std. Default for key: 'cat_acc'

    :param ignore_index: Specifies a target value that is ignored and does not contribute to the metric output. See `<https://pytorch.org/docs/stable/nn.html#crossentropyloss>`_
    :type ignore_index: int
    """

    def __init__(self, ignore_index=-100):
        super().__init__('acc')

        self.ignore_index = ignore_index

    def process(self, *args):
        state = args[0]
        y_pred = state[tb.Y_PRED]
        y_true = state[keys.CLASS]
        mask = y_true.eq(self.ignore_index).eq(0)
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        _, y_pred = torch.max(y_pred, 1)
        return (y_pred == y_true).float()



histories = []
for i in range(n_runs):
    classifiers_hist = {}
    model = MLP_Model(n_motion=n_size, n_shape=n_shape)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    callback_list = [gen_target,
                     tbm.key_kl_loss(1.0, keys.MU, keys.LOGVAR),
                     tbm.key_kl_loss(b, keys.MUV, keys.LOGVARV),
                     tbm.visualise_motional_linspace(n_size, row_size=8),
                     tbm.save_recon(), MultiStepLR([200, 250, 275, ])]

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
                      metrics=['loss'], callbacks=callback_list)
    trial.to('cuda')
    trial.state[keys.TL2] = trainloader_2
    trial.state[keys.VL2] = validationloader_2
    trial.with_train_generator(trainloader).with_val_generator(validationloader)
    trial.run(n_epoch_vae, verbose=1)

    main_model = model

    for p in main_model.parameters():
        p.requires_grad = False

    model = ClassifierV(main_model)
    optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    torchbearermodel = Trial(model, optimiser, None,
                      metrics=['loss', CategoricalAccuracy()], callbacks=[gen_target, celoss])
    torchbearermodel.to('cuda')
    torchbearermodel.state[keys.TL2] = trainloader_2
    torchbearermodel.state[keys.VL2] = validationloader_2
    torchbearermodel.with_train_generator(trainloader).with_val_generator(validationloader)
    h = torchbearermodel.run(n_epoch_classifier, verbose=1)
    classifiers_hist['V'] = h

    model = ClassifierNoV(main_model)
    optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    torchbearermodel = Trial(model, optimiser, None,
                      metrics=['loss', CategoricalAccuracy()], callbacks=[gen_target, celoss])
    torchbearermodel.to('cuda')
    torchbearermodel.state[keys.TL2] = trainloader_2
    torchbearermodel.state[keys.VL2] = validationloader_2
    torchbearermodel.with_train_generator(trainloader).with_val_generator(validationloader)
    h = torchbearermodel.run(n_epoch_classifier, verbose=1)
    classifiers_hist['NoV'] = h

    model = ClassifierCombined(main_model)
    optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    torchbearermodel = Trial(model, optimiser, None,
                      metrics=['loss', CategoricalAccuracy()], callbacks=[gen_target, celoss])
    torchbearermodel.to('cuda')
    torchbearermodel.state[keys.TL2] = trainloader_2
    torchbearermodel.state[keys.VL2] = validationloader_2
    torchbearermodel.with_train_generator(trainloader).with_val_generator(validationloader)
    h = torchbearermodel.run(n_epoch_classifier, verbose=1)
    classifiers_hist['Comb'] = h

    histories.append(classifiers_hist)

torch.save(histories, 'classify_our_histories_{}.pt'.format(b))
