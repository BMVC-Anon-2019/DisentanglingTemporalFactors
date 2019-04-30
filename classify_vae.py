import sys

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import torchbearer as tb
from torchbearer import Trial
from torchbearer.callbacks import *

import state_keys as keys
from networks import VanillaVAE
import torchbearer_modules as tbm

b = float(sys.argv[1])

n_shape = 44
lr = 3e-4
batch_size = 128
n_runs = 5
n_epoch_vae = 300
n_epoch_classifier = 50
datapath = './datasets/mnist'
histories = []


@tb.callbacks.on_sample_validation
@tb.callbacks.on_sample
def gen_target(state):
    state[keys.CLASS] = state[tb.Y_TRUE]
    state[tb.Y_TRUE] = state[tb.X]

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


@tb.callbacks.on_backward
def celoss(state):
    state[tb.LOSS] = state[tb.LOSS] + torch.nn.CrossEntropyLoss()(state[tb.Y_PRED], state[keys.CLASS])


class Classifier(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.classify = torch.nn.Sequential(
            torch.nn.Linear(n_shape, 10),
        )

    def forward(self, x, state):
        mu, logvar = self.model.encode(x.view(x.shape[0], -1))
        x = self.model.reparameterize(mu, logvar)
        return self.classify(x)


for i in range(n_runs):

    model = VanillaVAE(n_shape)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    callback_list = [gen_target, tbm.key_kl_loss(b, keys.MU, keys.LOGVAR),
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

    trial = Trial(model, optimiser, torch.nn.BCELoss(reduction='sum'),
                      metrics=['loss'], callbacks=callback_list)
    trial.to('cuda')
    trial.state[keys.TL2] = trainloader_2
    trial.state[keys.VL2] = validationloader_2
    trial.with_train_generator(trainloader).with_val_generator(validationloader)
    trial.run(n_epoch_vae, verbose=1)

    main_model = model

    model = Classifier(main_model)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    torchbearermodel = Trial(model, optimiser, None,
                      metrics=['loss', CategoricalAccuracy()], callbacks=[gen_target, celoss, MultiStepLR([40, 45])])
    torchbearermodel.to('cuda')
    torchbearermodel.state[keys.TL2] = trainloader_2
    torchbearermodel.state[keys.VL2] = validationloader_2
    torchbearermodel.with_train_generator(trainloader).with_val_generator(validationloader)
    history = torchbearermodel.run(n_epoch_classifier, verbose=1)

    histories.append(history)

torch.save(histories, 'classifier_histories_vae_{}.pt'.format(b))
