import numpy as np
import torch
import torchbearer as tb
import tqdm
from torch.utils.data import Dataset, DataLoader
from torchbearer import Trial


class HigginsMetric(Dataset):
    def __init__(self, sprites, model, mu_key):
        super().__init__()
        self.ds = sprites
        self.model = model
        self.MU = mu_key

    def _get_random_latent(self):
        f = []
        for factor in self.ds.latents_sizes:
            f.append(np.random.randint(0, factor))
        return np.array(f)

    def difference(self, bs, K):
        with torch.no_grad():

            K_value = np.random.randint(0, self.ds.latents_sizes[K])
            diffs = []
            for i in range(bs):
                latent_1 = self._get_random_latent()
                latent_2 = self._get_random_latent()
                latent_11 = self._get_random_latent()
                latent_21 = self._get_random_latent()

                latent_1[K] = K_value
                latent_2[K] = K_value
                latent_11[K] = K_value
                latent_21[K] = K_value
                latent_21[1] = latent_11[1]

                img1 = self.ds.get_img_by_latent(latent_1).to('cuda').unsqueeze(0)
                img2 = self.ds.get_img_by_latent(latent_2).to('cuda').unsqueeze(0)
                img11 = self.ds.get_img_by_latent(latent_11).to('cuda').unsqueeze(0)
                img21 = self.ds.get_img_by_latent(latent_21).to('cuda').unsqueeze(0)

                state = {tb.Y_TRUE: img11}
                self.model(img1, state)
                z1 = state[self.MU]
                state = {tb.Y_TRUE: img21}
                self.model(img2, state)
                z2 = state[self.MU]

                diffs.append(torch.abs(z1 - z2))

            diffs = tuple(diffs)
            return torch.mean(torch.cat(diffs), 0)

    def __getitem__(self, index):
        rand_K = np.random.randint(2, len(self.ds.latents_sizes))
        return self.difference(10, rand_K), rand_K

    def __len__(self):
        return 1000


class FactorMetric(Dataset):
    def __init__(self, sprites, model, nlatents=10, mu_key=None, num_stats=100):
        super().__init__()
        self.ds = sprites
        self.model = model
        self.nlatents = nlatents
        self.mu_key = mu_key
        self.num_stats = num_stats
        self.norm_stats = self.get_dim_stats()

    def _get_random_latent(self):
        f = []
        for factor in self.ds.latents_sizes:
            f.append(np.random.randint(0, factor))
        return np.array(f)

    def get_dim_stats(self):
        with torch.no_grad():
            state = {}
            z_list = []
            loader = iter(DataLoader(self.ds, 128, shuffle=True))

            tloader = tqdm.tqdm(loader)
            for i, data in enumerate(tloader):
                img1, img2 = data
                state[tb.Y_TRUE] = img2.cuda()
                self.model(img1.cuda(), state)
                z_list.append(state[self.mu_key])
                if i > self.num_stats:
                    break

        return torch.cat(z_list).std(0)

    def difference(self, bs, K):
        with torch.no_grad():
            K_value = np.random.randint(0, self.ds.latents_sizes[K])
            diffs = []
            for i in range(bs):
                latent_1 = self._get_random_latent()
                latent_1[K] = K_value

                latent_2 = self._get_random_latent()
                latent_2[1] = latent_1[1]
                latent_2[K] = K_value

                img1 = self.ds.get_img_by_latent(latent_1).to('cuda').unsqueeze(0)
                img2 = self.ds.get_img_by_latent(latent_2).to('cuda').unsqueeze(0)

                state = {tb.Y_TRUE: img2}
                self.model(img1, state)
                z1 = state[self.mu_key]

                z1 = z1 / self.norm_stats
                diffs.append(z1)

            amin = torch.var(torch.cat(diffs, 0), 0).argmin().view(1,1)
            y_onehot = self.to_one_hot(amin, self.nlatents)
            return y_onehot

    def to_one_hot(self, y, n_dims=None):
        """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims.
        Method by justheuristic on github here: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/23
        """
        y_tensor = y
        n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
        y_one_hot = torch.zeros(y_tensor.shape[0], n_dims, device=y_tensor.device).scatter_(1, y_tensor, 1)
        y_one_hot = y_one_hot.view(y.shape[0], -1)
        return y_one_hot

    def __getitem__(self, index):
        rand_K = np.random.randint(2, len(self.ds.latents_sizes))
        return self.difference(10, rand_K), rand_K

    def __len__(self):
        return 1000


@tb.callbacks.on_forward
@tb.callbacks.on_forward_validation
def flatt(state):
    state[tb.Y_PRED] = state[tb.Y_PRED].view(-1, 6)


class DissentangleMetric:
    def __init__(self, dsprites):
        super().__init__()
        self.data = dsprites

    def get_higgins(self, latents, model, mu_key, epochs=100):
        classifier = torch.nn.Linear(latents, 6)
        optimiser = torch.optim.Adam(classifier.parameters(), lr=0.01)

        dataset = HigginsMetric(self.data, model, mu_key)
        trainloader = DataLoader(dataset, batch_size=2)

        trial = Trial(classifier, optimiser, torch.nn.CrossEntropyLoss(), metrics=['acc'], callbacks=[flatt, ])
        trial.with_train_generator(trainloader).cuda()
        l = trial.run(epochs, verbose=1)

    def get_factor(self, latents, model, mu_key, epochs=100):
        classifier = torch.nn.Linear(latents, 6)
        optimiser = torch.optim.Adam(classifier.parameters(), lr=0.01)

        dataset = FactorMetric(self.data, model, nlatents=latents, mu_key=mu_key)
        trainloader = DataLoader(dataset, batch_size=2)

        trial = Trial(classifier, optimiser, torch.nn.CrossEntropyLoss(), metrics=['acc'], callbacks=[flatt, ])
        trial.with_train_generator(trainloader).cuda()
        l = trial.run(epochs, verbose=1)
