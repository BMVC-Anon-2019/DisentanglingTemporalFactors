import h5py
import torch
from PIL import Image
from torch.utils.data import Dataset


class Shapes3D(Dataset):
    def __init__(self, root, transform=None, in_mem=True):
        """Class for 3D Shapes dataset here: https://github.com/deepmind/3d-shapes.
        Modified from here: https://github.com/deepmind/3d-shapes/blob/master/3dshapes_loading_example.ipynb

        :param root: Path to 3dshapes.h5 file
        :param transform: Torchvision transform to apply to loaded image
        :param in_mem: If True, load entire dataset into memory. If False load from h5 file as needed.
        """
        super().__init__()
        self.root = root
        self.images = None
        self.labels = None
        self.image_shape = None
        self.label_shape = None
        self.n_samples = None
        self._FACTORS_IN_ORDER = None
        self._NUM_VALUES_PER_FACTOR = None
        self.linspaces = None
        self.in_mem = in_mem
        self.transform = transform
        self.load_meta()

    def load_meta(self):
        self.images = h5py.File(self.root, 'r')['images']
        self.labels = h5py.File(self.root, 'r')['labels']
        if self.in_mem:
            self.images = self.images[:]
            self.labels = self.labels[:]
        self.image_shape = self.images.shape[1:]
        self.label_shape = self.labels.shape[1:]
        self.n_samples = self.labels.shape[0]

        self._FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
        self._NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                                  'scale': 8, 'shape': 4, 'orientation': 15}

        self.latents_sizes = [10, 10, 10, 8, 4, 15]

        self.linspaces = {
            'floor_hue': torch.linspace(0, 1, self._NUM_VALUES_PER_FACTOR['floor_hue']+1),
            'wall_hue': torch.linspace(0, 1, self._NUM_VALUES_PER_FACTOR['wall_hue']+1),
            'object_hue': torch.linspace(0, 1, self._NUM_VALUES_PER_FACTOR['object_hue']+1),
            'scale': torch.linspace(0.75, 1.25, self._NUM_VALUES_PER_FACTOR['scale']+1),
            'shape': torch.linspace(0, 3, self._NUM_VALUES_PER_FACTOR['shape']+1),
            'orientation': torch.linspace(-30, 30, self._NUM_VALUES_PER_FACTOR['orientation']+1)
        }

    def get_img_by_latent(self, latent):
        id = self.get_index(latent)
        return self.__getitem__(id)[0]

    def get_index(self, factors):
        """ Converts factors to indices in range(num_data)
        Args:
          factors: np array shape [6,batch_size].
                   factors[i]=factors[i,:] takes integer values in
                   range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[i]]).

        Returns:
          indices: np array shape [batch_size].
        """
        indices = 0
        base = 1
        for factor, name in reversed(list(enumerate(self._FACTORS_IN_ORDER))):
            indices += factors[factor] * base
            base *= self._NUM_VALUES_PER_FACTOR[name]
        return indices

    def latent_to_int(self, latent_code):
        codes = []
        for i, k in enumerate(self.linspaces):
            if i != 4:
                split = self.linspaces[k][1]-self.linspaces[k][0]
                start = self.linspaces[k][0]/split
                code = latent_code[i] / split - start
                codes.append(int(code.item()))
            else:
                codes.append(latent_code[i])
        return codes

    def __getitem__(self, index):
        img = Image.fromarray(self.images[index])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return self.n_samples
