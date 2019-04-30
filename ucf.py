import os
import random

from PIL import Image
from torch.utils.data import Dataset


class ucf_ds(Dataset):
    def __init__(self, root, diff=10, transform=None, target_transform=None):
        """
        :param root: Root path to ucf folder with extracted jpgs in UCF_jpgs with same structure as avis
        :param diff: Frame difference to load
        :param transform: Torchvision transform
        :param target_transform: Torchvision transform
        """
        super().__init__()
        self.root = root
        self.jpgs = os.path.join(self.root, 'UCF_jpgs')
        self.diff = diff
        self.videos = self.get_videos()
        self.classes = self.get_classes()
        self.transform = transform
        self.target_transform = target_transform

    def get_videos(self):
        classes = os.listdir(self.jpgs)
        bad_dirs = ['.', '..']
        classes = [c for c in classes if c not in bad_dirs]
        videos = []
        for c in classes:
            class_path = os.path.join(self.jpgs, c)
            vids = os.listdir(class_path)
            for v in vids:
                videos.append(os.path.join(class_path, v))
        return videos

    def get_offset(self):
        pm = round((random.random()*2-1)*self.diff)
        if pm == 0:
            pm = random.random()*2-1
            pm = pm / abs(pm)
        return pm

    def get_classes(self):
        splits = os.path.join(self.root, 'ucfTrainTestlist')
        classes = {}
        for line in open(os.path.join(splits, 'classInd.txt')):
            pair = line.replace('\n', '').split(" ")
            classes[pair[1]] = pair[0]
        return classes

    def __getitem__(self, index):
        vid = self.videos[index]
        class_id = self.classes[os.path.split(os.path.split(vid)[-2])[-1]]
        n_frames = int(open(os.path.join(vid, 'n_frames')).readline())
        frame_1 = random.randint(self.diff+1, n_frames)

        pm = self.get_offset()
        frame_2 = frame_1 + pm if frame_1 + pm < n_frames else frame_1 - pm
        img1path = os.path.join(vid, 'image_{:05d}.jpg'.format(int(frame_1)))
        img2path = os.path.join(vid, 'image_{:05d}.jpg'.format(int(frame_2)))

        img1, img2 = Image.open(img1path), Image.open(img2path)

        if self.transform is not None:
            img1 = self.transform(img1)
        if self.target_transform is not None:
            img2 = self.target_transform(img2)
        return (img1, img2), int(class_id)

    def __len__(self):
        return len(self.videos)
