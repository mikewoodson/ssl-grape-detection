from torchvision.datasets.folder import pil_loader, accimage_loader, default_loader
from pathlib import Path
from torchvision import transforms as T
from helpers.annotation_utils import convert_bbox_format, bbox_to_imgaug, bbox_to_tensor

import os
import numpy as np
import pdb
import functools
import json

import torch.utils.data as data
import torch


class Cr2(data.Dataset):
    """ Cr2 dataset
    Args:
        root (string): Root directory path to dataset.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g. ``transforms.RandomCrop``
        loader (callable, optional): A function to load an image given its path.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in the root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'https://github.com/MPBA/CR2.git'
    splits = ('test')

    def __init__(self, root, split='test', transform=None,
                 loader=default_loader, download=False):
        if split not in self.splits:
            raise ValueError(
                'Split "{}" not found. Valid splits are: {}'.format(
                    split, ', '.join(
                        self.splits), ))
        self.root = Path(root)
        self.image_dir = self.root / 'images'
        self.annotations_file = self.root / 'annotations.json'
        self.split = split
        self.transform = transform

        if download:
            self.download()

        self.loader = loader
        self.id_to_fname = {}

        self.samples = []
        self.annotations = {}
        self.create_dataset()

    def create_dataset(self):
        image_paths = self.image_dir.glob('*.jpg')

        with self.annotations_file.open() as f:
            annotation_list = json.loads(f.read())

        for img in annotation_list:
            img_name = Path(img['filename']).parts[-1]
            points = img['annotations']
            dots_tensor = torch.zeros([len(points),2], dtype=torch.int32)
            for idx, point in enumerate(points):
                dots_tensor[idx] = torch.tensor([point['x'], point['y']])
            self.annotations[img_name] = dots_tensor

        # Read bbox annotations from file
        for idx, img_path in enumerate(image_paths):
            target = {}
            img_name = img_path.parts[-1]
            img_path = str(img_path)

            image_id = torch.tensor([idx])
            self.id_to_fname[image_id.item()] = img_name
            target['image_id'] = image_id
            target['dots'] = self.annotations[img_name]

            self.samples.append((img_path, target))

    @functools.cached_property
    def mean(self):
        n_pixels = 0
        pix_sum = torch.zeros([3])
        for img_path, _ in self.samples:
            img = self.loader(img_path)
            w,h = img.size
            im_tensor = T.ToTensor()(img)
            pix_sum += im_tensor.sum([1,2])
            n_pixels += (w*h)
        pix_avg = pix_sum / n_pixels
        return pix_avg.squeeze().tolist()

    @functools.cached_property
    def stddev(self):
        avg = torch.tensor(self.mean).reshape([3, 1, 1])
        var_sum = torch.zeros([3])
        n_pixels = 0
        for img_path, _ in self.samples:
            img = self.loader(img_path)
            w,h = img.size
            im_tensor = T.ToTensor()(img)
            var_sum += ((im_tensor - avg)**2).sum([1,2])
            n_pixels += (w*h)
        var = var_sum / n_pixels
        return torch.sqrt(var).squeeze().tolist()

    def get_fname(self, img_id):
        return self.id_to_fname[img_id.item()]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            np_img = np.asarray(img)

            img = self.transform(image=np_img)
        sample = T.ToTensor()(img)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.transform.__repr__().replace('\n',
                                                                       '\n' + ' ' * len(tmp)))
        return fmt_str

    def _check_exists(self):
        return self.root.exists()

    def download(self):
        """Download the wgisd data if it doesn't exist already."""
        import requests
        import tarfile
        from git import Repo

        if self._check_exists():
            return

        print('Downloading %s ... (may take a few minutes)' % self.url)
        self.root.mkdir()
        Repo.clone_from(self.url, str(self.root))

        print('Done!')
