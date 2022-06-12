from torchvision.datasets.folder import pil_loader, accimage_loader, default_loader
from pathlib import Path
from torchvision import transforms as T
from helpers.annotation_utils import convert_bbox_format, bbox_to_imgaug, bbox_to_tensor
from helpers.annotation_utils import kp_to_imgaug, kp_to_tensor

import os
import numpy as np
import pdb
import functools

import torch.utils.data as data
import torch


class Wgisd(data.Dataset):
    """ WGISD dataset
    Args:
        root (string): Root directory path to dataset.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g. ``transforms.RandomCrop``
        loader (callable, optional): A function to load an image given its path.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in the root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'https://github.com/thsant/wgisd.git'
    splits = ('train', 'val', 'trainval', 'test')

    def __init__(self, root, split='train', transform=None,
                 loader=default_loader, download=False,
                 val_size=0.2):
        if split not in self.splits:
            raise ValueError(
                'Split "{}" not found. Valid splits are: {}'.format(
                    split, ', '.join(
                        self.splits), ))
        if val_size < 0 or val_size > 1:
            raise ValueError('val_size should be a fraction between 0 and 1')
        self.root = Path(root)
        self.split = split

        # There's no file specifying a validation dataset, so use a subset of the
        # training dataset
        dset_file = 'train' if self.split in ('train', 'val', 'trainval') else 'test'
        self.data_file = self.root / f'{dset_file}.txt'
        self.dot_annot_path = self.root / 'contrib' / 'berries'
        self.data_dir = self.root / 'data'

        if download:
            self.download()

        self.transform = transform
        self.loader = loader
        self.id_to_fname = {}
        self.val_size = val_size

        self.total_set = None
        self.samples = None
        self.create_dataset()
        self.partition_dset()

    def create_dataset(self):
        image_names = []
        samples = []
        with open(self.data_file, 'r') as f:
            for line in f:
                image_names.append(line.rstrip())

        # Read bbox annotations from file
        for idx, img_name in enumerate(image_names):
            target = {}
            gt_boxes = []
            box_annotations = self.data_dir / f'{img_name}.txt'
            dot_annotations = self.dot_annot_path / f'{img_name}-berries.txt'
            img_path = self.data_dir / f'{img_name}.jpg'

            gt_boxes = np.loadtxt(box_annotations)[:,1:]
            gt_dots = np.loadtxt(dot_annotations)
            gt_dots_tensor = torch.as_tensor(gt_dots, dtype=torch.int32)
            gt_boxes_tensor = torch.as_tensor(gt_boxes, dtype=torch.float32)

            boxes = convert_bbox_format(gt_boxes_tensor, conversionType=1)
            img = self.loader(img_path)
            
            width, height = img.size
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * width
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * height
            boxes = boxes.to(dtype=torch.int32)
            numObjs = boxes.shape[0]
            labels = torch.ones((numObjs,), dtype=torch.int64)
            iscrowd = torch.zeros((numObjs,), dtype=torch.int64)
            image_id = torch.tensor([idx])
            self.id_to_fname[image_id.item()] = img_path.parts[-1]
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            target['boxes'] = boxes
            target['labels'] = labels
            target['image_id'] = image_id
            target['area'] = area
            target['iscrowd'] = iscrowd
            target['dots'] = gt_dots_tensor

            samples.append((img_path, target))
        self.total_set = samples

    def partition_dset(self):
        num_images = len(self.total_set)
        split = int(np.floor(self.val_size * num_images))
        if self.split in ('trainval', 'test'):
            self.samples = self.total_set
        elif self.split == 'train':
            self.samples = self.total_set[split:]
        elif self.split == 'val':
            self.samples = self.total_set[:split]
        else:
            self.samples = self.total_set

    @functools.cached_property
    def mean(self):
        n_pixels = 0
        pix_sum = torch.zeros([3])
        for img_path, _ in self.total_set:
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
        for img_path, _ in self.total_set:
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
        target_copy = target.copy()
        img = self.loader(path)
        if self.transform is not None:
            np_img = np.asarray(img)
            tensor_boxes = target_copy['boxes']
            tensor_dots = target_copy['dots']
            bboxes = bbox_to_imgaug(tensor_boxes, np_img.shape)
            dots = kp_to_imgaug(tensor_dots, np_img.shape)

            img, boxes, dots = self.transform(
                image=np_img,
                bounding_boxes=bboxes,
                keypoints=dots)
            target_copy['boxes'] = bbox_to_tensor(boxes)
            target_copy['dots'] = kp_to_tensor(dots)
        sample = T.ToTensor()(img)

        return sample, target_copy

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
