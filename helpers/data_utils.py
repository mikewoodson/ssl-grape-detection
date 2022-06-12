import imgaug.augmenters as iaa
import torch
from datahandlers.wgisd import Wgisd
from datahandlers.cr2 import Cr2
from pathlib import Path

data_dir = Path.home() / 'hdrive' / 'data'
data_roots = {
    'wgisd' : data_dir/'wgisd',
    'cr2' : data_dir/'CR2',
}

data_to_annotations = {
    'wgisd' : 'bbox',
    'cr2' : 'dot',
}

def get_transform(split):
    transform = None

    # experiment with rotation to see how much it impacts accuracy
    if split == 'train' or split == 'trainval':
        aug_list = [
            iaa.Dropout(),
            iaa.AdditiveGaussianNoise(),
            iaa.GaussianBlur(),
            iaa.pillike.EnhanceContrast(),
            iaa.Fliplr(),
        ]
        #aug_list.append(iaa.Rotate())

        transform = iaa.SomeOf((0, len(aug_list)), aug_list, random_order=True)

    return transform

def get_dataset(dataset, split):
    if dataset == 'wgisd':
        dset = Wgisd
        root = data_roots[dataset]
    elif dataset == 'cr2':
        dset = Cr2
        root = data_roots[dataset]
    else:
        raise ValueError()

    if split not in dset.splits:
        raise ValueError(f'Argument not one of {dset.splits}')

    transform = get_transform(split)
    return dset(root, split=split, transform=transform)
