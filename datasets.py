from __future__ import print_function, division

import torch


from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Subset, ConcatDataset
import time
import os
import copy
from random import sample


class custom_subset(torch.utils.data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        is_normal: is a normal class or anomal class
    """

    def __init__(self, dataset, indices, is_normal=False):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.is_normal = is_normal

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = 0 if self.is_normal else 1
        return (image, target)

    def __len__(self):
        return len(self.dataset)


def _extract_class(dataset, class_label, is_normal=False):
    labels = np.array(dataset.targets)
    mask_indices = np.where(labels == class_label)[0]
    subset = custom_subset(dataset, mask_indices, is_normal)
    return subset


class SmallImbalancedDataset(torch.utils.data.Dataset):
    def __init__(self, normal_dataset, anomal_dataset,
                 normal_mask_indices=[], anomal_mask_indices=[],
                 normal_subset_size=150, anomal_subset_size=10):
        normal_indices = [i for i in range(len(normal_dataset))
                          if i not in normal_mask_indices]
        anomal_indices = [i for i in range(len(anomal_dataset))
                          if i not in anomal_mask_indices]

        self.normal_subset_indices = sample(normal_indices, normal_subset_size)
        self.anomal_subset_indices = sample(anomal_indices, anomal_subset_size)

        balanced_indices_normal = self.normal_subset_indices[:anomal_subset_size]
        one_class_indices = self.normal_subset_indices[anomal_subset_size:]

        anomal_subset = Subset(anomal_dataset, self.anomal_subset_indices)
        normal_subset = Subset(normal_dataset, balanced_indices_normal)

        self.balanced_dataset = ConcatDataset([normal_subset, anomal_subset])
        self.one_class_dataset = Subset(normal_dataset, one_class_indices)

    def __getitem__(self, i):
        return (self.balanced_dataset[i % len(self.balanced_dataset)],
                self.one_class_dataset[i // len(self.balanced_dataset)])

    def __len__(self):
        return len(self.balanced_dataset) * len(self.one_class_dataset)


def getCifarTestset(normal_dataset, anomal_dataset,
                    normal_subset_size=150, anomal_subset_size=150,
                    normal_mask_indices=[], anomal_mask_indices=[]):
    normal_indices = [i for i in range(len(normal_dataset))
                      if i not in normal_mask_indices]
    anomal_indices = [i for i in range(len(anomal_dataset))
                      if i not in anomal_mask_indices]

    normal_subset_indices = sample(normal_indices, normal_subset_size)
    anomal_subset_indices = sample(anomal_indices, anomal_subset_size)

    normal_subset = Subset(normal_dataset, normal_subset_indices)
    anomal_subset = Subset(anomal_dataset, anomal_subset_indices)

    return ConcatDataset([normal_subset, anomal_subset])


def getCifarSmallImbalancedDatasets(normal_dataset_target):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    normal_dataset = _extract_class(data_set, normal_dataset_target, True)
    anomal_dataset = ConcatDataset([_extract_class(data_set, i, False)
                                    for i in range(10) if i != normal_dataset_target])

    train_set = SmallImbalancedDataset(normal_dataset, anomal_dataset)
    normal_mask_indices = train_set.normal_subset_indices
    anomal_mask_indices = train_set.anomal_subset_indices
    validation_set = SmallImbalancedDataset(normal_dataset, anomal_dataset,
                                            normal_mask_indices, anomal_mask_indices)

    normal_mask_indices += validation_set.normal_subset_indices
    anomal_mask_indices += validation_set.anomal_subset_indices

    test_set = getCifarTestset(normal_dataset, anomal_dataset,
                               normal_subset_size=150, anomal_subset_size=150,
                               normal_mask_indices=normal_mask_indices, anomal_mask_indices=anomal_mask_indices)

    return train_set, validation_set, test_set

