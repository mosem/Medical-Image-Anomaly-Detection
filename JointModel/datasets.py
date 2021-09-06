from __future__ import print_function, division

import torchvision
from torchvision import transforms
from torch.utils.data import Subset, ConcatDataset
from random import sample
from rsnaDataset_tmp import *


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
        image, raw_label = self.dataset[idx]
        label = 0 if self.is_normal else 1
        return image, label, raw_label

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


def getCifarSmallImbalancedDatasets(normal_dataset_target,
                                    anomal_classes=[],
                                    num_anomal_classes=2,
                                    normal_subset_size=150,
                                    anomal_subset_size=10):
    if anomal_classes == []:
        anomal_classes = sample([i for i in range(10) if i != normal_dataset_target], num_anomal_classes)

    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    val_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=train_transform)

    validation_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                  download=True, transform=val_transform)

    normal_train_dataset = _extract_class(train_set, normal_dataset_target, True)

    anomal_train_dataset = ConcatDataset([_extract_class(train_set, i, False)
                                          for i in anomal_classes])

    normal_val_dataset = _extract_class(validation_set, normal_dataset_target, True)
    anomal_val_dataset = ConcatDataset([_extract_class(validation_set, i, False)
                                        for i in anomal_classes])

    train_set = SmallImbalancedDataset(normal_train_dataset, anomal_train_dataset,
                                       normal_mask_indices=[],
                                       anomal_mask_indices=[],
                                       normal_subset_size=normal_subset_size,
                                       anomal_subset_size=anomal_subset_size)

    validation_set = SmallImbalancedDataset(normal_val_dataset, anomal_val_dataset,
                                            normal_mask_indices=[],
                                            anomal_mask_indices=[],
                                            normal_subset_size=normal_subset_size,
                                            anomal_subset_size=anomal_subset_size)

    test_set = getCifarTestset(normal_val_dataset, anomal_val_dataset,
                               normal_subset_size=50, anomal_subset_size=50,
                               normal_mask_indices=validation_set.normal_subset_indices,
                               anomal_mask_indices=validation_set.anomal_subset_indices)

    return train_set, validation_set, test_set


def getRsnaSmallImbalancedDatasets(train_lookup_tables_paths,
                                   validation_lookup_tables_paths,
                                   test_lookup_tables_paths,
                                   normal_subset_size=150,
                                   anomal_subset_size=10):
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    val_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    vanilla_transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor()])

    normal_train_dataset, anomal_train_dataset = split_tables_to_datasets(train_lookup_tables_paths, train_transform)
    normal_val_dataset, anomal_val_dataset = split_tables_to_datasets(validation_lookup_tables_paths, val_transform)

    train_set = SmallImbalancedDataset(normal_train_dataset, anomal_train_dataset,
                                       normal_mask_indices=[],
                                       anomal_mask_indices=[],
                                       normal_subset_size=normal_subset_size,
                                       anomal_subset_size=anomal_subset_size)

    validation_set = SmallImbalancedDataset(normal_val_dataset, anomal_val_dataset,
                                            normal_mask_indices=[],
                                            anomal_mask_indices=[],
                                            normal_subset_size=normal_subset_size,
                                            anomal_subset_size=anomal_subset_size)

    test_set = getRsnaTestset(test_lookup_tables_paths, vanilla_transform)

    return train_set, validation_set, test_set