import torch
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Subset, ConcatDataset
from tqdm import tqdm

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


def get_cifar_datasets(normal_class, anomal_classes):
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_set = torchvision.datasets.CIFAR10(root='../data', train=True,
                                             download=True, transform=train_transform)

    test_set = torchvision.datasets.CIFAR10(root='../data', train=False,
                                            download=True, transform=test_transform)

    normal_train_dataset = _extract_class(train_set, normal_class, True)
    anomal_train_dataset = ConcatDataset([_extract_class(train_set, i, False)
                                          for i in anomal_classes])

    train_dataset = ConcatDataset([normal_train_dataset, anomal_train_dataset])

    normal_test_dataset = _extract_class(test_set, normal_class, True)
    anomal_test_dataset = ConcatDataset([_extract_class(test_set, i, False)
                                         for i in anomal_classes])

    test_dataset = ConcatDataset([normal_test_dataset, anomal_test_dataset])

    return train_dataset, test_dataset


def get_feature_space(device, feature_extractor, train_dataset, test_dataset):
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                                   shuffle=False, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32,
                                                  shuffle=False, num_workers=2)

    train_features = []
    train_labels = []
    with torch.no_grad():
        for (imgs, labels, _) in tqdm(train_dataloader, desc='Train set feature extracting'):
            imgs = imgs.to(device)
            features = feature_extractor(imgs)
            train_features.append(features)
            train_labels.append(labels)
    train_feature_space = torch.cat(train_features, dim=0).contiguous().cpu().numpy()
    train_labels_space = torch.cat(train_labels, dim=0).contiguous().cpu().numpy()

    test_features = []
    test_labels = []
    with torch.no_grad():
        for (imgs, labels, _) in tqdm(test_dataloader, desc='Test set feature extracting'):
            imgs = imgs.to(device)
            features = feature_extractor(imgs)
            test_features.append(features)
            test_labels.append(labels)
    test_feature_space = torch.cat(test_features, dim=0).contiguous().cpu().numpy()
    test_labels_space = torch.cat(test_labels, dim=0).contiguous().cpu().numpy()

    return (train_feature_space, train_labels_space), (test_feature_space, test_labels_space)