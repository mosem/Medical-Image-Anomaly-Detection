import numpy as np
import pydicom as dicom
import torch
from torch.utils.data import Subset, ConcatDataset
from pandas import read_csv
from pathlib import Path
from PIL import Image
from torchvision import transforms
import ast
import pandas as pd


def window_image(pixel_array, window_center, window_width, is_normalize=True):
    image_min = window_center - window_width // 2
    image_max = window_center + window_width // 2
    image = np.clip(pixel_array, image_min, image_max)

    if is_normalize:
        image = (image-image_min)/(image_max-image_min)
    return image

def normalize_dicom(dicom):

    if (dicom.BitsStored == 12) and (dicom.PixelRepresentation == 0) and (int(dicom.RescaleIntercept) > -100):
        # see: https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai
        p = dicom.pixel_array + 1000
        p[p>=4096] = p[p>=4096] - 4096
        dicom.PixelData = p.tobytes()
        dicom.RescaleIntercept = -1000

    pixel_array = dicom.pixel_array * dicom.RescaleSlope + dicom.RescaleIntercept
    brain       = window_image(pixel_array, 40,  80)
    subdural    = window_image(pixel_array, 80, 200)
    soft_tissue = window_image(pixel_array, 40, 380)

    image_array = np.dstack([soft_tissue,subdural,brain])
    image_array = (image_array*255).astype(np.uint8)
    return image_array

class RsnaDataset3D(torch.utils.data.Dataset):

    def __init__(self, lookup_table_file_path, transform):
        self.lookup_table = read_csv(lookup_table_file_path)
        self.transform = transform
        self.targets = self.lookup_table['label'].to_numpy()


    def __len__(self):
        return len(self.targets)


    def __getitem__(self, idx):
        label = self.lookup_table.loc[idx, 'label']
        path = self.lookup_table.loc[idx, 'path']
        indices = ast.literal_eval(self.lookup_table.loc[idx, 'indices'])
        return self.__get_images(path, indices), label


    def __get_images(self, item_table_path, indices):
        table = read_csv(item_table_path)
        images = []
        for idx in indices:
            img_path = table.loc[idx, 'filepath']
            dicom_image = dicom.dcmread(img_path)
            normalized_image = normalize_dicom(dicom_image)
            tensor_image = self.transform(normalized_image)
            images.append(tensor_image)
        return torch.stack(images, dim=3)


class RsnaDataset(torch.utils.data.Dataset):

    def __init__(self, lookup_table_file_path, transform, balanced=False):
        self.lookup_table_file_path = Path(lookup_table_file_path)
        self.transform = transform
        self.lookup_table = read_csv(lookup_table_file_path)
        normal_indices = self.lookup_table.index[self.lookup_table['Label']==0].tolist()
        anomal_indices = self.lookup_table.index[self.lookup_table['Label']==1].tolist()
        if balanced:
            max_num_samples = min(len(anomal_indices), len(normal_indices))
            normal_indices = normal_indices[:max_num_samples]
            anomal_indices = anomal_indices[:max_num_samples]
            self.indices = normal_indices + anomal_indices
        else:
            self.indices = [i for i in range(len(self.lookup_table))]
        self.targets = np.array(self.lookup_table.loc[self.indices, 'Label'])


    def __len__(self):
        return len(self.indices)


    def __getitem__(self, idx):
        raw_idx = self.indices[idx]
        img_path = self.lookup_table.loc[raw_idx, 'filepath']
        if (type(img_path) != str):
            img_path = img_path.item()
        img_label = self.lookup_table.loc[raw_idx, 'Label']
        if (type(img_label) != int):
            img_label = img_label.item()
        dicom_image = dicom.dcmread(img_path)
        normalized_image = Image.fromarray(normalize_dicom(dicom_image))
        tensor_image = self.transform(normalized_image)
        return tensor_image, img_label

class RsnaConcatDataset(torch.utils.data.ConcatDataset):

    def __init__(self, rsna_datasets):
        super().__init__(rsna_datasets)
        self.targets = []
        for rsna_dataset in rsna_datasets:
            if hasattr(rsna_dataset, 'targets'):
                self.targets.extend(rsna_dataset.targets)

def split_tables_to_datasets(lookup_tables_paths, transform):
    normal_subsets = []
    anomal_subsets = []
    for path in lookup_tables_paths:
        dataset = RsnaDataset(path, transform)
        normal_indices = np.argwhere(dataset.targets==0)
        anomal_indices = np.argwhere(dataset.targets==1)
        normal_subset = Subset(dataset, normal_indices)
        normal_subset.targets = dataset.targets[normal_indices]
        anomal_subset = Subset(dataset, anomal_indices)
        anomal_subset.targets = dataset.targets[anomal_indices]
        normal_subsets.append(normal_subset)
        anomal_subsets.append(anomal_subset)
    normal_dataset = RsnaConcatDataset(normal_subsets)
    anomal_dataset = RsnaConcatDataset(anomal_subsets)
    return normal_dataset, anomal_dataset


def getRsnaTestset(lookup_tables_paths, transform):
    datasets = []
    for path in lookup_tables_paths:
        datasets.append(RsnaDataset(path, transform))
    return RsnaConcatDataset(datasets)


def get_loaders(lookup_tables_paths, batch_size):
    train_lookup_tables_paths, test_lookup_tables_paths = lookup_tables_paths
    # train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
    #                                       transforms.RandomHorizontalFlip(),
    #                                       transforms.ToTensor(),
    #                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])

    # test_transform = transforms.Compose([transforms.Resize(256),
    #                                     transforms.CenterCrop(224),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor()])

    normal_train_dataset, _ = split_tables_to_datasets(train_lookup_tables_paths, train_transform)

    test_dataset = getRsnaTestset(test_lookup_tables_paths, test_transform)

    train_dataloader = torch.utils.data.DataLoader(normal_train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2, drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=2, drop_last=False)

    return train_dataloader, test_dataloader


