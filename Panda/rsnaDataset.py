import numpy as np
import dicom
import torch
from pandas import read_csv
from pathlib import Path
from PIL import Image

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
    image = Image.fromarray(image_array)
    return image

class rsna_dataset(torch.utils.data.Dataset):

    def __init__(self, lookup_table_file_path, transform):
        self.lookup_table_file_path = Path(lookup_table_file_path)
        self.transform = transform
        self.lookup_table = read_csv(lookup_table_file_path)
        self.targets = np.array(self.lookup_table['Label'])


    def __len__(self):
        return len(self.lookup_table.index)


    def __getitem__(self, idx):
        img_path = self.lookup_table.loc[idx, 'filepath']
        img_label = self.lookup_table.loc[idx, 'Label']
        dicom_image = dicom.dcmread(img_path)
        normalized_image = normalize_dicom(dicom_image)
        tensor_image = self.transform(normalized_image)
        return tensor_image, img_label