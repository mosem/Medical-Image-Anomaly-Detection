

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import faiss
import ResNet
import rsnaDataset
from TimeSformerUtils import TimeSformerWrapper
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
from pandas import read_csv
import ast
from PIL import Image
import pydicom as dicom
from rsnaDataset import window_image
from pathlib import Path

from itertools import islice

from skimage import measure, filters
from skimage.morphology import binary_dilation, disk, reconstruction

mvtype = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
          'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
          'wood', 'zipper']

transform_color = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_gray = transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

def get_resnet_model(resnet_type=152):
    """
    A function that returns the required pre-trained resnet model
    :param resnet_number: the resnet type
    :return: the pre-trained model
    """
    if resnet_type == 18:
        return ResNet.resnet18(pretrained=True, progress=True)
    elif resnet_type == 50:
        return ResNet.wide_resnet50_2(pretrained=True, progress=True)
    elif resnet_type == 101:
        return ResNet.resnet101(pretrained=True, progress=True)
    else:  #152
        return ResNet.resnet152(pretrained=True, progress=True)

def get_timesformer_model(mode):
    return TimeSformerWrapper(mode)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return


def freeze_resnet_parameters(model, train_fc=False):
    for p in model.conv1.parameters():
        p.requires_grad = False
    for p in model.bn1.parameters():
        p.requires_grad = False
    for p in model.layer1.parameters():
        p.requires_grad = False
    for p in model.layer2.parameters():
        p.requires_grad = False
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False


def freeze_timesformer_parameters(model, train_norm=True, n_attention_layers_to_train = 3):
    for i in range(model.model.depth - n_attention_layers_to_train):
        for p in model.model.blocks[i].parameters():
            p.requires_grad = False
    if not train_norm:
        for p in model.model.norm.parameters():
            p.requires_grad = False
    # for p in model.model.blocks[0].parameters():
    #     p.requires_grad = False
    # for p in model.model.blocks[1].parameters():
    #     p.requires_grad = False
    # for p in model.model.blocks[2].parameters():
    #     p.requires_grad = False


def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, indices = index.search(test_set, n_neighbours)
    return D, indices

def get_outliers_loader(batch_size):
    dataset = torchvision.datasets.ImageFolder(root='./data/tiny', transform=transform_color)
    outlier_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return outlier_loader

def get_loaders(dataset, label_class, batch_size, lookup_tables_paths=None):
    if dataset in ['cifar10', 'fashion']:
        if dataset == "cifar10":
            ds = torchvision.datasets.CIFAR10
            transform = transform_color
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse)
        elif dataset == "fashion":
            ds = torchvision.datasets.FashionMNIST
            transform = transform_gray
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse)

        idx = np.array(trainset.targets) == label_class
        testset.targets = [int(t != label_class) for t in testset.targets]
        trainset.data = trainset.data[idx]
        trainset.targets = [trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
        shuffled_train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
        sorted_train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
        return sorted_train_loader, shuffled_train_loader, test_loader
    elif dataset in ['rsna', 'rsna3D']:
        if dataset == 'rsna':
            sorted_train_loader, shuffled_train_loader, test_loader = rsnaDataset.get_loaders(lookup_tables_paths, batch_size)
            return sorted_train_loader, shuffled_train_loader, test_loader
        if dataset == 'rsna3D':
            sorted_train_loader, shuffled_train_loader, test_loader = rsnaDataset.get_loaders3D(lookup_tables_paths, batch_size)
            return sorted_train_loader, shuffled_train_loader, test_loader
    else:
        print('Unsupported Dataset')
        exit()

def clip_gradient(optimizer, grad_clip):
    assert grad_clip>0, 'gradient clip value must be greater than 1'
    for group in optimizer.param_groups:
        for param in group['params']:
            # gradient
            if param.grad is None:
                continue
            param.grad.data.clamp_(-grad_clip, grad_clip)


def normalize_features(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)

    return starts_from_zero / value_range


def get_tsne(train_features, test_features):
    tsne = TSNE(n_components=2).fit_transform(np.concatenate([train_features, test_features]).astype(np.float64))

    n_train_features = train_features.shape[0]

    train_x = tsne[:n_train_features, 0]
    train_y = tsne[:n_train_features, 1]

    test_x = tsne[n_train_features:,0]
    test_y = tsne[n_train_features:,1]

    train_x = normalize_features(train_x)
    train_y = normalize_features(train_y)

    test_x = normalize_features(test_x)
    test_y = normalize_features(test_y)

    return (train_x, train_y), (test_x, test_y)

def plot_features(train_features, test_features, test_labels, results_dir_path, epoch):
    (train_tx, train_ty), (test_tx, test_ty) = get_tsne(train_features, test_features)

    test_normal_tx = np.take(test_tx, np.argwhere(test_labels == 0), axis=0)
    test_normal_ty = np.take(test_ty, np.argwhere(test_labels == 0), axis=0)
    test_anomal_tx = np.take(test_tx, np.argwhere(test_labels == 1), axis=0)
    test_anomal_ty = np.take(test_ty, np.argwhere(test_labels == 1), axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(train_tx, train_ty, c='blue',
               label='train', marker='o')


    ax.scatter(test_normal_tx, test_normal_ty, c='green',
               label='test-normal', marker='^')

    ax.scatter(test_anomal_tx, test_anomal_ty, c='red',
               label='test-anomal', marker='^')

    ax.legend(loc='best')
    filename = os.path.join(results_dir_path, str(epoch) + '-tsne.png')
    plt.savefig(filename)
    plt.close()


def fill_mask(mask):
    seed = np.ones_like(mask)
    h, w = seed.shape
    seed[0, 0] = 0 if not mask[0, 0] else 1
    seed[h - 1, 0] = 0 if not mask[h - 1, 0] else 1
    seed[h - 1, w - 1] = 0 if not mask[h - 1, w - 1] else 1
    seed[0, w - 1] = 0 if not mask[0, w - 1] else 1

    filled = reconstruction(seed, mask.copy(), method='erosion')

    return filled


def get_largest_connected_components(image, n_components=1):
    labels, num_of_cc = measure.label(image, connectivity=2, return_num=True)

    background_label = labels[0, 0]

    unique, counts = np.unique(labels, return_counts=True)
    mask = unique != background_label
    counts = counts[mask]
    unique = unique[mask]
    sorted_indices = np.argsort(counts)
    largest_component_values = unique[sorted_indices[-n_components:]]

    single_component_data_mask = np.isin(labels, largest_component_values)
    single_component_data = np.zeros_like(image)
    single_component_data[single_component_data_mask] = 1

    return single_component_data


def get_mask(pixel_array):
    DICOM_THRESHOLD = -10
    MASK_THRESHOLD = 0.2

    image = np.array(pixel_array > DICOM_THRESHOLD, dtype=np.float64)

    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    image = filters.gaussian(image, sigma=0.2)

    image = image > MASK_THRESHOLD

    largest_component_mask = get_largest_connected_components(image)

    mask = binary_dilation(largest_component_mask, disk(10))

    return fill_mask(mask)

def convert_dcm_to_img(dcm_path, window_center=40, window_width=80, mask_flag=True):
    dicom_image = dicom.dcmread(dcm_path)
    if (dicom_image.BitsStored == 12) and (dicom_image.PixelRepresentation == 0) and (int(dicom_image.RescaleIntercept) > -100):
        # see: https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai
        p = dicom_image.pixel_array + 1000
        p[p >= 4096] = p[p >= 4096] - 4096
        dicom_image.PixelData = p.tobytes()
        dicom_image.RescaleIntercept = -1000

    pixel_array = dicom_image.pixel_array * dicom_image.RescaleSlope + dicom_image.RescaleIntercept
    clipped_array = window_image(pixel_array, window_center, window_width)

    image_array = (clipped_array * 255).astype(np.uint8)

    if mask_flag:
        mask = get_mask(pixel_array)
        image_array = np.where(mask > 0, image_array, 0)

    image = Image.fromarray(image_array)
    return image

def concatente_frames(dcm_paths):

    images = [convert_dcm_to_img(x) for x in dcm_paths]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('L', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_im


def create_plot(test_samples_paths, train_samples_paths, train_sample_ids, distances, output_dir, title):
    output_path = os.path.join(output_dir, title + '.png')
    if os.path.isfile(output_path):
        print(f"{output_path} exists. skipping...")
        return
    n_neighbours = len(distances)
    fig = plt.figure(figsize=(50, 20))
    # f, axarr = plt.subplots(n_neighbours+1, 1)
    test_ax = fig.add_subplot(n_neighbours+1,1,1)
    test_ax.imshow(concatente_frames(test_samples_paths))
    test_ax.set_title('Test sample: ' + title, loc='left', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    test_ax.axis('off')
    for i in range(n_neighbours):
        ax = fig.add_subplot(n_neighbours+1, 1,i+2)
        ax.imshow(concatente_frames(train_samples_paths[i]))
        ax_title = f"{train_sample_ids[i]}: {str(distances[i])}"
        ax.set_title(ax_title, loc='left', fontdict={'fontsize': 20})
        ax.axis('off')

    print(f"saving fig {title} to {output_path}")
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_results_per_sample(results_per_sample_frame, output_dir, train_lookup_table_paths, test_lookup_table_paths):
    train_frames = [read_csv(path, index_col=[0]) for path in train_lookup_table_paths]
    train_lookup_table = pd.concat(train_frames, sort=False, ignore_index=True)

    test_frames = [read_csv(path, index_col=[0]) for path in test_lookup_table_paths]
    test_lookup_table = pd.concat(test_frames, sort=False, ignore_index=True)

    tp_dir_path = os.path.join(output_dir, 'tp')
    tn_dir_path = os.path.join(output_dir, 'tn')
    fp_dir_path = os.path.join(output_dir, 'fp')
    fn_dir_path = os.path.join(output_dir, 'fn')

    if not Path(tp_dir_path).is_dir():
        os.mkdir(tp_dir_path)
    if not Path(tn_dir_path).is_dir():
        os.mkdir(tn_dir_path)
    if not Path(fp_dir_path).is_dir():
        os.mkdir(fp_dir_path)
    if not Path(fn_dir_path).is_dir():
        os.mkdir(fn_dir_path)


    for i, row in results_per_sample_frame.iterrows():
        target = row['target']
        prediction = row['prediction']
        id = row['ID']

        if target == 0:
            if prediction == 0:
                dir_path = tp_dir_path
            elif prediction == 1:
                dir_path = fn_dir_path
        elif target == 1:
            if prediction == 0:
                dir_path = fp_dir_path
            elif prediction == 1:
                dir_path = tn_dir_path


        test_samples_paths = ast.literal_eval(test_lookup_table.loc[test_lookup_table['ID'] == id]['filepaths'].values[0])

        nearest_neighbours = ast.literal_eval(row['nearest neighbours'])
        train_samples_ids, distances = zip(*nearest_neighbours)
        train_samples_paths = [ast.literal_eval(train_lookup_table.loc[train_lookup_table['ID'] == train_sample_id]['filepaths'].values[0])
                               for train_sample_id in train_samples_ids]

        title = f"{id}"

        create_plot(test_samples_paths, train_samples_paths, train_samples_ids, distances, dir_path, title)

    print(f"Done saving figures to {output_dir}")