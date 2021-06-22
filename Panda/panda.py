import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
import torch.optim as optim
import argparse
import os
from pathlib import Path

from ResNet import ResNet3D
from losses import CompactnessLoss, EWCLoss
import utils
from copy import deepcopy
from tqdm import tqdm
import pandas as pd

def train_model(model, sorted_train_loader, shuffled_train_loader, test_loader, device, args, ewc_loss):
    model.eval()
    auc, feature_space, results = get_score(model, device, sorted_train_loader, test_loader, args.epochs == 0)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005, momentum=0.9)
    center = torch.FloatTensor(feature_space).mean(dim=0)
    criterion = CompactnessLoss(center.to(device))
    for epoch in range(args.epochs):
        running_loss = run_epoch(model, shuffled_train_loader, optimizer, criterion, device, args.ewc, ewc_loss)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        auc, feature_space, results = get_score(model, device, sorted_train_loader, test_loader,
                                                epoch == args.epochs - 1, criterion)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))
    save_results(results, args)


def save_results(results, args):
    head, tail = os.path.split(args.test_lookup_table)
    results_dir_name = head.split(os.path.sep)[-1]
    results_dir_full_path = os.path.join(args.results_output_dir, results_dir_name)
    if not Path(results_dir_full_path).is_dir():
        os.mkdir(results_dir_full_path)
    results_filename = '_'.join([args.dataset, args.model, str(args.lr), str(args.epochs)])
    results_path = os.path.join(results_dir_full_path, results_filename + '.csv')
    results.to_csv(results_path)


def run_epoch(model, train_loader, optimizer, criterion, device, ewc, ewc_loss):
    running_loss = 0.0
    for i, (imgs, _) in enumerate(train_loader):

        images = imgs.to(device)

        optimizer.zero_grad()

        _, features = model(images)

        loss = criterion(features)

        if ewc:
            loss += ewc_loss(model)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / (i + 1)


def find_optimal_threshold(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    return list(roc_t['threshold'])


def get_nearest_neighbours_results(train_loader, raw_distances, indices, is_3d_data):
    results = []
    if is_3d_data:
        for patient_distances, patient_indices in zip(raw_distances, indices):
            patient_result = []
            for neighbours_distances, neighbours_idxs in zip(patient_distances, patient_indices):
                slice_result = []
                for distance, idx in zip(neighbours_distances, neighbours_idxs):
                    slice_result.append((train_loader.dataset.ids[idx[0]], idx[1], distance))
                patient_result.append(slice_result)
            results.append(patient_result)
    else:
        for neighbours_distances, neighbours_idxs in zip(raw_distances, indices):
            patient_result = []
            for distance, idx in zip(neighbours_distances, neighbours_idxs):
                patient_result.append((train_loader.dataset.ids[idx], distance))
            results.append(patient_result)
    return results


def get_loss_results(test_feature_space, criterion):
    losses = []
    for feature in test_feature_space:
        print(feature)
        losses.append(criterion(feature))
    return losses


def get_results(test_loader, summed_distances, nearest_neighbors_results, loss_results):
    results  = pd.DataFrame(columns=['ID', 'target', 'prediction', 'nearest neighbours', 'loss'])
    results['ID'] = test_loader.dataset.ids
    results['target'] = test_loader.dataset.targets

    optimal_threshold = find_optimal_threshold(test_loader.dataset.targets, summed_distances)
    print(f"optimal threshold: {optimal_threshold}")
    results['prediction'] = np.where(summed_distances < optimal_threshold, 0, 1)
    results['nearest neighbours'] = nearest_neighbors_results
    results['loss'] = loss_results
    return results

def get_train_feature_space(model, device, train_loader):
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.to(device)
            _, features = model(imgs)
            if (len(features.size()) == 3):
                batch_size, n_slices = features.size()[:2]
                two_d_features = features.view(batch_size * n_slices, -1)
                train_feature_space.append(two_d_features)
            else:
                train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    return train_feature_space


def get_test_feature_space(model, device, test_loader):
    test_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.to(device)
            _, features = model(imgs)
            if (len(features.size()) == 3):
                batch_size, n_slices = features.size()[:2]
                two_d_features = features.view(batch_size * n_slices, -1)
                test_feature_space.append(two_d_features)
            else:
                test_feature_space.append(features)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
    return test_feature_space


def get_score(model, device, train_loader, test_loader, results_flag=False, criterion=None):
    train_feature_space = get_train_feature_space(model, device, train_loader)
    test_feature_space = get_test_feature_space(model, device, test_loader)
    test_labels = test_loader.dataset.targets

    raw_distances, indices = utils.knn_score(train_feature_space, test_feature_space)
    summed_distances = np.sum(raw_distances, axis=1)
    is_3d_data = type(model) is ResNet3D
    if is_3d_data:
        summed_distances = np.array(list(map(min, np.split(summed_distances, len(test_labels))))) # MIN from each set of slices
        indices = np.array(list(map(lambda idx_list: [(i // 8, i % 8) for i in idx_list], indices)))
        indices = np.split(indices, len(test_labels))
        raw_distances = np.split(raw_distances, len(test_labels))

    if results_flag:
        nearest_neighbours_results = get_nearest_neighbours_results(train_loader, raw_distances, indices, is_3d_data)
        loss_results = get_loss_results(test_feature_space, criterion)
        results = get_results(test_loader, summed_distances, nearest_neighbours_results, loss_results)
    else:
        results = None

    auc = roc_auc_score(test_labels, summed_distances)

    return auc, train_feature_space, results

def main(args):
    print('Dataset: {}, Normal Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model_type = args.model
    if model_type == 'resnet':
        model = utils.get_resnet_model(resnet_type=args.resnet_type)
        if args.dataset in ['rsna3D']:
            model = ResNet3D(model)
    elif model_type == 'timesformer':
        model = utils.get_timesformer_model(mode=args.timesformer_mode)
    model = model.to(device)

    ewc_loss = None

    # Freezing Pre-trained model for EWC
    if args.ewc:
        frozen_model = deepcopy(model).to(device)
        frozen_model.eval()
        utils.freeze_model(frozen_model)
        fisher = torch.load(args.diag_path)
        ewc_loss = EWCLoss(frozen_model, fisher)

    utils.freeze_parameters(model)

    sorted_train_loader, shuffled_train_loader, test_loader = utils.get_loaders(dataset=args.dataset, label_class=args.label,
                                                  batch_size=args.batch_size,
                                                  lookup_tables_paths=(args.train_lookup_table, args.test_lookup_table))
    train_model(model, sorted_train_loader, shuffled_train_loader, test_loader, device, args, ewc_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--diag_path', default='./data/fisher_diagonal.pth', help='fim diagonal path')
    parser.add_argument('--ewc', action='store_true', help='Train with EWC')
    parser.add_argument('--epochs', default=15, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-2, help='The initial learning rate.')
    parser.add_argument('--model', default='resnet')
    parser.add_argument('--timesformer_mode', default='standard')
    parser.add_argument('--resnet_type', default=152, type=int, help='which resnet to use')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--train_lookup_table',
                        default='/content/drive/MyDrive/anomaly_detection/data/rsna/8-frame-data-1000-png-train/lookup_table.csv')
    parser.add_argument('--test_lookup_table',
                        default='/content/drive/MyDrive/anomaly_detection/data/rsna/8-frame-data-200-png-test-1000/lookup_table.csv')
    parser.add_argument('--results_output_dir',
                        default='/content/drive/MyDrive/anomaly_detection/results/rsna')

    args = parser.parse_args()

    main(args)
