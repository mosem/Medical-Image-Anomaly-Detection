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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

N_SLICES = 8

def train_model(model, sorted_train_loader, shuffled_train_loader, test_loader, device, args, ewc_loss):
    model.eval()
    is_3d_data = type(model) is ResNet3D
    test_feature_space = get_test_feature_space(model, device, test_loader)
    auc, feature_space, results_per_sample = get_score(model, device, sorted_train_loader, test_loader,
                                                       test_feature_space,args, 0)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005, momentum=0.9)
    criterion = get_criterion(feature_space, device)
    test_normal_loss, test_anomal_loss = get_test_losses(model, test_feature_space, test_loader.dataset.targets,
                                                         criterion, device, args.ewc, ewc_loss, is_3d_data)
    results_per_epoch = {'training_loss': [None], 'auc': [auc], 'test_normal_loss': [test_normal_loss], 'test_anomal_loss': [test_anomal_loss]}
    results_dir_path = get_results_dir_path(args)
    for epoch in range(args.epochs):
        running_loss = run_epoch(model, shuffled_train_loader, optimizer, criterion, device, args.ewc, ewc_loss)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        test_feature_space = get_test_feature_space(model, device, test_loader)
        auc, feature_space, results_per_sample = get_score(model, device, sorted_train_loader, test_loader,
                                                           test_feature_space, args, epoch, criterion)
        test_normal_loss, test_anomal_loss = get_test_losses(model, test_feature_space, test_loader.dataset.targets,
                                                             criterion, device, args.ewc,ewc_loss, is_3d_data)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))
        results_per_epoch['training_loss'].append(running_loss)
        results_per_epoch['auc'].append(auc)
        results_per_epoch['test_normal_loss'].append(test_normal_loss)
        results_per_epoch['test_anomal_loss'].append(test_anomal_loss)
    save_results(results_per_sample, pd.DataFrame(results_per_epoch), results_dir_path, args)


def get_criterion(feature_space, device):
    center = torch.FloatTensor(feature_space).mean(dim=0)
    return CompactnessLoss(center.to(device))


def get_results_dir_path(args):
    head, tail = os.path.split(args.test_lookup_table)
    test_dataset_name = head.split(os.path.sep)[-1]
    root_dir_path = os.path.join(args.results_output_root_dir, test_dataset_name)
    if not os.path.isdir(root_dir_path):
        os.mkdir(root_dir_path)

    results_dir_name = '_'.join([args.dataset, args.model, str(args.lr), str(args.epochs)])
    if args.results_output_dir_name is not None:
        results_dir_name = '_'.join([args.results_output_dir_name, results_dir_name])
    results_dir_path = os.path.join(root_dir_path, results_dir_name)
    if not os.path.isdir(results_dir_path):
        os.mkdir(results_dir_path)
    return results_dir_path


def save_results_summary(results_per_sample, results_per_epoch, args, results_dir_path):
    tp = results_per_sample.query('target == 1 & prediction == 1').shape[0]
    tn = results_per_sample.query('target == 0 & prediction == 0').shape[0]
    fp = results_per_sample.query('target == 0 & prediction == 1').shape[0]
    fn = results_per_sample.query('target == 1 & prediction == 0').shape[0]
    optimal_threshold = results_per_sample['optimal threshold'][0]
    final_auc = results_per_epoch.iloc[-1].auc
    filepath = os.path.join(results_dir_path, 'results_summary.txt')
    with open(filepath, 'a') as f:
        f.write(f'Train paths: {args.train_lookup_table}\nTest paths: {args.test_lookup_table}')
        f.write(f'Dataset: {args.dataset}\n Model: {args.model}\n LR: {args.lr}\n n_neighbours: {args.n_neighbours}\n n_unfrozen_layers: {args.n_unfrozen_layers}\n\n\n')
        f.write(f'tp: {tp}\ntn: {tn}\nfp: {fp}\nfn: {fn}\noptimal threshold:{optimal_threshold}\nauc: {final_auc}\n')


def save_results(results_per_sample, results_per_epoch, results_dir_path, args):
    print(f'saving results to {results_dir_path}')
    head, tail = os.path.split(results_dir_path)
    results_per_sample_path = os.path.join(results_dir_path, f'results_per_sample({tail}).csv')
    results_per_sample.to_csv(results_per_sample_path)

    results_per_epoch_path = os.path.join(results_dir_path, f'results_per_epoch({tail}).csv')
    results_per_epoch.to_csv(results_per_epoch_path)

    train_lookup_table_paths, test_lookup_table_paths = get_lookup_table_paths(args)
    save_results_summary(results_per_sample, results_per_epoch, args, results_dir_path)
    # utils.plot_results_per_sample(results_per_sample, results_dir_path, train_lookup_table_paths,
    #                               test_lookup_table_paths)


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
    roc_t = roc.iloc[(roc.tf).abs().argsort()[:1]]
    return list(roc_t['threshold'])[0]


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


def get_loss_results(test_feature_space, criterion, device, is_3d_data):
    losses = []
    if is_3d_data:
        test_feature_space = np.split(np.array(test_feature_space), len(test_feature_space)//N_SLICES)
    for feature in test_feature_space:
        feature_tensor = torch.unsqueeze(torch.from_numpy(feature), 0).to(device)
        losses.append(criterion(feature_tensor).item())
    return losses


def get_results_per_sample(test_loader, summed_distances, nearest_neighbors_results, loss_results):
    results  = pd.DataFrame(columns=['ID', 'target', 'prediction', 'nearest neighbours', 'loss'])
    results['ID'] = test_loader.dataset.ids
    results['target'] = test_loader.dataset.targets

    optimal_threshold = find_optimal_threshold(test_loader.dataset.targets, summed_distances)
    print(f"optimal threshold: {optimal_threshold}")

    results['prediction'] = np.where(summed_distances < optimal_threshold, 0, 1)
    results['nearest neighbours'] = nearest_neighbors_results
    results['loss'] = loss_results
    results['optimal threshold'] = optimal_threshold
    return results

def get_train_feature_space(model, device, train_loader):
    train_feature_space = []
    print(f"get_train_feature_space: iterating on features!!!")
    with torch.no_grad():
        i = 0
        for imgs, _ in tqdm(train_loader, desc='Train set feature extracting'):
            i+=1
            imgs = imgs.to(device)
            _, features = model(imgs)
            # print(f"{i}: appending features...")
            if (len(features.size()) == 3):
                # print(f"{i}: length==3.")
                batch_size, n_slices = features.size()[:2]
                two_d_features = features.view(batch_size * n_slices, -1)
                train_feature_space.append(two_d_features.contiguous().cpu().numpy())
            else:
                # print(f"{i}: length==2.")
                train_feature_space.append(features.contiguous().cpu().numpy())
        # print(f"get_train_feature_space: concataneting features")
        # train_feature_space = torch.cat(train_feature_space, dim=0)
        #
        train_feature_space = np.concatenate(train_feature_space, axis=0)
    print(f"get_train_feature_space: done.")
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
                test_feature_space.append(two_d_features.contiguous().cpu().numpy())
            else:
                test_feature_space.append(features.contiguous().cpu().numpy())
        test_feature_space = np.concatenate(test_feature_space, axis=0)
    return test_feature_space


def get_test_losses(model, test_feature_space, test_labels, criterion, device, ewc, ewc_loss, is_3d_data):
    normal_loss = 0.0
    anomal_loss = 0.0
    normal_count = 0
    anomal_count = 0
    if is_3d_data:
        test_feature_space = np.split(np.array(test_feature_space), len(test_feature_space) // N_SLICES)
    with torch.no_grad():
        for i, features in enumerate(test_feature_space):

            loss = criterion(torch.unsqueeze(torch.from_numpy(features), 0).to(device))

            if ewc:
                loss += ewc_loss(model)

            if test_labels[i] == 0:
                normal_loss += loss.item()
                normal_count += 1
            else:
                anomal_loss += loss.item()
                anomal_count += 1

    return normal_loss / normal_count, anomal_loss / anomal_count


def get_score(model, device, train_loader, test_loader, test_feature_space, args, epoch, criterion=None):
    train_feature_space = get_train_feature_space(model, device, train_loader)
    print('get_score: got train feature space')
    test_labels = test_loader.dataset.targets

    raw_distances, indices = utils.knn_score(train_feature_space, test_feature_space, n_neighbours=args.n_neighbours)
    summed_distances = np.sum(raw_distances, axis=1)
    is_3d_data = type(model) is ResNet3D
    print(f'get_score: is 3d data: {is_3d_data}')
    if is_3d_data:
        summed_distances = np.array(list(map(min, np.split(summed_distances, len(test_labels))))) # MIN from each set of slices
        indices = np.array(list(map(lambda idx_list: [(i // N_SLICES, i % N_SLICES) for i in idx_list], indices)))
        indices = np.split(indices, len(test_labels))
        raw_distances = np.split(raw_distances, len(test_labels))

    print('get_score: plotting results.')
    utils.plot_features(train_feature_space, test_feature_space, test_labels, get_results_dir_path(args), epoch)

    if args.epochs-1 == epoch or args.epochs == 0:
        print('get_score: getting nearest neighbours results')
        nearest_neighbours_results = get_nearest_neighbours_results(train_loader, raw_distances, indices, is_3d_data)
        if criterion is None:
            criterion = get_criterion(train_feature_space, device)
        print('get_score: getting loss results')
        loss_results = get_loss_results(test_feature_space, criterion, device, is_3d_data)
        print('get_score: getting results per sample')
        results = get_results_per_sample(test_loader, summed_distances, nearest_neighbours_results, loss_results)
    else:
        results = None

    print('get_score: getting roc score.')
    auc = roc_auc_score(test_labels, summed_distances)

    return auc, train_feature_space, results

def get_lookup_table_paths(args):
    train_lookup_table_paths = str(args.train_lookup_table).split(',')
    test_lookup_table_paths = str(args.test_lookup_table).split(',')
    return train_lookup_table_paths, test_lookup_table_paths

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

    if model_type == 'timesformer':
        utils.freeze_timesformer_parameters(model.timesformer_model, train_norm=args.train_norm,
                                      n_attention_layers_to_train=args.n_unfrozen_layers)
    else:
        utils.freeze_resnet_parameters(model)

    train_lookup_table_paths, test_lookup_table_paths = get_lookup_table_paths(args)

    sorted_train_loader, shuffled_train_loader, test_loader = utils.get_loaders(dataset=args.dataset, label_class=args.label,
                                                  batch_size=args.batch_size,
                                                  lookup_tables_paths=(train_lookup_table_paths, test_lookup_table_paths))
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
                        default="/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/8-frame-data-train-2 /lookup_table.csv")
    parser.add_argument('--test_lookup_table',
                        default="/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/8-frame-data-test-2/lookup_table.csv")
    parser.add_argument('--results_output_root_dir',
                        default="/vol/ep/mm/anomaly_detection/results")
    parser.add_argument('--results_output_dir_name', default=None)
    parser.add_argument('--n_neighbours', default=2, type=int)
    parser.add_argument('--n_unfrozen_layers', default=3, type=int)
    parser.add_argument('--train_norm', action='store_true')

    args = parser.parse_args()

    main(args)
