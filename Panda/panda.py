import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse

from ResNet import ResNet3D
from losses import CompactnessLoss, EWCLoss
import utils
from copy import deepcopy
from tqdm import tqdm
import gc

def train_model(model, train_loader, test_loader, device, args, ewc_loss):
    model.eval()
    auc, feature_space = get_score(model, device, train_loader, test_loader)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005, momentum=0.9)
    center = torch.FloatTensor(feature_space).mean(dim=0)
    criterion = CompactnessLoss(center.to(device))
    del feature_space
    gc.collect()
    for epoch in range(args.epochs):
        running_loss = run_epoch(model, train_loader, optimizer, criterion, device, args.ewc, ewc_loss)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        auc, feature_space = get_score(model, device, train_loader, test_loader)
        print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))
        del feature_space
        gc.collect()

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


def get_score(model, device, train_loader, test_loader):
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.to(device)
            _, features = model(imgs)
            if (len(features.size()) == 3):
                batch_size, n_slices = features.size()[:2]
                two_d_features = features.view(batch_size*n_slices, -1)
                train_feature_space.extend(two_d_features)
            else:
                train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.to(device)
            _, features = model(imgs)
            if (len(features.size()) == 3):
                batch_size, n_slices = features.size()[:2]
                two_d_features = features.view(batch_size * n_slices, -1)
                test_feature_space.extend(two_d_features)
            else:
                test_feature_space.append(features)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = test_loader.dataset.targets
    print(train_feature_space.shape())
    distances = utils.knn_score(train_feature_space, test_feature_space)

    auc = roc_auc_score(test_labels, distances)

    return auc, train_feature_space

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

    if model_type == 'resnet':
        utils.freeze_parameters(model)
    train_lookup_tables = args.train_lookup_tables.split(' ')
    test_lookup_tables = args.test_lookup_tables.split(' ')
    train_loader, test_loader = utils.get_loaders(dataset=args.dataset, label_class=args.label,
                                                  batch_size=args.batch_size, lookup_tables_paths=(train_lookup_tables, test_lookup_tables))
    train_model(model, train_loader, test_loader, device, args, ewc_loss)


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
    parser.add_argument('--train_lookup_tables',
                        default='/content/drive/MyDrive/anomaly_detection/data/rsna/8-frame-data-train-dummy-png/lookup_table.csv')
    parser.add_argument('--test_lookup_tables',
                        default='/content/drive/MyDrive/anomaly_detection/data/rsna/8-frame-data-test-dummy-png/lookup_table.csv')

    args = parser.parse_args()

    main(args)
