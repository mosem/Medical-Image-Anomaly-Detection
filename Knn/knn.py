import argparse

import torch
from data_loader import get_cifar_datasets, get_feature_space
from predictor import predict


def main(args):
    normal_class = args.normal_class
    anomal_classes = args.anomal_classes
    train_dataset, test_dataset = get_cifar_datasets(normal_class, anomal_classes)
    print(f"train dataset length: {len(train_dataset)}")
    print(f"test dataset length: {len(test_dataset)}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    feature_extractor_version = args.feature_extractor_version # resnet18, 'resnet34', 'resnet50', resnet101, resnet 152
    feature_extractor = torch.hub.load('pytorch/vision:v0.9.0', feature_extractor_version, pretrained=True)
    feature_extractor.to(device)

    train_set, test_set = get_feature_space(device, feature_extractor, train_dataset, test_dataset)
    accuracy, tpr, fpr, recall, precision = predict(train_set, test_set)


    print(f"feature extractor: {feature_extractor_version}")
    print(f"normal class: {normal_class}")
    print(f"anomal classes: {anomal_classes}")
    print(f"accuracy: {accuracy}")
    print(f"tpr: {tpr}, fpr: {fpr}")
    print(f"recall: {recall}, precision: {precision}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--normal_class', type=int, default=5)
    parser.add_argument('--anomal_classes', nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--feature_extractor_version', default='resnet18')
    args = parser.parse_args()
    main(args)