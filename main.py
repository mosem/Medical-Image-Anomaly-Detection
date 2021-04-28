import argparse
import os
from pathlib import Path

from datasets import getCifarSmallImbalancedDatasets
from utils import plot_results, plot_features
from modelUtils import *


def main(args):
    class_names = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
                   5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

    num_epochs = args.num_epochs
    target_class = args.target_class
    anomal_classes = args.anomal_classes
    num_anomal_classes = len(anomal_classes)
    anomal_classes_names = ','.join([class_names[i] for i in anomal_classes])
    gamma = args.gamma
    normal_subset_size = args.normal_subset_size
    anomal_subset_size = args.anomal_subset_size
    lr = args.lr
    feature_extractor_version = args.feature_extractor_version
    root_dir_path = args.dir_path

    balanced_dataset_size = anomal_subset_size
    one_class_dataset_size = normal_subset_size - anomal_subset_size

    file_name_str = f"results_{num_epochs}_{num_anomal_classes}_{gamma}_{balanced_dataset_size}_{one_class_dataset_size}_{lr}"
    dir_path_str = f"{root_dir_path}/{file_name_str}"

    if not Path(dir_path_str).exists():
        os.makedirs(dir_path_str)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    train_set, validation_set, test_set = getCifarSmallImbalancedDatasets(target_class, anomal_classes,
                                                                          num_anomal_classes = num_anomal_classes,
                                                                          normal_subset_size=normal_subset_size,
                                                                          anomal_subset_size=anomal_subset_size)


    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=20,
                                                  shuffle=True, num_workers=2)
    validation_dataloader = torch.utils.data.DataLoader(validation_set, batch_size=20,
                                                  shuffle=False, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set),
                                                  shuffle=False, num_workers=2)

    model = build_model()
    model.to(device)



    with open(f'{dir_path_str}/parameters.txt', 'w') as writefile:
        writefile.write(f"epochs: {num_epochs}\n")
        writefile.write(f"target class: {class_names[target_class]}\n")
        writefile.write(f"anomal classes: {anomal_classes_names}\n")
        writefile.write(f"gamma: {gamma}\n")
        writefile.write(f"balanced dataset size: {balanced_dataset_size}*2\n")
        writefile.write(f"one-class dataset size: {one_class_dataset_size}\n")
        writefile.write(f"learning rate: {lr}")
        writefile.write(f"feature extractor version: {feature_extractor_version}")

    log_file = open(f"{dir_path_str}/log.txt", "a")

    train_losses, train_accuracies, validation_losses, validation_accuracies = train_model(device, train_dataloader,
                                                                                           validation_dataloader, model,
                                                                                           log_file,
                                                                                           num_epochs=num_epochs, lr=lr,
                                                                                           gamma=gamma)
    print('Done training.')
    print('===================================\nTest results:')
    test_loop(device, test_dataloader, model, torch.nn.CrossEntropyLoss(), log_file)
    log_file.close()
    plot_results(train_losses, train_accuracies, validation_losses, validation_accuracies, dir_path_str)
    plot_features(device, model, test_dataloader, dir_path_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--target_class', type=int, default=5)
    parser.add_argument('--anomal_classes', nargs= '+' ,default=[0,1,2,3])
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--normal_subset_size', type=int, default=150)
    parser.add_argument('--anomal_subset_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--feature_extractor_version', default='resnet18')
    parser.add_argument('--dir_path', default='.')


    args = parser.parse_args()
    main(args)