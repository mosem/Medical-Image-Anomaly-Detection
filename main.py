import os
from pathlib import Path

import torch

from datasets import getCifarSmallImbalancedDatasets
from utils import plot_results
from modelUtils import *



if __name__ == "__main__":

    class_names = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
                   5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

    num_epochs = 25
    target_class = 5
    anomal_classes = [0]
    num_anomal_classes = len(anomal_classes)
    anomal_classes_names = ','.join([class_names[i] for i in anomal_classes])
    gamma = 0.9
    normal_subset_size = 150
    anomal_subset_size = 10
    lr = 5e-5
    feature_extractor_version = 'resnet18'

    balanced_dataset_size = anomal_subset_size
    one_class_dataset_size = normal_subset_size - anomal_subset_size

    file_name_str = f"results_{num_epochs}_{num_anomal_classes}_{gamma}_{balanced_dataset_size}_{one_class_dataset_size}_{lr}"
    dir_path_str = f"/content/{file_name_str}"

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