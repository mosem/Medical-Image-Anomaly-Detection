import torch
import torch.optim as optim
import torch.nn as nn

from jointModel import JointModel
from losses import CompactnessLoss

from sklearn import metrics
import numpy as np


def train_loop(device, dataloader, model, compactness_loss_fn, cross_entropy_loss_fn, optimizer, log_file, gamma):
    dataloader_size = len(dataloader.dataset)
    running_compactness_loss = 0.0
    running_cross_entropy_loss = 0.0
    running_total_loss = 0.0


    for i, ((balanced_data, balanced_labels, _), (one_class_data, one_class_labels, _)) in enumerate(dataloader):
        balanced_data, balanced_labels = balanced_data.to(device), balanced_labels.to(device)
        one_class_data, one_class_labels = one_class_data.to(device), one_class_labels.to(device)

        # get features for one class data
        one_class_features = model.get_features(one_class_data)


        # Compute prediction and loss
        balanced_prediction = model(balanced_data)
        cross_entropy_loss = cross_entropy_loss_fn(balanced_prediction, balanced_labels)
        compactness_loss = compactness_loss_fn(one_class_features)
        total_loss = gamma*cross_entropy_loss + (1-gamma)*compactness_loss


        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_compactness_loss += compactness_loss.item()
        running_cross_entropy_loss += cross_entropy_loss.item()
        running_total_loss += total_loss.item()


        if i%40 == 39:
            running_compactness_loss /= 40
            running_cross_entropy_loss /= 40
            running_total_loss /= 40

            X = torch.cat((balanced_data, one_class_data))
            y = torch.cat((balanced_labels, one_class_labels))

            pred = model.predict(X)
            accuracy = (pred.argmax(1) == y).type(torch.float).sum().item()

            accuracy /= dataloader.batch_size*2

            print('compactness loss: {}'.format(running_compactness_loss))
            print('cross-entropy loss: {}'.format(running_cross_entropy_loss))
            log_file.write('compactness loss: {}'.format(running_compactness_loss))
            log_file.write('\ncross-entropy loss: {}'.format(running_cross_entropy_loss))

            print(f"({i*dataloader.batch_size}/{dataloader_size}):Training Error:\n\tAvg loss: {running_total_loss:>8f}\n\tAccuracy: {(100*accuracy):>0.1f}%\n")
            log_file.write(f"\n({i*dataloader.batch_size}/{dataloader_size}):Training Error:\n\tAvg loss: {running_total_loss:>8f}\n\tAccuracy: {(100*accuracy):>0.1f}%\n")
            if i != dataloader_size-1:
              running_compactness_loss = 0.0
              running_cross_entropy_loss = 0.0
              running_total_loss = 0.0
              accuracy = 0
    return running_compactness_loss, running_cross_entropy_loss, running_total_loss


def validation_loop(device, dataloader, model, loss_fn, log_file):
    model.eval()
    size = len(dataloader.dataset)
    loss, accuracy = 0, 0

    with torch.no_grad():

        for i, ((balanced_data, balanced_labels, _), (one_class_data, one_class_labels, _)) in enumerate(dataloader):
            balanced_data, balanced_labels = balanced_data.to(device), balanced_labels.to(device)
            one_class_data, one_class_labels = one_class_data.to(device), one_class_labels.to(device)

            X = torch.cat((balanced_data, one_class_data))
            y = torch.cat((balanced_labels, one_class_labels))

            pred = model.predict(X)
            loss += loss_fn(pred, y).item()
            accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

    loss /= size*2
    accuracy /= size*2
    print(f"\n\tClassification loss: {loss:>8f}\n\tAccuracy: {(100*accuracy):>0.1f}%\n")
    log_file.write(f"\n\tClassification loss: {loss:>8f}\n\tAccuracy: {(100*accuracy):>0.1f}%\n")
    return loss, accuracy


def test_loop(device, dataloader, model, loss_fn, log_file):
    model.eval()
    size = len(dataloader.dataset)
    loss, accuracy = 0, 0

    total_labels = np.array([], dtype=np.int8)
    total_preds = np.array([], dtype=np.int8)

    with torch.no_grad():
        for i, (data, labels, _) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)

            pred = model.predict(data)
            loss += loss_fn(pred, labels).item()
            accuracy += (pred.argmax(1) == labels).type(torch.float).sum().item()

            total_labels = np.concatenate((total_labels, labels.cpu().numpy().flatten()), axis=0)
            total_preds = np.concatenate((total_preds, pred.argmax(1).cpu().numpy().flatten()), axis=0)

    loss /= size
    accuracy /= size

    fpr, tpr, thresholds = metrics.roc_curve(total_labels, total_preds)
    f1 = metrics.f1_score(total_labels, total_preds)
    auc = metrics.auc(fpr, tpr)

    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
    print(f"fpr: {fpr}, tpr: {tpr}, f1 score: {f1}, AUROC: {auc:>0.3f}")
    log_file.write(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
    log_file.write(f"fpr: {fpr}, tpr: {tpr}, f1 score: {f1}, AUROC: {auc:>0.3f}")


def train_model(device, train_dataloader, validation_dataloader, model, log_file, num_epochs=25, lr=1e-3, gamma=0.5):
    compactness_loss_fn = CompactnessLoss()
    cross_entropy_loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    train_accuracies = []
    train_compactness_losses, train_classification_losses, train_avg_losses = [], [], []
    validation_losses, validation_accuracies = [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        log_file.write(f"Epoch {epoch + 1}\n-------------------------------")
        train_compactness_loss, train_classification_loss, train_avg_loss = train_loop(device, train_dataloader, model,
                                                                                       compactness_loss_fn,
                                                                                       cross_entropy_loss_fn, optimizer,
                                                                                       log_file, gamma)
        train_compactness_losses.append(train_compactness_loss)
        train_classification_losses.append(train_classification_loss)
        train_avg_losses.append(train_avg_loss)
        print('validation on training set:')
        log_file.write('validation on training set:')
        _, train_accuracy = validation_loop(device, train_dataloader, model, torch.nn.CrossEntropyLoss(), log_file)
        train_accuracies.append(train_accuracy)
        print('validation on validation set:')
        log_file.write('validation on validation set:')
        validate_loss, validate_accuracy = validation_loop(device, validation_dataloader, model, torch.nn.CrossEntropyLoss(),
                                                           log_file)
        validation_losses.append(validate_loss)
        validation_accuracies.append(validate_accuracy)
        print('Done!')

    train_losses = {'compactness': train_compactness_losses, 'classification': train_classification_losses,
                    'average': train_avg_losses}
    return train_losses, train_accuracies, validation_losses, validation_accuracies


def build_model(feature_extractor_version='resnet18'):
    feature_extractor = torch.hub.load('pytorch/vision:v0.9.0', feature_extractor_version, pretrained=True)

    for name, param in feature_extractor.named_parameters():
        if name.split('.')[0] in ['layer1', 'layer2']:
            param.requires_grad = False

    classifier = nn.Linear(feature_extractor.fc.out_features, 2)
    model = JointModel(feature_extractor, classifier)
    return model