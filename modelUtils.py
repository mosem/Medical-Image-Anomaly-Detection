import torch
import torch.optim as optim
import torch.nn as nn

from jointModel import JointModel
from losses import CompactnessLoss

from sklearn import metrics
import numpy as np


def train_loop(dataloader, model, compactness_loss_fn, cross_entropy_loss_fn, optimizer, gamma):
    dataloader_size = len(dataloader.dataset)
    running_compactness_loss = 0.0
    running_cross_entropy_loss = 0.0
    running_total_loss = 0.0


    for i, ((balanced_data, balanced_labels), (one_class_data, one_class_labels)) in enumerate(dataloader):
        balanced_data, balanced_labels = balanced_data.to(device), balanced_labels.to(device)
        one_class_data, one_class_labels = one_class_data.to(device), one_class_labels.to(device)

        # get features for one class data
        one_class_features = model.get_features(one_class_data)


        # Compute prediction and loss
        balanced_prediction = model(balanced_data)
        cross_entropy_loss = cross_entropy_loss_fn(balanced_prediction, balanced_labels)
        compactness_loss = compactness_loss_fn(one_class_features)
        total_loss = gamma*compactness_loss + (1-gamma)*cross_entropy_loss


        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_compactness_loss += compactness_loss.item()
        running_cross_entropy_loss += cross_entropy_loss.item()
        running_total_loss += total_loss


        if i%40 == 0:
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

            # for param_group in optimizer.param_groups:
            #     print("learning rate: {}".format(param_group['lr']))

            print(f"({i*dataloader.batch_size}/{dataloader_size}):Training Error:\n\tAvg loss: {running_total_loss:>8f}\n\tAccuracy: {(100*accuracy):>0.1f}%\n")

            running_compactness_loss = 0.0
            running_cross_entropy_loss = 0.0
            running_total_loss = 0.0
            accuracy = 0


def validation_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    loss, accuracy = 0, 0

    for i, ((balanced_data, balanced_labels), (one_class_data, one_class_labels)) in enumerate(dataloader):
        balanced_data, balanced_labels = balanced_data.to(device), balanced_labels.to(device)
        one_class_data, one_class_labels = one_class_data.to(device), one_class_labels.to(device)

        X = torch.cat((balanced_data, one_class_data))
        y = torch.cat((balanced_labels, one_class_labels))

        pred = model.predict(X)
        loss += loss_fn(pred, y).item()
        accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

    loss /= size*2
    accuracy /= size*2
    print(f"Validation Error:\n\tAvg loss: {loss:>8f}\n\tAccuracy: {(100*accuracy):>0.1f}%\n")
    return loss, accuracy


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    loss, accuracy = 0, 0

    total_labels = np.array([], dtype=np.int8)
    total_preds = np.array([], dtype=np.int8)

    for i, (data, label) in enumerate(dataloader):
        data, label = data.to(device), label.to(device)

        pred = model.predict(data)
        loss += loss_fn(pred, label).item()
        accuracy += (pred.argmax(1) == label).type(torch.float).sum().item()

        total_labels = np.concatenate((total_labels, label.cpu().numpy().flatten()), axis=0)
        total_preds = np.concatenate((total_preds, pred.argmax(1).cpu().numpy().flatten()), axis=0)

    loss /= size
    accuracy /= size

    fpr, tpr, thresholds = metrics.roc_curve(total_labels, total_preds)
    auc = metrics.auc(fpr, tpr)

    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
    print(f"AUROC: {auc:>0.3f}")


def train_model(train_dataloader, validation_dataloader, model, num_epochs=25, lr=1e-3, gamma=0.5):
    compactness_loss_fn = CompactnessLoss()
    cross_entropy_loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=5e-5)

    train_losses, train_accuracies = [], []
    validation_losses, validation_accuracies = [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_dataloader, model, compactness_loss_fn, cross_entropy_loss_fn, optimizer, gamma)
        print('validation on training set:')
        train_loss, train_accuracy = validation_loop(train_dataloader, model, cross_entropy_loss_fn)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print('validation on validation set:')
        validate_loss, validate_accuracy = validation_loop(validation_dataloader, model, cross_entropy_loss_fn)
        validation_losses.append(validate_loss)
        validation_accuracies.append(validate_accuracy)
        print('Done!')

    return train_losses, train_accuracies, validation_losses, validation_accuracies


def build_model():
    # feature_extractor = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    # feature_extractor = torch.hub.load('pytorch/vision:v0.9.0', 'resnet34', pretrained=True)
    feature_extractor = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
    # feature_extractor = torch.hub.load('pytorch/vision:v0.9.0', 'resnet101', pretrained=True)
    # feature_extractor = torch.hub.load('pytorch/vision:v0.9.0', 'resnet152', pretrained=True)
    for name, param in feature_extractor.named_parameters():
        if name.split('.')[0] in ['layer1', 'layer2']:
            param.requires_grad = False

    classifier = nn.Linear(feature_extractor.fc.out_features, 2)
    model = JointModel(feature_extractor, classifier)
    return model

