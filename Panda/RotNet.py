import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import argparse
import os
from rsnaDataset import get_loaders3DSliced
import utils



class RotNet3D(nn.Module):

    def __init__(self, model_path):
        super(RotNet3D, self).__init__()
        self.model = utils.get_resnet_model(resnet_type=152, pretrained=False, num_classes=16)
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)

        self.conv1 = self.model.conv1
        self.bn1 = self.model.bn1
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.fc = self.model.fc


    def forward(self, x):
        """"
            :x tensor of size batch_size x channels x frames x height x width
            :returns    pred of size batch_size x frames x 1
                        features of size batch_size x frames x features_dimension
        """

        batch_size, n_channels, n_frames, height, width = x.shape
        x = x.permute((0, 2, 1, 3, 4))  # batch x frames x channels x height x width
        x = x.reshape(batch_size * n_frames, n_channels, height, width)
        pred, features = self.model.forward(x)  # batch_size*frames x features_dimension
        return torch.stack(pred.split(n_frames, 0), dim=0), torch.stack(features.split(n_frames, 0), dim=0)


def train_model(model, train_loader, test_loader, device, args):
    print('Training RotNet Model...')
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005, momentum=0.9)
    loss = CrossEntropyLoss()
    for epoch in range(args.epochs):
        epoch_loss = train_epoch(model, train_loader, optimizer, loss, device, args)
        print(f'Epoch: {epoch + 1}, Loss: {epoch_loss}')
    test_loss = test_model(model, test_loader, device)
    print(f'Test Loss: {test_loss}')

    output_path = os.path.join(args.output_dir_path, get_output_filename(args))
    if not os.path.isdir(args.output_dir_path):
        os.mkdir(args.output_dir_path)
    print(f'Saving model to {output_path}')
    torch.save(model.state_dict(), output_path)


def train_epoch(model, train_loader, optimizer, loss_fn, device, args):
    running_loss = 0.0
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        y, _ = model(data)

        # print(f"y: {y}")
        # print(f"labels: {labels}")
        # print(f"{y0.type()}, {y1.type()}, {y2.type()}, {y3.type()}")
        # print(f"{y0}, {y1}, {y2}, {y3}")
        #
        # print(f"{target.type()}, {target_90.type()}, {target_180.type()}, {target_270.type()}")

        optimizer.zero_grad()
        loss = loss_fn(y, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / (i + 1)


def test_model(model, test_loader, device):
    avg_loss = 0.0
    loss_fn = CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            labels = labels.to(device)

            y, _ = model(data)

            loss = loss_fn(y, labels)
            avg_loss += loss.item()

    return avg_loss / (i + 1)


def get_lookup_table_paths(args):
    train_lookup_table_paths = str(args.train_lookup_table).split(',')
    test_lookup_table_paths = str(args.test_lookup_table).split(',')
    return train_lookup_table_paths, test_lookup_table_paths


def main(args):
    print(f"Running RotNet.py")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_lookup_table_paths, test_lookup_table_paths = get_lookup_table_paths(args)
    print("Got lookup tables")
    train_dataloader, test_dataloader = get_loaders3DSliced((train_lookup_table_paths, test_lookup_table_paths),
                                                            args.batch_size)
    print("Got dataloaders")
    model = utils.get_resnet_model(resnet_type=args.resnet_type, pretrained=False, num_classes=16)
    model = model.to(device)
    print("Got model")
    train_model(model, train_dataloader, test_dataloader, device, args)
    print("Done runing RotNet.py")


def get_output_filename(args):
    return '_'.join([args.output_model_filepath, str(args.epochs), str(args.batch_size), str(args.lr)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epochs', default=15, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='The initial learning rate.')
    parser.add_argument('--model', default='resnet')
    parser.add_argument('--resnet_type', default=152, type=int, help='which resnet to use')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--output_dir_path')
    parser.add_argument('--output_model_filepath')
    parser.add_argument('--train_lookup_table',
                        default="/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/8-frame-data-train-2/lookup_table.csv")
    parser.add_argument('--test_lookup_table',
                        default="/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/8-frame-data-test-2/lookup_table.csv")

    args = parser.parse_args()
    main(args)
