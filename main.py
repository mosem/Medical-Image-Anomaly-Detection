import torch

from datasets import getCifarSmallImbalancedDatasets
from utils import plot_results
from modelUtils import *



if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    train_set, validation_set, test_set = getCifarSmallImbalancedDatasets(5)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=20,
                                              shuffle=True, num_workers=2)
    validation_dataloader = torch.utils.data.DataLoader(validation_set, batch_size=20,
                                              shuffle=True, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=20,
                                              shuffle=False, num_workers=2)

    model = build_model()
    model.to(device)


    train_losses, train_accuracies, validation_losses, validation_accuracies = train_model(train_dataloader, validation_dataloader, model, num_epochs=25)
    print('Done training.')
    print('===================================\nTest results:')
    test_loop(test_dataloader, model, torch.nn.CrossEntropyLoss(), device)

    plot_results(train_losses, train_accuracies, validation_losses, validation_accuracies)