import numpy as np
import matplotlib.pyplot as plt


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def plot_results(train_losses, validation_losses, train_accuracies, validation_accuracies):
    x_axis = [i for i in range(len(train_losses))]

    plt.plot(x_axis, train_losses, label='Train')
    plt.plot(x_axis, validation_losses, label='Validation')
    plt.title('Loss')
    plt.ylim(0, 0.1)
    plt.show()

    plt.plot(x_axis, train_accuracies, label='Train')
    plt.plot(x_axis, validation_accuracies, label='Validation')
    plt.title('Accuracy')
    plt.ylim(0, 1)
    plt.show()