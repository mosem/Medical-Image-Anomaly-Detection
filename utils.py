import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


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


def plot_results(train_losses, validation_losses, train_accuracies, validation_accuracies, dir_path):
    x_axis = [i for i in range(len(train_accuracies))]

    plt.plot(x_axis, train_losses['classification'], label='Train (Classification)')
    plt.plot(x_axis, train_losses['compactness'], label='Train (Compactness)')
    plt.plot(x_axis, train_losses['average'], label='Train (Average)')
    plt.plot(x_axis, validation_losses, label='Validation (Classification)')
    plt.title('Loss')
    y_max_lim = max(train_losses['average'][1:] + [0.05])
    plt.ylim(0, y_max_lim)
    plt.xticks(x_axis)
    plt.legend()
    plt.savefig(str(dir_path) + '/loss plot.png')
    plt.show()

    plt.plot(x_axis, train_accuracies, label='Train')
    plt.plot(x_axis, validation_accuracies, label='Validation')
    plt.title('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(x_axis)
    plt.legend()
    plt.savefig(str(dir_path) + '/accuracy plot.png')
    plt.show()


def normalize_features(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)

    return starts_from_zero / value_range


def get_tsne(features):
    tsne = TSNE(n_components=2).fit_transform(features)

    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = normalize_features(tx)
    ty = normalize_features(ty)

    return tx, ty


def plot_features(device, model, test_dataloader, dir_path):
    data, labels, raw_labels = next(iter(test_dataloader))
    model.eval()
    features = model.get_features(data.to(device)).cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    raw_labels = raw_labels.cpu().detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    tx, ty = get_tsne(features)

    colors = np.array(['red', 'blue', 'green', 'yellow', 'purple', 'cyan',
                       'magenta', 'pink', 'olive', 'brown'])
    class_names = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
                   5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    markers = ['o', '^']

    for raw_label in set(raw_labels):
        indices = [i for i, l in enumerate(raw_labels) if l == raw_label]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        normal_anomal_labels = labels[indices]

        assert all(normal_anomal_labels == 0) or all(normal_anomal_labels == 1)

        label = 0 if all(normal_anomal_labels == 0) else 1

        label_substr = 'normal' if all(normal_anomal_labels == 0) else 'anomal'
        label_str = f'{class_names[raw_label]}({label_substr})'

        ax.scatter(current_tx, current_ty, c=colors[raw_label],
                   label=label_str, marker=markers[label])

    ax.legend(loc='best')
    plt.savefig(str(dir_path) + '/tsne scatter.png')
    plt.show()






