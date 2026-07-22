# Data loading and preprocessing utilities for Fashion-MNIST (Q1)

import numpy as np
import matplotlib.pyplot as plt

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def load_fashion_mnist():
    # load fashion mnist using keras (allowed for data loading)
    from keras.datasets import fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    return (X_train, y_train), (X_test, y_test)


def preprocess(X, flatten=True, normalize=True):
    # scale pixels to [0, 1] and flatten 28x28 images to 784
    X = X.astype(np.float64)
    if normalize:
        X = X / 255.0
    if flatten:
        X = X.reshape(X.shape[0], -1)
    return X


def one_hot(y, num_classes=10):
    out = np.zeros((y.shape[0], num_classes))
    out[np.arange(y.shape[0]), y] = 1
    return out


def train_val_split(X, y, val_fraction=0.1, seed=42):
    # random 90-10 train/val split
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def plot_class_examples(X, y, class_names=CLASS_NAMES, log_to_wandb=False):
    # Q1: plot 1 sample image per class in a 2x5 grid
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    fig.suptitle("examples")
    fig.subplots_adjust(hspace=0.4, top=0.88)

    for class_idx in range(10):
        row = class_idx // 5
        col = class_idx % 5
        ax = axes[row, col]
        sample_idx = np.where(y == class_idx)[0][0]
        ax.imshow(X[sample_idx], cmap="gray")
        ax.set_title(class_names[class_idx], fontsize=10)
        ax.axis("off")

    if log_to_wandb:
        import wandb
        wandb.log({"examples": wandb.Image(fig)})

    return fig


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_fashion_mnist()
    plot_class_examples(X_train, y_train)
    plt.show()