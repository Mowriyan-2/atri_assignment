# question 8 - compare cross entropy loss vs squared error loss
# same architecture and optimizer for both, only the loss function changes

import numpy as np
import wandb

from src.data import load_fashion_mnist, preprocess, one_hot, train_val_split
from src.nn import NeuralNetwork
from src.losses import LOSSES
from src.optimizers import OPTIMIZERS

config = {
    "hidden_layers": 3,
    "hidden_size": 64,
    "activation": "relu",
    "weight_init": "xavier",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "weight_decay": 0,
    "batch_size": 64,
    "epochs": 10,
}


def get_batches(X, y, batch_size):
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    for start in range(0, n, batch_size):
        b = idx[start:start + batch_size]
        yield X[b], y[b]


def run_training(loss_name):
    loss_fwd, loss_grad = LOSSES[loss_name]

    run = wandb.init(project="atri-fashion-mnist", name=f"loss_compare_{loss_name}",
                      config=config, reinit=True)

    hidden_layers_list = [config["hidden_size"]] * config["hidden_layers"]
    model = NeuralNetwork(
        input_size=784,
        hidden_layers=hidden_layers_list,
        output_size=10,
        activation=config["activation"],
        weight_init=config["weight_init"],
    )
    optimizer = OPTIMIZERS[config["optimizer"]](
        model.get_params(), model.is_weight_mask(),
        lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )

    for epoch in range(config["epochs"]):
        for X_batch, y_batch in get_batches(X_tr, y_tr, config["batch_size"]):
            probs, cache = model.forward(X_batch)
            grads = model.backward(cache, y_batch, loss_grad)
            optimizer.step(grads)

        val_probs, _ = model.forward(X_val)
        val_loss = loss_fwd(val_probs, y_val)
        val_acc = np.mean(np.argmax(val_probs, axis=1) == np.argmax(y_val, axis=1))
        wandb.log({"epoch": epoch, "val_loss": val_loss, "val_accuracy": val_acc})
        print(f"{loss_name} epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    run.finish()


(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = load_fashion_mnist()
X_train = preprocess(X_train_raw)
y_train_oh = one_hot(y_train_raw)
X_tr, y_tr, X_val, y_val = train_val_split(X_train, y_train_oh, val_fraction=0.1)

run_training("cross_entropy")
run_training("squared_error")
