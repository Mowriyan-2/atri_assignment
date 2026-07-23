# this script gets called by wandb sweep agent with different configs each time
# it trains one model and logs val loss/accuracy per epoch

import numpy as np
import wandb

from src.data import load_fashion_mnist, preprocess, one_hot, train_val_split
from src.nn import NeuralNetwork
from src.losses import LOSSES
from src.optimizers import OPTIMIZERS


def get_batches(X, y, batch_size):
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    for start in range(0, n, batch_size):
        b = idx[start:start + batch_size]
        yield X[b], y[b]


def train():
    run = wandb.init()
    config = wandb.config

    # give the run a name that actually tells you what config it used
    run.name = f"hl_{config.hidden_layers}_bs_{config.batch_size}_ac_{config.activation}_op_{config.optimizer}"

    hidden_layers_list = [config.hidden_size] * config.hidden_layers

    model = NeuralNetwork(
        input_size=784,
        hidden_layers=hidden_layers_list,
        output_size=10,
        activation=config.activation,
        weight_init=config.weight_init,
    )

    loss_fwd, loss_grad = LOSSES["cross_entropy"]

    optimizer = OPTIMIZERS[config.optimizer](
        model.get_params(),
        model.is_weight_mask(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    for epoch in range(config.epochs):
        train_loss_total = 0
        n_batches = 0

        for X_batch, y_batch in get_batches(X_tr, y_tr, config.batch_size):
            if optimizer.needs_lookahead():
                # nesterov needs the gradient computed at the lookahead point
                real_params = model.snapshot_params()
                model.set_params(optimizer.lookahead_params())
                probs, cache = model.forward(X_batch)
                grads = model.backward(cache, y_batch, loss_grad)
                model.set_params(real_params)
                optimizer.step(grads)
                probs, _ = model.forward(X_batch)
            else:
                probs, cache = model.forward(X_batch)
                grads = model.backward(cache, y_batch, loss_grad)
                optimizer.step(grads)

            train_loss_total += loss_fwd(probs, y_batch)
            n_batches += 1

        train_loss = train_loss_total / n_batches
        train_preds = model.predict(X_tr)
        train_acc = np.mean(train_preds == np.argmax(y_tr, axis=1))

        val_probs, _ = model.forward(X_val)
        val_loss = loss_fwd(val_probs, y_val)
        val_acc = np.mean(np.argmax(val_probs, axis=1) == np.argmax(y_val, axis=1))

        wandb.log({
            "epoch": epoch,
            "loss": train_loss,
            "accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

    run.finish()


if __name__ == "__main__":
    # load data once, keep it outside train() so the sweep agent doesn't
    # reload fashion mnist from scratch every single run
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = load_fashion_mnist()
    X_train = preprocess(X_train_raw)
    y_train_oh = one_hot(y_train_raw)
    X_tr, y_tr, X_val, y_val = train_val_split(X_train, y_train_oh, val_fraction=0.1)

    train()
