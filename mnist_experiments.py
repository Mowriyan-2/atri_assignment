# question 10 - apply what we learned from fashion mnist sweep to real mnist
# picked 3 configs with a limited budget instead of doing another full sweep

import numpy as np
import wandb

from src.data import preprocess, one_hot, train_val_split
from src.nn import NeuralNetwork
from src.losses import LOSSES
from src.optimizers import OPTIMIZERS


def load_mnist():
    from keras.datasets import mnist
    return mnist.load_data()


def get_batches(X, y, batch_size):
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    for start in range(0, n, batch_size):
        b = idx[start:start + batch_size]
        yield X[b], y[b]


def train_and_evaluate(config, X_tr, y_tr, X_val, y_val, X_test, y_test_raw):
    run = wandb.init(project="atri-fashion-mnist", name=f"mnist_{config['name']}", config=config, reinit=True)

    hidden_layers_list = [config["hidden_size"]] * config["hidden_layers"]
    model = NeuralNetwork(
        input_size=784,
        hidden_layers=hidden_layers_list,
        output_size=10,
        activation=config["activation"],
        weight_init=config["weight_init"],
    )

    loss_fwd, loss_grad = LOSSES["cross_entropy"]
    optimizer = OPTIMIZERS[config["optimizer"]](
        model.get_params(), model.is_weight_mask(),
        lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )

    for epoch in range(config["epochs"]):
        for X_batch, y_batch in get_batches(X_tr, y_tr, config["batch_size"]):
            if optimizer.needs_lookahead():
                real_params = model.snapshot_params()
                model.set_params(optimizer.lookahead_params())
                probs, cache = model.forward(X_batch)
                grads = model.backward(cache, y_batch, loss_grad)
                model.set_params(real_params)
                optimizer.step(grads)
            else:
                probs, cache = model.forward(X_batch)
                grads = model.backward(cache, y_batch, loss_grad)
                optimizer.step(grads)

        val_probs, _ = model.forward(X_val)
        val_acc = np.mean(np.argmax(val_probs, axis=1) == np.argmax(y_val, axis=1))
        wandb.log({"epoch": epoch, "val_accuracy": val_acc})

    test_probs, _ = model.forward(X_test)
    test_preds = np.argmax(test_probs, axis=1)
    test_acc = np.mean(test_preds == y_test_raw)

    print(f"{config['name']}: test_accuracy = {test_acc:.4f}")
    wandb.log({"test_accuracy": test_acc})
    run.finish()

    return test_acc


# 3 configs, budget limited so no new sweep here
# config 1: same as the best config we found for fashion mnist, since it worked well there
config1 = {
    "name": "config1_fashion_best",
    "hidden_layers": 4, "hidden_size": 64, "activation": "relu",
    "weight_init": "xavier", "optimizer": "nadam", "learning_rate": 0.001,
    "weight_decay": 0, "batch_size": 64, "epochs": 10,
}

# config 2: smaller and simpler, since mnist digits are easier to tell apart
# than fashion mnist clothes, so a smaller network might be enough
config2 = {
    "name": "config2_smaller",
    "hidden_layers": 3, "hidden_size": 32, "activation": "relu",
    "weight_init": "xavier", "optimizer": "adam", "learning_rate": 0.001,
    "weight_decay": 0, "batch_size": 64, "epochs": 5,
}

# config 3: same size as config1 but with rmsprop instead, since it also
# did well in the sweep and worth checking if it transfers to mnist too
config3 = {
    "name": "config3_rmsprop",
    "hidden_layers": 4, "hidden_size": 64, "activation": "relu",
    "weight_init": "xavier", "optimizer": "rmsprop", "learning_rate": 0.001,
    "weight_decay": 0, "batch_size": 64, "epochs": 10,
}

configs = [config1, config2, config3]

(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = load_mnist()
X_train = preprocess(X_train_raw)
y_train_oh = one_hot(y_train_raw)
X_tr, y_tr, X_val, y_val = train_val_split(X_train, y_train_oh, val_fraction=0.1)
X_test = preprocess(X_test_raw)

results = {}
for config in configs:
    acc = train_and_evaluate(config, X_tr, y_tr, X_val, y_val, X_test, y_test_raw)
    results[config["name"]] = acc

print("\nfinal results:")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")
