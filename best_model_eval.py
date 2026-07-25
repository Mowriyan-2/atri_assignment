# question 7 - train the config we found from the sweep and check test accuracy

import numpy as np
import wandb

from src.data import load_fashion_mnist, preprocess, one_hot, train_val_split
from src.nn import NeuralNetwork
from src.losses import LOSSES
from src.optimizers import OPTIMIZERS

# best config found from the sweep in question 6
config = {
    "hidden_layers": 4,
    "hidden_size": 64,
    "activation": "relu",
    "weight_init": "xavier",
    "optimizer": "nadam",
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


(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = load_fashion_mnist()

X_train = preprocess(X_train_raw)
y_train_oh = one_hot(y_train_raw)
X_tr, y_tr, X_val, y_val = train_val_split(X_train, y_train_oh, val_fraction=0.1)

X_test = preprocess(X_test_raw)
y_test_oh = one_hot(y_test_raw)

run = wandb.init(project="atri-fashion-mnist", name="best_model_test_eval", config=config)

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
    print(f"epoch {epoch}: val_accuracy = {val_acc:.4f}")

# now the actual test set evaluation, this is the real result for question 7
test_probs, _ = model.forward(X_test)
test_preds = np.argmax(test_probs, axis=1)
test_acc = np.mean(test_preds == y_test_raw)

print(f"\nfinal test accuracy: {test_acc:.4f}")
wandb.log({"test_accuracy": test_acc})

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_test_raw,
        preds=test_preds,
        class_names=class_names,
    )
})

run.finish()
