# Loss functions and their gradients w.r.t logits for Fashion-MNIST 

import numpy as np


def cross_entropy_forward(y_pred, y_true):
    # compute average cross entropy loss
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def cross_entropy_grad_logits(y_pred, y_true):
    # gradient of cross entropy w.r.t pre-softmax logits
    batch_size = y_pred.shape[0]
    return (y_pred - y_true) / batch_size


def squared_error_forward(y_pred, y_true):
    # mean squared error loss
    return np.mean(np.sum((y_pred - y_true) ** 2, axis=1))


def squared_error_grad_logits(y_pred, y_true):
    # gradient of MSE w.r.t pre-softmax logits using softmax jacobian
    batch_size = y_pred.shape[0]
    dL_dy = 2 * (y_pred - y_true) / y_pred.shape[1]

    dot = np.sum(dL_dy * y_pred, axis=1, keepdims=True)
    dL_dz = y_pred * (dL_dy - dot)
    return dL_dz / batch_size


LOSSES = {
    "cross_entropy": (cross_entropy_forward, cross_entropy_grad_logits),
    "squared_error": (squared_error_forward, squared_error_grad_logits),
}