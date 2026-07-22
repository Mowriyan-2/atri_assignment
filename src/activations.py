# Activation functions and their derivatives (Q2 & Q3)

import numpy as np


def sigmoid(x):
    # clip to avoid overflow in np.exp for very negative values
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    t = np.tanh(x)
    return 1 - t ** 2


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(x.dtype)


def softmax(x):
    # row-wise softmax, subtracting max for numerical stability
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# dictionary mapping for easy lookup in nn.py
ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "relu": (relu, relu_derivative),
}