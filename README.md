# Atri Assignment - Fashion MNIST Neural Network

This is my submission for the Atri AI assignment. I implemented a feedforward
neural network from scratch using numpy, along with backpropagation and 6
different optimizers, and used wandb to track experiments.

## What's done so far

- Q1: loading Fashion-MNIST and plotting one sample per class
- Q2: feedforward neural network with configurable hidden layers
- Q3: backpropagation + sgd, momentum, nesterov, rmsprop, adam, nadam

Still working on the hyperparameter sweep and the rest of the questions.

## Folder structure

```
src/
  activations.py   - sigmoid, tanh, relu, softmax
  data.py          - loading fashion mnist + plotting sample images
  nn.py            - the neural network class (forward + backward pass)
  losses.py        - cross entropy and squared error loss
  optimizers.py    - all 6 optimizers
```

## How to run

Install requirements:
```
pip install numpy pandas matplotlib keras wandb
```

Basic usage:
```python
from src.data import load_fashion_mnist, plot_class_examples, preprocess
from src.nn import NeuralNetwork

(X_train, y_train), (X_test, y_test) = load_fashion_mnist()
plot_class_examples(X_train, y_train)

net = NeuralNetwork(input_size=784, hidden_layers=[128, 64, 32],
                     output_size=10, activation="relu", weight_init="xavier")
probs, cache = net.forward(preprocess(X_train[:32]))
```

I mostly ran everything in Google Colab since it's easier to use with wandb
and don't need to worry about GPU setup.

## Wandb report

https://wandb.ai/mowriyan52-iit-kharagpur/atri-fashion-mnist/reports/Atri_Assignment--VmlldzoxNzU1OTMzNA?accessToken=gpa3m820k8903mfd9coje0cc183zx9ptasuzp5wsddjkbqz4bhebql9an00h8ewi

## GitHub

link to be added
