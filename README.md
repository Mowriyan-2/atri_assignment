# Atri Assignment - Fashion MNIST Neural Network

This is my submission for the Atri AI assignment. I implemented a feedforward
neural network from scratch using numpy, along with backpropagation and 6
different optimizers, ran a hyperparameter sweep using wandb, and used the
best config to evaluate on the test set.

## What's done

- Q1: loading Fashion-MNIST and plotting one sample per class
- Q2: feedforward neural network with configurable hidden layers
- Q3: backpropagation + sgd, momentum, nesterov, rmsprop, adam, nadam
- Q4: wandb sweep across hyperparameters
- Q5: best val accuracy plot across all sweep runs
- Q6: sweep analysis + recommended config, in the wandb report
- Q7: trained the recommended config, test accuracy + confusion matrix
- Q8: compared cross entropy loss vs squared error loss
- Q9: repo cleanup
- Q10: applied learnings to real MNIST with 3 chosen configs

## Folder structure

```
src/
  __init__.py
  activations.py       - sigmoid, tanh, relu, softmax
  data.py              - loading fashion mnist + plotting sample images
  nn.py                - the neural network class (forward + backward pass)
  losses.py            - cross entropy and squared error loss
  optimizers.py        - all 6 optimizers

train.py               - script the wandb sweep agent runs for question 4
sweep.yaml             - sweep config, defines the search space and strategy
best_model_eval.py     - trains the best config from the sweep, evaluates on test set (question 7)
loss_comparison.py     - trains with cross entropy and squared error to compare (question 8)
mnist_experiments.py   - applies the learnings from the sweep to real MNIST (question 10)
colab_test.ipynb       - full colab notebook with all training runs and outputs
requirements.txt
```

## How to run

Install requirements:
```
pip install numpy pandas matplotlib keras wandb
```

Basic usage, just the network on its own:
```python
from src.data import load_fashion_mnist, plot_class_examples, preprocess
from src.nn import NeuralNetwork

(X_train, y_train), (X_test, y_test) = load_fashion_mnist()
plot_class_examples(X_train, y_train)

net = NeuralNetwork(input_size=784, hidden_layers=[128, 64, 32],
                     output_size=10, activation="relu", weight_init="xavier")
probs, cache = net.forward(preprocess(X_train[:32]))
```

Running the hyperparameter sweep:
```
wandb login
wandb sweep sweep.yaml
wandb agent <sweep id printed above>
```

Evaluating the best config on the actual test set, with confusion matrix:
```
python best_model_eval.py
```

Comparing cross entropy vs squared error loss:
```
python loss_comparison.py
```

Running the MNIST experiments:
```
python mnist_experiments.py
```

I ran everything in Google Colab since it's easier to use with wandb and
don't need to worry about GPU setup, `wandb.login()` needs to be run once
per session before any of the scripts above. `colab_test.ipynb` has the
full notebook with all the outputs from these runs.

## Wandb report

https://wandb.ai/mowriyan52-iit-kharagpur/atri-fashion-mnist/reports/Atri_Assignment--VmlldzoxNzU1OTMzNA?accessToken=gpa3m820k8903mfd9coje0cc183zx9ptasuzp5wsddjkbqz4bhebql9an00h8ewi

## GitHub

https://github.com/Mowriyan-2/atri_assignment
