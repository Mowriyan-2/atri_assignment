# Feedforward Neural Network class for Fashion-MNIST (Q2 & Q3)
# Handles dynamic hidden layers, forward pass, and backpropagation gradients

import numpy as np
from src.activations import ACTIVATIONS, softmax


class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size,
                 activation="relu", weight_init="xavier", seed=42):
        if activation not in ACTIVATIONS:
            raise ValueError("Invalid activation function selected")

        self.activation_name = activation
        self.activation_fn, self.activation_deriv = ACTIVATIONS[activation]
        self.layer_sizes = [input_size] + list(hidden_layers) + [output_size]
        self.num_layers = len(self.layer_sizes) - 1

        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []

        for i in range(self.num_layers):
            fan_in, fan_out = self.layer_sizes[i], self.layer_sizes[i + 1]
            
            if weight_init == "xavier":
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                w = rng.uniform(-limit, limit, size=(fan_in, fan_out))
            elif weight_init == "random":
                w = rng.normal(0, 0.01, size=(fan_in, fan_out))
            else:
                raise ValueError("unknown init method")
                
            self.weights.append(w)
            self.biases.append(np.zeros((1, fan_out)))
    def forward(self, X):
        # returns probabilities and cache for backprop
        activations = [X]      
        pre_activations = []   

        a = X
        for i in range(self.num_layers):
            z = a @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            # print(z.shape) # debugging dim error

            if i == self.num_layers - 1:
                a = softmax(z)          
            else:
                a = self.activation_fn(z)  

            activations.append(a)

        cache = {"activations": activations, "pre_activations": pre_activations}
        return a, cache

    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def num_parameters(self):
        return sum(w.size for w in self.weights) + sum(b.size for b in self.biases)

    # get params as a flat list W1, b1, W2, b2... for the optimizer
    # mask tells the optimizer which are weights (for L2) vs biases
    def get_params(self):
        params = []
        for w, b in zip(self.weights, self.biases):
            params.append(w)
            params.append(b)
        return params

    def is_weight_mask(self):
        mask = []
        for _ in range(self.num_layers):
            mask.append(True)   # weight matrix
            mask.append(False)  # bias vector
        return mask

    def set_params(self, flat_params):
        """Overwrites weights/biases in-place from a flat [W1,b1,W2,b2,...] list."""
        for i in range(self.num_layers):
            self.weights[i][...] = flat_params[2 * i]
            self.biases[i][...] = flat_params[2 * i + 1]

    def snapshot_params(self):
        """Deep copy of current params, for temporarily shifting (Nesterov lookahead)."""
        return [p.copy() for p in self.get_params()]

    def backward(self, cache, y_true, loss_grad_fn):
        # computes gradients w.r.t weights and biases using cache
        activations = cache["activations"]          
        pre_activations = cache["pre_activations"]  
        y_pred = activations[-1]

        grads_W = [None] * self.num_layers       
        grads_b = [None] * self.num_layers
        # Gradient w.r.t. the OUTPUT layer's pre-activation (logits).        
        # loss_grad_fn already returns dL/dz_L directly (works for both        
        # cross-entropy and squared-error, see losses.py).        
        delta = loss_grad_fn(y_pred, y_true)  # (batch, output_size)
        for layer in reversed(range(self.num_layers)):
            a_prev = activations[layer]  # activation feeding INTO this layer
            grads_W[layer] = a_prev.T @ delta
            grads_b[layer] = np.sum(delta, axis=0, keepdims=True)

            if layer > 0:
                # propagate delta back through this layer's weights,
                # then through the previous layer's activation derivative
                da_prev = delta @ self.weights[layer].T
                z_prev = pre_activations[layer - 1]
                delta = da_prev * self.activation_deriv(z_prev)

        # interleave into flat [dW1,db1,dW2,db2,...] to match get_params()
        grads = []
        for gW, gb in zip(grads_W, grads_b):
            grads.append(gW)
            grads.append(gb)
        return grads


if __name__ == "__main__":
    # Quick sanity check: random batch through a 3-hidden-layer network
    net = NeuralNetwork(input_size=784, hidden_layers=[128, 64, 32],
                         output_size=10, activation="relu", weight_init="xavier")
    X_dummy = np.random.rand(5, 784)
    probs, _ = net.forward(X_dummy)
    print("Output shape:", probs.shape)          # (5, 10)
    print("Row sums (should be ~1):", probs.sum(axis=1))
    print("Total parameters:", net.num_parameters())