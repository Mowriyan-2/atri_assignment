# Optimizers for neural network training 

import numpy as np


class Optimizer:
    def __init__(self, params, is_weight, lr=1e-3, weight_decay=0.0):
        self.params = params            
        self.is_weight = is_weight      
        self.lr = lr
        self.weight_decay = weight_decay

    def _apply_weight_decay(self, grads):
        # add L2 penalty (weight_decay * weight) only to weights, skip biases
        out = []
        for g, p, is_w in zip(grads, self.params, self.is_weight):
            if is_w and self.weight_decay > 0:
                out.append(g + self.weight_decay * p)
            else:
                out.append(g)
        return out

    def step(self, grads):
        raise NotImplementedError

    def needs_lookahead(self):
        return False


class SGD(Optimizer):
    def step(self, grads):
        grads = self._apply_weight_decay(grads)
        for p, g in zip(self.params, grads):
            p -= self.lr * g


class Momentum(Optimizer):
    def __init__(self, params, is_weight, lr=1e-3, weight_decay=0.0, momentum=0.9):
        super().__init__(params, is_weight, lr, weight_decay)
        self.momentum = momentum
        self.velocity = [np.zeros_like(p) for p in params]

    def step(self, grads):
        grads = self._apply_weight_decay(grads)
        for i, (p, g) in enumerate(zip(self.params, grads)):
            # build up velocity to power through flat regions
            self.velocity[i] = self.momentum * self.velocity[i] + self.lr * g
            p -= self.velocity[i]


class Nesterov(Optimizer):
    def __init__(self, params, is_weight, lr=1e-3, weight_decay=0.0, momentum=0.9):
        super().__init__(params, is_weight, lr, weight_decay)
        self.momentum = momentum
        self.velocity = [np.zeros_like(p) for p in params]

    def needs_lookahead(self):
        return True

    def lookahead_params(self):
        # peek ahead by our current velocity before calculating the gradient
        return [p - self.momentum * v for p, v in zip(self.params, self.velocity)]

    def step(self, grads):
        grads = self._apply_weight_decay(grads)
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.velocity[i] = self.momentum * self.velocity[i] + self.lr * g
            p -= self.velocity[i]


class RMSprop(Optimizer):
    def __init__(self, params, is_weight, lr=1e-3, weight_decay=0.0, beta=0.9, eps=1e-8):
        super().__init__(params, is_weight, lr, weight_decay)
        self.beta = beta
        self.eps = eps
        # cache holds the moving average of squared gradients
        self.cache = [np.zeros_like(p) for p in params]

    def step(self, grads):
        grads = self._apply_weight_decay(grads)
        for i, (p, g) in enumerate(zip(self.params, grads)):
            # downscale step size if gradient has been consistently huge
            self.cache[i] = self.beta * self.cache[i] + (1 - self.beta) * (g ** 2)
            p -= self.lr * g / (np.sqrt(self.cache[i]) + self.eps)


class Adam(Optimizer):
    def __init__(self, params, is_weight, lr=1e-3, weight_decay=0.0,
                 beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params, is_weight, lr, weight_decay)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params] # first moment (momentum)
        self.v = [np.zeros_like(p) for p in params] # second moment (rmsprop scale)
        self.t = 0

    def step(self, grads):
        grads = self._apply_weight_decay(grads)
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            
            # bias correction to prevent tiny steps at the very beginning
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class Nadam(Optimizer):
    def __init__(self, params, is_weight, lr=1e-3, weight_decay=0.0,
                 beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params, is_weight, lr, weight_decay)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        grads = self._apply_weight_decay(grads)
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            
            # folds nesterov's momentum lookahead directly into adam's m_hat math
            # saves us from having to do a separate forward/backward pass here
            m_hat = (self.beta1 * self.m[i] / (1 - self.beta1 ** (self.t + 1))) + \
                    ((1 - self.beta1) * g / (1 - self.beta1 ** self.t))
            
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


OPTIMIZERS = {
    "sgd": SGD,
    "momentum": Momentum,
    "nesterov": Nesterov,
    "rmsprop": RMSprop,
    "adam": Adam,
    "nadam": Nadam,
}