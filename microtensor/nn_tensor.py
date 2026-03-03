import numpy as np
from microtensor.tensor import Tensor

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        return []

class Linear(Module):
    def __init__(self, nin, nout):
        # We initialize a single Weight matrix of shape (nin, nout)
        self.w = Tensor(np.random.uniform(-1, 1, (nin, nout)))
        # We initialize a single Bias matrix of shape (1, nout)
        self.b = Tensor(np.random.uniform(-1, 1, (1, nout)))

    def __call__(self, x):
        # Forward pass: X @ W + b
        # If x is shape (1, nin) and w is (nin, nout), the result is (1, nout)
        act = x @ self.w + self.b
        return act.tanh()

    def parameters(self):
        return [self.w, self.b]

class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Linear(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
