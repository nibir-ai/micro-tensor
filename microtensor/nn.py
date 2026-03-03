import random
from microtensor.engine import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin):
        # nin is the Number of Inputs. We create a random weight for each input.
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        # We create a single random bias
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # Forward pass: w * x + b
        # zip(self.w, x) pairs up each weight with its corresponding input
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]
