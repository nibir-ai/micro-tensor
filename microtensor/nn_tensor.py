import numpy as np
from microtensor.tensor import Tensor

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        return []

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        # Xavier/Glorot Initialization for better convergence
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.w = Tensor(np.random.randn(in_features, out_features) * scale, label='weight')
        self.b = Tensor(np.zeros((1, out_features)), label='bias') if bias else None

    def __call__(self, x):
        out = x @ self.w
        if self.b:
            out = out + self.b
        return out

    def parameters(self):
        return [self.w] + ([self.b] if self.b else [])

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize kernels: (out_channels, in_channels, k, k)
        scale = np.sqrt(2.0 / (in_channels * kernel_size**2))
        self.weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale)
        self.bias = Tensor(np.zeros((out_channels, 1, 1)))

    def __call__(self, x):
        # x shape: (N, C, H, W)
        # Note: For now, we are implementing a simplified version for the portfolio
        # A full Im2Col is complex, so we will start with the functional definition
        # We will refine the internal math in the next session.
        return self.forward(x)

    def forward(self, x):
        # We will implement the high-performance sliding window here next.
        # This is a placeholder to verify the structure.
        return x 

    def parameters(self):
        return [self.weight, self.bias]

class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Linear(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x).tanh()
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

