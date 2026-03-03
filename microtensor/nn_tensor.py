import numpy as np
from microtensor.tensor import Tensor

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)
    def parameters(self): return []

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.w = Tensor(np.random.randn(in_features, out_features) * scale)
        self.b = Tensor(np.zeros((1, out_features))) if bias else None
    def __call__(self, x):
        out = x @ self.w
        if self.b: out = out + self.b
        return out
    def parameters(self): return [self.w] + ([self.b] if self.b else [])

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        scale = np.sqrt(2.0 / (in_channels * kernel_size**2))
        self.weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale)
        self.bias = Tensor(np.zeros((1, out_channels, 1, 1)))

    def __call__(self, x):
        N, C, H, W = x.data.shape
        K = self.kernel_size
        out_h = (H + 2*self.padding - K) // self.stride + 1
        out_w = (W + 2*self.padding - K) // self.stride + 1
        kernels = self.weight.reshape((self.out_channels, -1))
        patches = []
        for i in range(out_h):
            for j in range(out_w):
                h_s, w_s = i * self.stride, j * self.stride
                patch = x.data[:, :, h_s:h_s+K, w_s:w_s+K].reshape(N, -1)
                patches.append(patch)
        input_matrix = Tensor(np.array(patches)).transpose(1, 0, 2)
        res = input_matrix @ kernels.transpose(1, 0)
        out = res.transpose(0, 2, 1).reshape((N, self.out_channels, out_h, out_w))
        return out + self.bias
    def parameters(self): return [self.weight, self.bias]

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
    def __call__(self, x):
        N, C, H, W = x.data.shape
        K = self.kernel_size
        out_h, out_w = H // self.stride, W // self.stride
        res_data = np.zeros((N, C, out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                h_s, w_s = i * self.stride, j * self.stride
                patch = x.data[:, :, h_s:h_s+K, w_s:w_s+K]
                res_data[:, :, i, j] = np.max(patch, axis=(2, 3))
        out = Tensor(res_data, (x,), 'maxpool')
        def _backward():
            for i in range(out_h):
                for j in range(out_w):
                    h_s, w_s = i * self.stride, j * self.stride
                    patch = x.data[:, :, h_s:h_s+K, w_s:w_s+K]
                    mask = (patch == res_data[:, :, i, j][:, :, None, None])
                    x.grad[:, :, h_s:h_s+K, w_s:w_s+K] += mask * out.grad[:, :, i, j][:, :, None, None]
        out._backward = _backward
        return out
    def parameters(self): return []

class Flatten(Module):
    def __call__(self, x):
        N = x.data.shape[0]
        return x.reshape((N, -1))

class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Linear(sz[i], sz[i+1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x).tanh()
        return x
    def parameters(self): return [p for layer in self.layers for p in layer.parameters()]
