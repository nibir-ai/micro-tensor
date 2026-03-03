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
        self.bias = Tensor(np.zeros((1, out_channels, 1, 1))) # 4D for broadcasting

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
        
        # Matrix shape: (N, num_patches, in_channels * K * K)
        input_matrix = Tensor(np.array(patches)).transpose(1, 0, 2)
        
        # Batch MatMul with transposed kernels
        res = input_matrix @ kernels.transpose(1, 0)
        
        # Reshape to (N, Out_C, out_h, out_w)
        out = res.transpose(0, 2, 1).reshape((N, self.out_channels, out_h, out_w))
        return out + self.bias

    def parameters(self): return [self.weight, self.bias]

