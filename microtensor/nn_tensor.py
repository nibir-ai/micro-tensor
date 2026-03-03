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
        K, S, P = self.kernel_size, self.stride, self.padding
        x_pad = np.pad(x.data, ((0,0), (0,0), (P,P), (P,P)), mode='constant') if P > 0 else x.data
        out_h = (H + 2*P - K) // S + 1
        out_w = (W + 2*P - K) // S + 1
        
        shape = (N, C, out_h, out_w, K, K)
        strides = (x_pad.strides[0], x_pad.strides[1], S*x_pad.strides[2], S*x_pad.strides[3], x_pad.strides[2], x_pad.strides[3])
        patches = np.lib.stride_tricks.as_strided(x_pad, shape=shape, strides=strides)
        
        input_matrix = Tensor(patches.transpose(0, 2, 3, 1, 4, 5).reshape(N, out_h*out_w, -1))
        kernels = self.weight.reshape((self.out_channels, -1))
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
        K, S = self.kernel_size, self.stride
        out_h, out_w = H // S, W // S
        shape = (N, C, out_h, out_w, K, K)
        strides = (x.data.strides[0], x.data.strides[1], S*x.data.strides[2], S*x.data.strides[3], x.data.strides[2], x.data.strides[3])
        patches = np.lib.stride_tricks.as_strided(x.data, shape=shape, strides=strides)
        res_data = np.max(patches, axis=(4, 5))
        out = Tensor(res_data, (x,), 'maxpool')
        def _backward():
            mask = (patches == res_data[:, :, :, :, None, None])
            grad_expanded = out.grad[:, :, :, :, None, None] * mask
            for i in range(out_h):
                for j in range(out_w):
                    x.grad[:, :, i*S:i*S+K, j*S:j*S+K] += grad_expanded[:, :, i, j, :, :]
        out._backward = _backward
        return out
    def parameters(self): return []

class Flatten(Module):
    def __call__(self, x): return x.reshape((x.data.shape[0], -1))

def cross_entropy(logits, target_indices):
    # Numerical stability: subtract max(logits)
    max_val = Tensor(np.max(logits.data, axis=1, keepdims=True))
    logits_stable = logits - max_val
    
    # LogSumExp
    sum_exp = logits_stable.exp().sum(axis=1, keepdims=True)
    log_sum_exp = sum_exp.log()
    
    # Negative Log Likelihood
    log_probs = logits_stable - log_sum_exp
    
    N = logits.data.shape[0]
    target_one_hot = np.zeros_like(logits.data)
    target_one_hot[np.arange(N), target_indices] = 1.0
    
    loss = (Tensor(target_one_hot) * log_probs).sum() * (-1.0 / N)
    return loss
