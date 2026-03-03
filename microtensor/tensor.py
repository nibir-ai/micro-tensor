import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op})"

    def _unbroadcast(self, grad, target_shape):
        out_grad = grad
        while len(out_grad.shape) > len(target_shape):
            out_grad = out_grad.sum(axis=0)
        for i, dim in enumerate(target_shape):
            if dim == 1:
                out_grad = out_grad.sum(axis=i, keepdims=True)
        return out_grad

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += self._unbroadcast(out.grad, self.data.shape)
            other.grad += self._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self, other), '-')
        def _backward():
            self.grad += self._unbroadcast(out.grad, self.data.shape)
            other.grad -= self._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += self._unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += self._unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __rmul__(self, other): return self * other

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')
        def _backward():
            def swap(x): return np.swapaxes(x, -1, -2) if x.ndim >= 2 else x
            self.grad += self._unbroadcast(out.grad @ swap(other.data), self.data.shape)
            other.grad += self._unbroadcast(swap(self.data) @ out.grad, other.data.shape)
        out._backward = _backward
        return out

    def exp(self):
        res = np.exp(self.data)
        out = Tensor(res, (self,), 'exp')
        def _backward():
            self.grad += res * out.grad
        out._backward = _backward
        return out

    def log(self):
        res = np.log(self.data + 1e-12)
        out = Tensor(res, (self,), 'log')
        def _backward():
            self.grad += (1.0 / (self.data + 1e-12)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')
        def _backward():
            self.grad += (1.0 - t**2) * out.grad
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        res = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(res, (self,), 'sum')
        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            self.grad += np.ones_like(self.data) * grad
        out._backward = _backward
        return out

    def reshape(self, shape):
        out = Tensor(self.data.reshape(shape), (self,), 'reshape')
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def transpose(self, *axes):
        out = Tensor(self.data.transpose(*axes), (self,), 'transpose')
        def _backward():
            inv_axes = np.argsort(axes)
            self.grad += out.grad.transpose(*inv_axes)
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev: build_topo(child)
                topo.append(v)
        build_topo(self)
        for node in topo: node.grad = np.zeros_like(node.data)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo): node._backward()
