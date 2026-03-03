import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, data=\n{self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            grad_self = out.grad
            grad_other = out.grad
            
            # Un-broadcast self
            while len(grad_self.shape) > len(self.data.shape):
                grad_self = grad_self.sum(axis=0)
            for i, dim in enumerate(self.data.shape):
                if dim == 1:
                    grad_self = grad_self.sum(axis=i, keepdims=True)
                    
            # Un-broadcast other
            while len(grad_other.shape) > len(other.data.shape):
                grad_other = grad_other.sum(axis=0)
            for i, dim in enumerate(other.data.shape):
                if dim == 1:
                    grad_other = grad_other.sum(axis=i, keepdims=True)

            self.grad += grad_self
            other.grad += grad_other
            
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self, other), '-')

        def _backward():
            grad_self = out.grad
            grad_other = -out.grad
            
            while len(grad_self.shape) > len(self.data.shape):
                grad_self = grad_self.sum(axis=0)
            for i, dim in enumerate(self.data.shape):
                if dim == 1:
                    grad_self = grad_self.sum(axis=i, keepdims=True)
                    
            while len(grad_other.shape) > len(other.data.shape):
                grad_other = grad_other.sum(axis=0)
            for i, dim in enumerate(other.data.shape):
                if dim == 1:
                    grad_other = grad_other.sum(axis=i, keepdims=True)

            self.grad += grad_self
            other.grad += grad_other
            
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')
        def _backward():
            self.grad += (1.0 - t**2) * out.grad
        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(np.sum(self.data), (self,), 'sum')
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
