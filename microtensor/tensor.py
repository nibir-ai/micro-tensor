import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op='', label=''):
        # Upgrade to float64 for numerical stability in gradient checking
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, data=\n{self.data})"

    def _unbroadcast(self, grad, target_shape):
        """Standard NumPy un-broadcasting logic."""
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

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')
        def _backward():
            # Analytical Gradients for Matrix Mult:
            # dL/dA = dL/dY @ B.T
            # dL/dB = A.T @ dL/dY
            self.grad += self._unbroadcast(out.grad @ other.data.T, self.data.shape)
            other.grad += self._unbroadcast(self.data.T @ out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only int/float powers supported"
        out = Tensor(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += self._unbroadcast((other * (self.data ** (other - 1))) * out.grad, self.data.shape)
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')
        def _backward():
            self.grad += self._unbroadcast((1.0 - t**2) * out.grad, self.data.shape)
        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(np.sum(self.data), (self,), 'sum')
        def _backward():
            # Gradient of sum is 1.0 everywhere, scaled by the incoming gradient
            self.grad += self._unbroadcast(np.ones_like(self.data) * out.grad, self.data.shape)
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

        # MANDATORY: Zero all gradients in the graph first
        for node in topo:
            node.grad = np.zeros_like(node.data)

        # Set the seed gradient (dL/dL = 1)
        self.grad = np.ones_like(self.data)
        
        # Backpropagate through the topological sort
        for node in reversed(topo):
            node._backward()
