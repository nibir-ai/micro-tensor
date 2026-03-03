class Value:
    """
    A scalar value that tracks its computational history for automatic differentiation.
    """
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0 # The derivative of the loss with respect to this value
        self._backward = lambda: None # Closure that calculates the local gradient
        self._prev = set(_children) # Set of nodes that created this node
        self._op = _op # The operation that produced this node
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
