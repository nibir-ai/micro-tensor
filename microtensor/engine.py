class Value:
    """
    A scalar value that tracks its computational history for automatic differentiation.
    """
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0 
        self._backward = lambda: None 
        self._prev = set(_children) 
        self._op = _op 
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        # 1. If 'other' is just a normal number (like 3), wrap it in a Value object so it can join the graph.
        other = other if isinstance(other, Value) else Value(other)
        
        # 2. Calculate the actual math (self.data + other.data).
        # 3. Pass in the parents (self, other) to record where this came from.
        # 4. Pass in the '+' string to record the operation.
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        return out

    # The "Reverse" Dunder Methods
    # What happens if we write `2 + Value(3)`? Python will try to call 2.__add__(Value(3)). 
    # Since '2' is a native integer, it doesn't know what a 'Value' is and will crash.
    # If standard __add__ fails, Python falls back to __radd__ (reverse add) on the second object.
    def __radd__(self, other): # other + self
        return self + other

    def __rmul__(self, other): # other * self
        return self * other
