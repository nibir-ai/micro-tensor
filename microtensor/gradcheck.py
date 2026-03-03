import numpy as np
from microtensor.tensor import Tensor

def rel_error(x, y):
    """ returns relative error between x and y """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def grad_check(f, x, analytic_grad, h=1e-5):
    """
    f: a function that takes a Tensor x and returns a scalar Tensor
    x: the Tensor input
    analytic_grad: the gradient calculated by x.backward()
    h: the small perturbation
    """
    numeric_grad = np.zeros_like(x.data)
    
    # Iterate over every element in the tensor
    it = np.nditer(x.data, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_val = x.data[idx]
        
        # f(x + h)
        x.data[idx] = old_val + h
        fxph = f(x).data
        
        # f(x - h)
        x.data[idx] = old_val - h
        fxmh = f(x).data
        
        # (f(x+h) - f(x-h)) / 2h
        numeric_grad[idx] = (fxph - fxmh) / (2 * h)
        
        x.data[idx] = old_val # reset
        it.iternext()
        
    error = rel_error(numeric_grad, analytic_grad)
    return error
