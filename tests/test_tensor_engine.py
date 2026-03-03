import numpy as np
import pytest
from microtensor.tensor import Tensor
from microtensor.nn_tensor import Linear, MLP
from microtensor.optim_tensor import Adam

def test_tensor_add_broadcasting():
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([1, 1]) # Should broadcast to [[1, 1], [1, 1]]
    c = a + b
    assert np.allclose(c.data, [[2, 3], [4, 5]])
    
    c.backward()
    # Gradient of a should be 1s, gradient of b should be 2s (sum of rows)
    assert np.allclose(a.grad, [[1, 1], [1, 1]])
    assert np.allclose(b.grad, [2, 2])

def test_matmul_gradient():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    w = Tensor([[0.5, -0.5], [1.0, 0.0]])
    out = x @ w
    loss = out.sum()
    loss.backward()
    
    # dL/dw = x.T @ dL/dout
    # dL/dout is ones(2,2)
    expected_w_grad = x.data.T @ np.ones((2, 2))
    assert np.allclose(w.grad, expected_w_grad)

def test_mlp_inference():
    model = MLP(2, [16, 1])
    x = Tensor(np.random.randn(4, 2))
    out = model(x)
    assert out.data.shape == (4, 1)

def test_adam_optimizer():
    params = [Tensor([1.0, 2.0])]
    params[0].grad = np.array([0.1, -0.1])
    opt = Adam(params, lr=0.1)
    opt.step()
    # Check if data actually changed
    assert not np.allclose(params[0].data, [1.0, 2.0])
