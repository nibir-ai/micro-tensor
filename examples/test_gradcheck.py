from microtensor.tensor import Tensor
from microtensor.gradcheck import grad_check
import numpy as np

def test_ops():
    print("=== Running Numerical Gradient Check ===")
    
    # Define a complex computation: Loss = sum(tanh(X @ W + B)^2)
    X = Tensor(np.random.randn(2, 3), label='X')
    W = Tensor(np.random.randn(3, 2), label='W')
    B = Tensor(np.random.randn(1, 2), label='B')

    def forward(w_input):
        # We check gradients with respect to W
        out = (X @ w_input + B).tanh()
        loss = (out * out).sum()
        return loss

    # 1. Calculate Analytical Gradient
    loss = forward(W)
    loss.backward()
    analytic_grad = W.grad

    # 2. Calculate Numerical Gradient and get relative error
    error = grad_check(forward, W, analytic_grad)

    print(f"Relative Error for Weight Matrix: {error:.2e}")
    
    if error < 1e-6:
        print("✅ GRADIENT CHECK PASSED: Your backprop math is correct.")
    else:
        print("❌ GRADIENT CHECK FAILED: There is a bug in the calculus.")

if __name__ == "__main__":
    test_ops()
