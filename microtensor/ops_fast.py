import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def fast_softmax(x):
    """
    JIT-compiled Softmax with parallel execution.
    Fuses the Max, Sub, Exp, and Sum operations into a single pass.
    """
    # x is expected to be 2D for this optimized helper (N*L, V)
    N, V = x.shape
    out = np.zeros_like(x)
    for i in prange(N):
        row = x[i, :]
        max_val = np.max(row)
        # Calculate exp(x - max)
        exp_row = np.exp(row - max_val)
        sum_exp = np.sum(exp_row)
        out[i, :] = exp_row / sum_exp
    return out

@njit(fastmath=True)
def fast_gelu(x):
    """
    GELU activation is used in GPT models. It's more expensive than Tanh
    but better for deep Transformers. JIT makes it 'free'.
    """
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
