import numpy as np
from microtensor.tensor import Tensor
from microtensor.nn_tensor import Conv2d, MaxPool2d, Flatten, Linear

def test_full_cnn():
    print("=== Testing Full CNN Integration ===")
    # 1. Input: (1 Batch, 1 Channel, 28x28 Image like MNIST)
    x = Tensor(np.random.randn(1, 1, 28, 28))
    
    # 2. Architect
    conv = Conv2d(1, 4, kernel_size=3, padding=0) # Out: (1, 4, 26, 26)
    pool = MaxPool2d(kernel_size=2)               # Out: (1, 4, 13, 13)
    flat = Flatten()                              # Out: (1, 4*13*13) = (1, 676)
    fc = Linear(676, 10)                          # Out: (1, 10)
    
    # 3. Forward Pass
    x1 = conv(x).tanh()
    x2 = pool(x1)
    x3 = flat(x2)
    logits = fc(x3)
    
    # 4. Backward Pass
    loss = logits.sum()
    loss.backward()
    
    print(f"Final Output Shape: {logits.data.shape}")
    print(f"Convolution Weight Grad Sum: {np.sum(conv.weight.grad):.4f}")
    print(f"Linear Weight Grad Sum: {np.sum(fc.w.grad):.4f}")
    
    if logits.data.shape == (1, 10) and not np.isnan(np.sum(conv.weight.grad)):
        print("\n✅ CNN ARCHITECTURE VALIDATED: Ready for MNIST Training.")

if __name__ == "__main__":
    test_full_cnn()
