import numpy as np
from microtensor.tensor import Tensor
from microtensor.nn_tensor import Conv2d

def test_conv_flow():
    print("=== Testing 4D Convolution Gradient Flow ===")
    # Batch=1, Channel=1, H=5, W=5
    x = Tensor(np.random.randn(1, 1, 5, 5))
    conv = Conv2d(in_channels=1, out_channels=1, kernel_size=3)
    
    out = conv(x)
    loss = out.sum()
    loss.backward()
    
    print(f"Input shape: {x.data.shape}")
    print(f"Output shape: {out.data.shape}")
    print(f"Gradient of kernel exists: {conv.weight.grad is not None}")
    print(f"Gradient sum: {np.sum(conv.weight.grad):.4f}")
    
    if conv.weight.grad is not None and out.data.shape == (1, 1, 3, 3):
        print("✅ SUCCESS: 4D Tensor gradients are flowing through the convolution.")
    else:
        print("❌ FAILURE: Gradient flow interrupted.")

if __name__ == "__main__":
    test_conv_flow()
