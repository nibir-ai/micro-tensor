import numpy as np
from microtensor.tensor import Tensor
from microtensor.nn_tensor import Conv2d

def test_conv_flow():
    print("=== Testing 4D Convolution Gradient Flow ===")
    
    # 1. Initialize data: (Batch=2, Channels=1, Height=5, Width=5)
    x = Tensor(np.random.randn(2, 1, 5, 5))
    
    # 2. Initialize Conv Layer: (In=1, Out=2, Kernel=3, Stride=1)
    conv = Conv2d(in_channels=1, out_channels=2, kernel_size=3)
    
    # 3. Forward Pass
    out = conv(x)
    
    # 4. Scalar Loss (Sum all outputs)
    loss = out.sum()
    
    # 5. Backward Pass
    loss.backward()
    
    # 6. Verification
    print(f"Input shape: {x.data.shape}")
    print(f"Output shape: {out.data.shape}")
    print(f"Kernel grad shape: {conv.weight.grad.shape}")
    
    expected_out_shape = (2, 2, 3, 3)
    
    if out.data.shape == expected_out_shape and conv.weight.grad is not None:
        print("\n✅ SUCCESS: 4D Tensor gradients are flowing through the convolution.")
        print(f"Kernel Gradient Mean: {np.mean(conv.weight.grad):.6f}")
    else:
        print("\n❌ FAILURE: Output shape mismatch or gradient interrupted.")

if __name__ == "__main__":
    test_conv_flow()
