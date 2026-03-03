import numpy as np
from microtensor.tensor import Tensor
from microtensor.nn_tensor import TransformerBlock

def test_full_block():
    print("=== Testing Full Transformer Block Gradient Flow ===")
    
    # N=1, L=4, D=8, Heads=2, FF=16
    x = Tensor(np.random.randn(1, 4, 8))
    block = TransformerBlock(embed_dim=8, num_heads=2, ff_dim=16)
    
    # Forward
    out = block(x)
    
    # Backward
    loss = out.sum()
    loss.backward()
    
    print(f"Input shape: {x.data.shape}")
    print(f"Output shape: {out.data.shape}")
    
    # Check if a parameter from the deep end got a gradient
    attn_grad_sum = np.sum(np.abs(block.attn.q_proj.w.grad))
    print(f"Attention Weight Gradient Sum: {attn_grad_sum:.6f}")
    
    if out.data.shape == (1, 4, 8) and attn_grad_sum > 0:
        print("\n✅ SUCCESS: Full Transformer Block is integrated and learning.")

if __name__ == "__main__":
    test_full_block()
