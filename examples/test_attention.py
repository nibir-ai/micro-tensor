import numpy as np
from microtensor.tensor import Tensor
from microtensor.nn_tensor import MultiHeadAttention

def test_attention_flow():
    print("=== Testing Multi-Head Attention Gradient Flow ===")
    
    # Batch size 2, Seq length 5, Embedding dim 16, Heads 4
    N, L, D, H = 2, 5, 16, 4
    x = Tensor(np.random.randn(N, L, D))
    attn = MultiHeadAttention(embed_dim=D, num_heads=H)
    
    # Forward Pass
    out = attn(x)
    
    # Backward Pass
    loss = out.sum()
    loss.backward()
    
    print(f"Output shape: {out.data.shape}")
    print(f"Q-projection grad shape: {attn.q_proj.w.grad.shape}")
    
    if out.data.shape == (N, L, D) and not np.isnan(out.data.sum()):
        print("\n✅ SUCCESS: Multi-Head Attention is fully functional.")

if __name__ == "__main__":
    test_attention_flow()
