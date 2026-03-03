import numpy as np
from microtensor.tensor import Tensor
from microtensor.nn_tensor import Embedding, PositionalEncoding

def test_sequence_input():
    print("=== Testing Sequence Pipeline (Embedding + PE) ===")
    
    vocab_size, embed_dim, seq_len = 10, 8, 5
    indices = np.array([[1, 2, 3, 2, 1]]) # Batch size 1
    
    embed = Embedding(vocab_size, embed_dim)
    pe = PositionalEncoding(max_seq_len=20, embed_dim=embed_dim)
    
    # 1. Map to vectors
    x_embed = embed(indices)
    
    # 2. Add position info
    x_final = pe(x_embed)
    
    # 3. Check shape and flow
    loss = x_final.sum()
    loss.backward()
    
    print(f"Input Shape: {indices.shape}")
    print(f"Output Shape: {x_final.data.shape}")
    
    # Verify that word index 1 at pos 0 and pos 4 have different final values
    pos0 = x_final.data[0, 0]
    pos4 = x_final.data[0, 4]
    
    diff = np.abs(pos0 - pos4).sum()
    print(f"Vector difference between same word at different positions: {diff:.4f}")
    
    if x_final.data.shape == (1, 5, 8) and diff > 0.1:
        print("\n✅ SUCCESS: Positional Encoding is injecting order into the sequence.")

if __name__ == "__main__":
    test_sequence_input()
