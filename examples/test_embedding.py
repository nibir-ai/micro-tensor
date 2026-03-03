import numpy as np
from microtensor.tensor import Tensor
from microtensor.nn_tensor import Embedding

def test_embedding_engine():
    print("=== Testing Embedding Layer & Gradient Accumulation ===")
    
    # 1. Setup: Vocab of 5 words, Embedding size of 2
    # This means weight matrix is (5, 2)
    embed = Embedding(vocab_size=5, embed_dim=2)
    
    # 2. Input: A sequence where index '1' appears twice
    # Indices: [1, 3, 1]
    indices = np.array([1, 3, 1])
    
    # 3. Forward
    out = embed(indices)
    
    # 4. Loss: sum of all elements
    loss = out.sum()
    loss.backward()
    
    # 5. The Verification Logic:
    # Since index '1' appeared twice, its gradient should be 2.0 (1.0 + 1.0)
    # Since index '3' appeared once, its gradient should be 1.0
    # Since index '0, 2, 4' were never used, their gradients should be 0.0
    
    g1 = embed.weight.grad[1]
    g3 = embed.weight.grad[3]
    g0 = embed.weight.grad[0]
    
    print(f"Grad for index 1 (used twice): {g1}")
    print(f"Grad for index 3 (used once):  {g3}")
    print(f"Grad for index 0 (unused):     {g0}")
    
    passed = np.all(g1 == 2.0) and np.all(g3 == 1.0) and np.all(g0 == 0.0)
    
    if passed:
        print("\n✅ SUCCESS: Embedding gradients are accumulating perfectly.")
    else:
        print("\n❌ FAILURE: Check the 'np.add.at' logic.")

if __name__ == "__main__":
    test_embedding_engine()
