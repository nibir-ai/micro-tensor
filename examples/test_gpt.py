import numpy as np
from microtensor.tensor import Tensor
from microtensor.nn_tensor import LanguageModel, cross_entropy

def test_language_model():
    print("=== Testing Full Language Model (GPT-Micro) Flow ===")
    
    # Vocab: 5 chars, Embed: 16, Heads: 2, Layers: 2
    model = LanguageModel(vocab_size=5, embed_dim=16, num_heads=2, ff_dim=32, num_layers=2, max_seq_len=10)
    
    # Input: Batch 1, Sequence Length 4
    idx = np.array([[0, 1, 2, 3]])
    targets = np.array([[1, 2, 3, 4]]) # Predict next character
    
    # Forward
    logits = model(idx)
    loss = cross_entropy(logits, targets)
    
    # Backward
    loss.backward()
    
    print(f"Logits shape: {logits.data.shape}")
    print(f"Loss: {loss.data:.4f}")
    
    if logits.data.shape == (1, 4, 5) and not np.isnan(loss.data):
        print("\n✅ SUCCESS: The Transformer Stack is fully operational.")

if __name__ == "__main__":
    test_language_model()
