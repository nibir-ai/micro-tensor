import numpy as np
from microtensor.tensor import Tensor
from microtensor.nn_tensor import LanguageModel, cross_entropy
from microtensor.optim_tensor import Adam

def test_gpt_overfit():
    print("=== Training GPT-Micro: The 'Hello' Memorization Test ===")
    
    # 1. Simple Tokenizer
    text = "hello"
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Convert text to indices
    # Input: 'hell', Target: 'ello'
    data = [char_to_idx[ch] for ch in text]
    idx = np.array([data[:-1]])     # (1, 4)
    targets = np.array([data[1:]])  # (1, 4)
    
    # 2. Initialize Model (Small for speed)
    model = LanguageModel(
        vocab_size=vocab_size, 
        embed_dim=16, 
        num_heads=2, 
        ff_dim=32, 
        num_layers=2, 
        max_seq_len=10
    )
    optimizer = Adam(model.parameters(), lr=0.01)
    
    print(f"Training on string: '{text}'")
    print("-" * 30)

    # 3. Training Loop
    for step in range(51):
        # Forward
        logits = model(idx)
        loss = cross_entropy(logits, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            # Check prediction
            preds = np.argmax(logits.data, axis=-1)[0]
            pred_text = "".join([idx_to_char[p] for p in preds])
            print(f"Step {step:2d} | Loss: {loss.data:7.4f} | Predicts: '{pred_text}'")

    # 4. Final Validation
    if loss.data < 0.5:
        print("-" * 30)
        print("✅ SUCCESS: The Transformer has memorized the sequence.")
        print("The 4D Autograd and Multi-Head Attention are officially 'Production-Ready'.")
    else:
        print("\n❌ FAILURE: Loss didn't converge. Check gradient accumulation.")

if __name__ == "__main__":
    test_gpt_overfit()
