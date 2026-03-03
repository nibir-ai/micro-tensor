import numpy as np
from microtensor.tensor import Tensor
from microtensor.nn_tensor import LanguageModel, cross_entropy
from microtensor.optim_tensor import Adam

def run_gpt_demo():
    print("=== GPT-Micro: Final Generative Demo ===")
    
    text = "hello world "
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    data = [char_to_idx[ch] for ch in text]
    idx_train = np.array([data[:-1]])
    targets = np.array([data[1:]])
    
    model = LanguageModel(vocab_size, 32, 4, 64, 3, 16)
    optimizer = Adam(model.parameters(), lr=0.01)
    
    print("Training model to memorize 'hello world '...")
    for step in range(101):
        logits = model(idx_train)
        loss = cross_entropy(logits, targets)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 20 == 0: print(f"Step {step} | Loss: {loss.data:.4f}")

    print("\n--- Generation Test ---")
    start_char = "h"
    start_idx = np.array([[char_to_idx[start_char]]])
    
    # Generate 11 more characters
    generated_indices = model.generate(start_idx, 11)
    result = "".join([idx_to_char[i] for i in generated_indices[0]])
    
    print(f"Seed: '{start_char}'")
    print(f"GPT Output: '{result}'")

if __name__ == "__main__":
    run_gpt_demo()
