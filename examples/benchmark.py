import numpy as np
from microtensor.nn_tensor import LanguageModel, cross_entropy
from microtensor.profiler import Profiler

# 1. Configuration for Stress Test
VOCAB_SIZE = 64
EMBED_DIM = 256
NUM_HEADS = 8
FF_DIM = 512
NUM_LAYERS = 4
MAX_SEQ_LEN = 32
BATCH_SIZE = 8

# 2. Setup
model = LanguageModel(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS, MAX_SEQ_LEN)
idx = np.random.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))
targets = np.random.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))

print(f"🚀 Benchmarking micro-tensor...")
print(f"Layers: {NUM_LAYERS} | Embed: {EMBED_DIM} | Heads: {NUM_HEADS}")

# 3. Warm-up and Loop
for i in range(5):
    logits = model(idx)
    loss = cross_entropy(logits, targets)
    loss.backward()
    print(f"Iteration {i+1}/5 complete.")

# 4. Final Audit
Profiler.report()
