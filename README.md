# micro-tensor 🚀

![Tests](https://github.com/nibir-ai/micro-tensor/actions/workflows/tests.yml/badge.svg)

A research-grade, first-principles deep learning framework built entirely in NumPy. Designed for educational transparency and architectural rigor.

## ✨ Features
- **Vectorized Autograd**: Support for N-dimensional tensors with automatic gradient broadcasting.
- **CNN Suite**: High-performance `Conv2d` (via `as_strided`), `MaxPool2d`, and `Flatten`.
- **Transformer Engine**: Multi-Head Self-Attention, Positional Encoding, and Layer Normalization.
- **Optimizers**: Integrated `Adam` and `SGD` implementations.
- **Stability**: Log-Sum-Exp stable `CrossEntropyLoss`.

## 📊 Benchmarks
- **MNIST**: 94%+ batch accuracy within 100 steps.
- **GPT-Micro**: Successfully performs autoregressive character-level generation.

## 🛠 Usage
```bash
# Clone and install
git clone [https://github.com/nibir-ai/micro-tensor](https://github.com/nibir-ai/micro-tensor)
cd micro-tensor
pip install -e .

# Run the Transformer demo
python examples/gpt_inference.py
