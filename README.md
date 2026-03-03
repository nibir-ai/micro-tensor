# micro-tensor 

![Tests](https://github.com/nibir-ai/micro-tensor/actions/workflows/tests.yml/badge.svg)

A high-performance, vectorized autograd engine and CNN library built from absolute first principles.

## 🚀 Phase 2 Achievements: Spatial Intelligence
- **4D Tensor Engine**: Optimized for Batch, Channel, Height, and Width $(N, C, H, W)$.
- **Vectorized Convolutions**: Implemented using `np.lib.stride_tricks.as_strided` for O(1) patch extraction.
- **Full CNN Toolkit**: Includes `Conv2d`, `MaxPool2d`, `Flatten`, and `Linear` modules.
- **MNIST Verified**: Achieved 75%+ batch accuracy on MNIST digits within 100 training steps on a CPU.

## 📊 Results
### MNIST Training (from scratch)
```text
Step   0 | Loss: 65.71 | Batch Acc:  0%
Step  50 | Loss: 10.77 | Batch Acc: 69%
Step 100 | Loss:  7.51 | Batch Acc: 75%
