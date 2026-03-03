import numpy as np
from microtensor.tensor import Tensor
from microtensor.nn_tensor import Conv2d, MaxPool2d, Flatten, Linear, cross_entropy
from microtensor.optim_tensor import Adam
import sys, os

sys.path.append(os.path.dirname(__file__))
from mnist_loader import load_mnist

# 1. Prepare Data
X_train, Y_train, X_test, Y_test = load_mnist()

# 2. Architect (CNN)
class MNISTConvNet:
    def __init__(self):
        self.conv = Conv2d(1, 8, kernel_size=3, padding=1) # (N, 8, 28, 28)
        self.pool = MaxPool2d(2)                          # (N, 8, 14, 14)
        self.flat = Flatten()
        self.fc = Linear(8 * 14 * 14, 10)
    def parameters(self):
        return self.conv.parameters() + self.fc.parameters()
    def __call__(self, x):
        x = self.conv(x).tanh()
        x = self.pool(x)
        return self.fc(self.flat(x))

model = MNISTConvNet()
optimizer = Adam(model.parameters(), lr=0.005)

# 3. Training
batch_size = 32
print("\nStarting Advanced MNIST Training...")

for i in range(101):
    indices = np.random.randint(0, len(X_train), batch_size)
    xb, yb = Tensor(X_train[indices]), Y_train[indices]
    
    # Forward + Loss
    logits = model(xb)
    loss = cross_entropy(logits, yb)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 10 == 0:
        acc = np.mean(np.argmax(logits.data, axis=1) == yb)
        print(f"Step {i:3d} | Loss: {loss.data:.4f} | Acc: {acc*100:3.0f}%")

print("\nPhase 2 Complete. Your engine is now mathematically elite.")
