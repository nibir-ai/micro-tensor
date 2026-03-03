import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from microtensor.nn_tensor import MLP
from microtensor.tensor import Tensor
from microtensor.optim_tensor import Adam  # Changed from SGD to Adam

# 1. Generate Dataset
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
y = y * 2 - 1  # Convert to -1 and 1
X_tensor = Tensor(X)
Y_tensor = Tensor(y.reshape(-1, 1))

# 2. Initialize Model and Adam Optimizer
# We use a slightly higher learning rate for Adam (0.01 is common)
model = MLP(2, [16, 16, 1])
optimizer = Adam(model.parameters(), lr=0.01)

# 3. Training Loop
print("Training on Moons Dataset using ADAM Optimizer...")
for k in range(100):
    # Forward pass
    y_pred = model(X_tensor)
    
    # Loss Calculation (Hinge Loss + L2 Regularization)
    reg_loss = Tensor(0.0)
    for p in model.parameters():
        reg_loss = reg_loss + (p * p).sum()
        
    data_loss = (Tensor(1.0) - Y_tensor * y_pred)
    data_loss.data[data_loss.data < 0] = 0.0 # Hinge loss max(0, 1-y*y_hat)
    
    total_loss = data_loss.sum() + reg_loss * 0.001
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    
    # Update
    optimizer.step()
    
    if (k + 1) % 10 == 0:
        print(f"Epoch {k+1:3d} | Loss: {total_loss.data:.4f}")

# 4. Plot Decision Boundary
print("\nPlotting Decision Boundary...")
h = 0.25
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Xmesh = np.c_[xx.ravel(), yy.ravel()]
inputs = Tensor(Xmesh)
scores = model(inputs)
Z = np.array([s > 0 for s in scores.data]).reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.savefig('moon_boundary_adam.png')
print("Successfully saved to moon_boundary_adam.png!")
