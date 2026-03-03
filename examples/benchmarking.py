import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from microtensor.nn_tensor import MLP
from microtensor.tensor import Tensor
from microtensor.optim_tensor import SGD, Adam

# Setup data
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
y = y * 2 - 1
X_tensor = Tensor(X)
Y_tensor = Tensor(y.reshape(-1, 1))

def train(opt_class):
    model = MLP(2, [16, 16, 1])
    if opt_class == SGD:
        optimizer = opt_class(model.parameters(), lr=0.1)
    else:
        optimizer = opt_class(model.parameters(), lr=0.01)
    
    losses = []
    for k in range(100):
        y_pred = model(X_tensor)
        reg_loss = Tensor(0.0)
        for p in model.parameters():
            reg_loss = reg_loss + (p * p).sum()
        data_loss = (Tensor(1.0) - Y_tensor * y_pred)
        data_loss.data[data_loss.data < 0] = 0.0
        total_loss = data_loss.sum() + reg_loss * 0.001
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.data)
    return losses

print("Benchmarking SGD vs ADAM...")
sgd_losses = train(SGD)
adam_losses = train(Adam)

plt.plot(sgd_losses, label='SGD (lr=0.1)')
plt.plot(adam_losses, label='Adam (lr=0.01)')
plt.title('Convergence Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('optimizer_comparison.png')
print("Comparison saved to optimizer_comparison.png")
