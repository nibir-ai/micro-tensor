from microtensor.nn import MLP
from microtensor.optim import SGD

# 1. Initialize Network and Optimizer
n = MLP(3, [4, 4, 1])
optimizer = SGD(n.parameters(), lr=0.05)

# 2. Dataset
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

# 3. The Professional Training Loop
epochs = 20

print("Starting training...")
for k in range(epochs):
    
    # Forward Pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    # Backward Pass
    optimizer.zero_grad()
    loss.backward()
    
    # Optimization Step
    optimizer.step()
        
    print(f"Epoch {k:2d} | Loss: {loss.data:.4f}")
