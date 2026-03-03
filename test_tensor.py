from microtensor.nn_tensor import MLP
from microtensor.tensor import Tensor
from microtensor.optim_tensor import SGD

# 1. Initialize Network and Optimizer
model = MLP(3, [4, 4, 1])
optimizer = SGD(model.parameters(), lr=0.05)

# 2. The Batch Input Matrix (4 examples, 3 features -> Shape: 4, 3)
X = Tensor([
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
])

# 3. The Target Matrix (4 examples, 1 target -> Shape: 4, 1)
Y = Tensor([
  [1.0], 
  [-1.0], 
  [-1.0], 
  [1.0]
])

print("Starting Vectorized Batch Training...")
epochs = 20

for k in range(epochs):
    # --- Forward Pass (ALL 4 EXAMPLES AT ONCE) ---
    y_pred = model(X)
    
    # --- Calculate Loss (Vectorized MSE) ---
    diff = y_pred - Y
    sq = diff ** 2
    loss = sq.sum()
    
    # --- Backward Pass ---
    optimizer.zero_grad()
    loss.backward()
    
    # --- Optimization Step ---
    optimizer.step()
        
    print(f"Epoch {k:2d} | Loss: {loss.data:.4f}")

# Final predictions
print("\nFinal Predictions:")
print(model(X).data)
