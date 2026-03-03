from microtensor.tensor import Tensor

# Predictions (Shape 2,1)
y_pred = Tensor([[0.8], 
                 [-0.5]])

# Targets (Shape 2,1)
y_true = Tensor([[1.0], 
                 [-1.0]])

# Calculate Mean Squared Error using our new operations
# Loss = sum((y_pred - y_true) ** 2)
diff = y_pred - y_true
sq = diff ** 2
loss = sq.sum()

# Trigger backprop
loss.backward()

print(f"Total Loss:\n{loss.data}")
print(f"\nGradient of y_pred (How much should predictions change?):\n{y_pred.grad}")
