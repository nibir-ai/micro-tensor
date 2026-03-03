from microtensor.nn import MLP

# 1. Initialize Network
n = MLP(3, [4, 4, 1])

# 2. Define a simple dataset
# 4 examples, each with 3 features
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
# The target answers we want the network to output for each example
ys = [1.0, -1.0, -1.0, 1.0]

# 3. The Training Loop (Gradient Descent)
epochs = 20
learning_rate = 0.05

print("Starting training...")
for k in range(epochs):
    
    # --- Forward Pass ---
    # Get predictions for all 4 examples
    ypred = [n(x) for x in xs]
    
    # Calculate Mean Squared Error loss
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    # --- Backward Pass ---
    # CRITICAL: Always zero your gradients before the backward pass, 
    # otherwise they will accumulate from the previous epoch!
    n.zero_grad()
    loss.backward()
    
    # --- Optimization (Weight Update) ---
    for p in n.parameters():
        p.data += -learning_rate * p.grad
        
    print(f"Epoch {k:2d} | Loss: {loss.data:.4f}")

# 4. View Final Predictions
print("\nFinal Predictions after training:")
for i, pred in enumerate(ypred):
    print(f"Target: {ys[i]:>4} | Prediction: {pred.data:.4f}")
