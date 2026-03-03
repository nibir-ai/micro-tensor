from microtensor.nn_tensor import MLP
from microtensor.tensor import Tensor

# 1. Initialize Vectorized Network
# 3 inputs, two hidden layers of 4, 1 output layer
model = MLP(3, [4, 4, 1])

# 2. Define a single input as a (1, 3) matrix
x = Tensor([[2.0, 3.0, -1.0]]) 

# 3. Forward Pass
out = model(x)

# 4. Backward Pass
out.backward()

print("Vectorized Network Output (Shape 1,1):")
print(out.data)

print("\nFirst Layer Weight Matrix Shape:")
print(model.layers[0].w.data.shape)

print("\nFirst Layer Weight Gradients (Should be fully populated):")
print(model.layers[0].w.grad)
