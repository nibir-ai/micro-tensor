from microtensor.nn import Neuron

# 1. Initialize a Neuron that expects 2 inputs
n = Neuron(2)

# 2. Define 2 inputs
x = [2.0, 3.0]

# 3. Forward pass (Calculate the output)
out = n(x)

# 4. Backward pass (Calculate the gradients for the weights and bias)
out.backward()

# 5. Interrogate the network
print(f"Neuron Output: {out.data}")
print(f"Weight 0 Gradient: {n.w[0].grad}")
print(f"Weight 1 Gradient: {n.w[1].grad}")
print(f"Bias Gradient: {n.b.grad}")
