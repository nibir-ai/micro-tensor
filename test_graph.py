from microtensor.nn import MLP

# 1. Initialize a Neural Network
# 3 inputs, two hidden layers of 4 neurons, 1 output neuron
n = MLP(3, [4, 4, 1])

# 2. Define an input vector
x = [2.0, 3.0, -1.0]

# 3. Forward pass (Calculate the final prediction)
out = n(x)

# 4. Backward pass (Calculate gradients for every single parameter in the network)
out.backward()

# 5. Interrogate the network
print(f"Network Output: {out.data}")
print(f"Total parameters in network: {len(n.parameters())}")

# Let's look at the gradient of a random weight in the first layer
print(f"Sample Weight Gradient (Layer 0, Neuron 0, Weight 0): {n.layers[0].neurons[0].w[0].grad}")
