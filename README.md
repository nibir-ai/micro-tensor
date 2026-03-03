# micro-tensor

![CI Status](https://github.com/nibir-ai/micro-tensor/actions/workflows/tests.yml/badge.svg)

A lightweight, from-scratch automatic differentiation engine and neural network library built purely in Python.

## Architecture & Theory

`micro-tensor` implements a dynamic computational graph (DAG) that builds mathematical operations on the fly. It utilizes reverse-mode automatic differentiation (via topological sort) to compute exact gradients.

### The Calculus Engine
The core `Value` class wraps scalar floats and tracks their computational history. For every mathematical operation, a closure records the local derivative. During the backward pass, gradients are accumulated sequentially using the multivariate Chain Rule:

$$\frac{\partial L}{\partial x} = \sum_{y} \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$$

### Neural Network Abstraction
Built on top of the Autograd engine is a PyTorch-like API for constructing deep learning architectures:
- `Neuron`: A single linear classifier with a non-linear `tanh` activation function.
- `Layer`: An array of independent Neurons operating in parallel.
- `MLP`: A Multi-Layer Perceptron allowing for deep, sequential representations.

## Usage

Training a multi-layer perceptron using Stochastic Gradient Descent (SGD) and Mean Squared Error (MSE):

```python
from microtensor.nn import MLP
from microtensor.optim import SGD

# Initialize a network (3 inputs, two hidden layers of 4, 1 output)
model = MLP(3, [4, 4, 1])
optimizer = SGD(model.parameters(), lr=0.05)

# Dummy dataset
x = [2.0, 3.0, -1.0]
y_target = 1.0

# Forward Pass
y_pred = model(x)
loss = (y_pred - y_target)**2

# Backward Pass
optimizer.zero_grad()
loss.backward()

# Optimization Step
optimizer.step()
