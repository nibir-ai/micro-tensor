from microtensor.engine import Value

# 1. Forward Pass
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')

e = a * b; e.label = 'e'
d = e + c; d.label = 'd'

# 2. Automated Backward Pass
d.backward()

# 3. Interrogate the gradients
print(f"Gradient of c: {c.grad} (Expected: 1.0)")
print(f"Gradient of a: {a.grad} (Expected: -3.0)")
print(f"Gradient of b: {b.grad} (Expected: 2.0)")
