from microtensor.tensor import Tensor
import numpy as np

# A is shape (2, 3)
A = Tensor([[1.0, 2.0, 3.0], 
            [4.0, 5.0, 6.0]])

# B is shape (3, 1)
B = Tensor([[7.0], 
            [8.0], 
            [9.0]])

# Forward Pass: C will be shape (2, 1)
C = A @ B 

# Backward Pass
C.backward()

print("Matrix C Output Data (Should be shape 2,1):")
print(C.data)
print("\nGradient of A (Should be shape 2,3):")
print(A.grad)
print("\nGradient of B (Should be shape 3,1):")
print(B.grad)
