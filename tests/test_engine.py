from microtensor.engine import Value

def test_forward_pass():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = a * b + c
    assert d.data == 4.0

def test_backward_pass():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = a * b + c
    d.backward()
    
    # Expected gradients for d = a * b + c
    assert a.grad == -3.0
    assert b.grad == 2.0
    assert c.grad == 1.0

def test_complex_math():
    a = Value(2.0)
    b = Value(3.0)
    
    # c = a^2 + b/a = 4.0 + 1.5 = 5.5
    c = a ** 2 + b / a 
    assert c.data == 5.5
    
    c.backward()
    
    # dc/da = 2a - b/(a^2) = 4 - 3/4 = 3.25
    # dc/db = 1/a = 0.5
    assert a.grad == 3.25
    assert b.grad == 0.5
