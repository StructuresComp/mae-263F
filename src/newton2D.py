import numpy as np

def f(x):
  return np.array([x[0]**3 + x[1], -x[0] + x[1]**3 + 1])

def J(x):
  return np.array([[3.0*x[0]**2, 1], [-1, 3*x[1]**2]])

def newton2D():
  # Guess solution
  x = np.array([0.5, 0.5])

  # Tolerance
  eps = 1e-6

  # Error
  err = eps*10 # initialize to a value larger than eps

  while err > eps:
    deltaX = np.linalg.solve(J(x), f(x))
    x = x - deltaX
    err = np.linalg.norm(f(x))

  return x

x = newton2D()
print('Solution is x1=%.2f, x2=%.2f' % (x[0], x[1]))