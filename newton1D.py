def f(x):
  return x**2 - 3*x + 2

def df(x):
  return 2*x - 3

def newton1D():
  # Guess solution
  x = -5

  # Tolerance
  eps = 1e-6

  # Error
  err = eps*10 # initialize to a value larger than eps

  while err > eps:
    deltaX = f(x) / df(x)
    x = x - deltaX
    # err = abs(deltaX)
    err = abs( f(x) )

  return x

x = newton1D()
print("Solution: ", x)

# Solution?
# Stopping criterion?