
import numpy as np

# Training data
x = np.array([-1., 0., 1.])
y = 2 * x**2 + x

# Initial parameters
a = 0.0
b = 0.0
c = 0.0

learning_rate = 0.1

for i in range(1000):
    # Forward pass
    y_hat = a * x**2 + b * x + c

    # Prediction error
    error = y_hat - y

    # Gradients (mean squared error)
    da = 2 * np.mean(error * x**2)
    db = 2 * np.mean(error * x)
    dc = 2 * np.mean(error)

    # Gradient descent update
    a -= learning_rate * da
    b -= learning_rate * db
    c -= learning_rate * dc

print(a, b, c)
