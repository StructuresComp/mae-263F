import numpy as np
import matplotlib.pyplot as plt

# Global variables
m = 1      # mass
K = 1      # spring constant
b = 10     # viscous damping coefficient
F0 = 0.1   # driving amplitude
omega = 1  # driving frequency
dt = 1e-1  # discrete time step
eps = 1e-6 * F0  # error tolerance
maxTime = 50     # total time of simulation

def mass_spring_damper():
    # Time array
    t = np.arange(0, maxTime + dt, dt)

    # Position and velocity arrays
    x = np.zeros_like(t)
    u = np.zeros_like(t)

    # Initial condition: position and velocity are zero
    x[0] = 0
    u[0] = 0

    # Time-stepping loop
    for k in range(len(t)-1):
        x_old = x[k]
        u_old = u[k]
        t_new = t[k+1]

        # Guess solution for x_new
        x_new = x_old
        err = eps * 100  # initialize to a large value

        # Iterative solution
        while err > eps:
            deltaX = f(x_new, x_old, u_old, t_new) / df(x_new, x_old, u_old, t_new)
            x_new = x_new - deltaX
            err = abs(f(x_new, x_old, u_old, t_new))

        # Update velocity
        u_new = (x_new - x_old) / dt

        # Store solution
        x[k+1] = x_new
        u[k+1] = u_new

    # Plot the result
    plt.plot(t, x, 'ro-')
    plt.xlabel('t [second]')
    plt.ylabel('x [meter]')
    plt.title('Mass-Spring-Damper System')
    plt.show()

# Define f function (residual)
def f(x_new, x_old, u_old, t_new):
    return m/dt * ((x_new - x_old)/dt - u_old) + K * x_new + b * (x_new - x_old) / dt - F0 * np.sin(omega * t_new)

# Define df function (Jacobian of f)
def df(x_new, x_old, u_old, t_new):
    return m/dt**2 + K + b/dt

mass_spring_damper()