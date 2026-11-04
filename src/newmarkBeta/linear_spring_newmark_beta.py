import numpy as np
import matplotlib.pyplot as plt

m = 1      # mass
K = 1      # spring constant
b = 0.0    # viscous damping coefficient
dt = 5e-2  # discrete time step
F0 = 10    # driving force
eps = 1e-6 * F0  # error tolerance
maxTime = 10     # total time of simulation
x_eq = F0 / K    # equilibrium position

beta = 0.25  # average acceleration: 0.25, fully implicit Newmark: 0.5
gamma = 0.5  # average acceleration: 0.5, fully implicit Newmark: 1.0

def getForce(t):
    return F0

def mass_spring_damper():
    # Time array
    t = np.arange(0, maxTime + dt, dt)

    # Position, velocity, and acceleration arrays
    x = np.zeros_like(t)
    u = np.zeros_like(t)
    a = np.zeros_like(t)

    # Initial condition: position and velocity are zero
    x[0] = 0
    u[0] = 0
    a[0] = (getForce(t[0]) - K * x[0] - b * u[0]) / m

    # Time-stepping loop
    for k in range(len(t)-1):
        x[k+1], u[k+1], a[k+1] = myInt(t[k+1], x[k], u[k], a[k])

    # Total energy relative to equilibrium
    E = 0.5 * m * u**2 + 0.5 * K * (x - x_eq)**2

    # Plot the result
    plt.figure()
    plt.plot(t, x, 'ro-')
    plt.xlabel('t [second]')
    plt.ylabel('x [meter]')
    plt.title('Mass-Spring-Damper System')

    plt.figure()
    plt.plot(t, E, 'ro-')
    plt.xlabel('t [second]')
    plt.ylabel('E [Joule]')
    plt.title('Mass-Spring-Damper System')
    plt.ylim(0, 100)
    plt.show()

def myInt(t_new, x_old, u_old, a_old):
    # Guess solution for x_new
    x_new = x_old
    err = eps * 100  # initialize to a large value

    # Newton-Raphson method
    while err > eps:
        deltaX = f_Newmark(t_new, x_new, x_old, u_old, a_old) / df_Newmark(x_new, x_old, u_old)
        x_new = x_new - deltaX
        err = abs(f_Newmark(t_new, x_new, x_old, u_old, a_old))

    # Update velocity and acceleration
    u_new, a_new = update_Newmark(x_new, x_old, u_old, a_old)

    return x_new, u_new, a_new

def update_Newmark(x_new, x_old, u_old, a_old):
    # Newmark relations for the velocity and acceleration at the new time step
    a_new = (0.5 / beta) * (2.0 / dt * ((x_new - x_old) / dt - u_old) - (1.0 - 2.0 * beta) * a_old)
    u_new = u_old + (1.0 - gamma) * dt * a_old + gamma * dt * a_new
    return u_new, a_new

def f_Newmark(t, x_new, x_old, u_old, a_old):
    # Residual of the equation of motion at the new time step
    u_new, a_new = update_Newmark(x_new, x_old, u_old, a_old)
    return m * a_new + b * u_new + K * x_new - getForce(t)

def df_Newmark(x_new, x_old, u_old):
    # Jacobian of the residual with respect to x_new
    return m / (beta * dt**2) + b * gamma / (beta * dt) + K

mass_spring_damper()
