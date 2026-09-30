import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 3 # Number of DOFs

m = np.array([10.0, 0.01, 0.001])  # Masses
K = np.array([1, 5, 0.1]) # Stiffnesses
b = np.array([10, 50, 100]) # Damping coefficients
F0 = np.array([0.1, 10, 1]) # Forcing function amplitudes
omega = np.array([0.1, 10, 2]) # Forcing function frequencies

# Time stepping parameters
maxTime = 10.0 # total time of simulation
dt = 0.01 # time step size
eps = 1.0e-6 * np.sum( np.abs(F0)) # tolerance
t = np.arange(0, maxTime + dt, dt) # time array

# Position and velocity arrays
x = np.zeros((N, len(t))) # position array
u = np.zeros((N, len(t))) # dot(x) : velocity array

# Simulation

# Initial conditions
x[:, 0] = np.array([1.0, 2.0, 3.0]) # initial positions
u[:, 0] = np.array([0.0, 0.0, 0.0]) # initial velocities

for k in range(len(t)-1):
    # "Old" position and velocity
    x_old = x[:, k]
    u_old = u[:, k]
    t_new = t[k+1]

    # Guess
    x_new = x_old.copy()
    err = 10.0 * eps

    f = np.zeros(N) # "Force" -- residual vector
    J = np.zeros((N, N))  # Jacobian matrix

    while err > eps:
        # Compute residual vector ("f" vector)
        for i in range(N):
            if i == 0:
                f[i] = m[i] / dt * ( (x_new[i] - x_old[i]) /dt - u_old[i]) + \
                    (K[i] + K[i+1]) * x_new[i] - K[i+1] * x_new[i+1] + \
                    (b[i] + b[i+1]) * (x_new[i] - x_old[i])/dt - b[i+1] * (x_new[i+1]-x_old[i+1])/dt + \
                    F0[i] * np.sin(omega[i] * t_new)
            elif i == N-1:
                f[i] = m[i] / dt * ( (x_new[i] - x_old[i]) /dt - u_old[i]) + \
                    (K[i]) * x_new[i] - K[i] * x_new[i-1] + \
                    (b[i]) * (x_new[i] - x_old[i])/dt - b[i] * (x_new[i-1]-x_old[i-1])/dt + \
                    F0[i] * np.sin(omega[i] * t_new)
            else:
                f[i] = m[i] / dt * ( (x_new[i] - x_old[i]) /dt - u_old[i]) + \
                    (K[i] + K[i+1]) * x_new[i] - K[i] * x_new[i-1] - K[i+1] * x_new[i+1] + \
                    (b[i] + b[i+1]) * (x_new[i] - x_old[i])/dt - b[i] * (x_new[i-1]-x_old[i-1])/dt - b[i+1] * (x_new[i+1]-x_old[i+1])/dt + \
                    F0[i] * np.sin(omega[i] * t_new)


        # Jacobian matrix
        for i in range(N):
            for j in range(N):
                ki = K[i]
                kip1 = K[i+1] if i < N-1 else 0.0
                bi = b[i]
                bip1 = b[i+1] if i < N-1 else 0.0

                if i == j:
                    J[i, j] = m[i] / dt**2 + ki + kip1 + (bi + bip1) / dt
                elif j == i-1:
                    J[i, j] = -ki - bi / dt
                elif j == i+1:
                    J[i, j] = -kip1 - bip1 / dt

        # Solve for the update
        dx = np.linalg.solve(J, f) # J \ f

        # Update the guess
        x_new -= dx

        # Compute error
        err = np.linalg.norm(dx)

    # Update velocity
    u_new = (x_new - x_old) / dt

    # Store solution
    x[:, k+1] = x_new
    u[:, k+1] = u_new

    # Plotting
    fig = plt.figure(1)
    plt.clf()
    plt.plot( [0] + list(x_new), np.zeros(N+1), 'ro', markersize=10)
    plt.ylim([-1, 1])
    plt.xlim([0, 5])
    plt.title('Time = %.2f s' % t_new)
    plt.axis('off')
    plt.draw()
    plt.pause(0.1)


# Final plot