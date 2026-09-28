import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 3  # number of DOF

m = np.array([10, 0.01, 0.001])  # mass
K = np.array([1, 5, 0.1])  # spring constant
b = np.array([10, 50, 100])  # viscous damping coefficient
F0 = np.array([0.1, 10, 1])  # driving amplitude
omega = np.array([0.1, 10, 2])  # driving frequency
x0 = np.array([1, 2, 3])  # initial positions

maxTime = 10  # total time of simulation
dt = 1e-2  # discrete time step
eps = 1e-6 * np.sum(F0)  # error tolerance

# Time array
t = np.arange(0, maxTime + dt, dt)

# Position and velocity arrays
x = np.zeros((N, len(t)))  # position
u = np.zeros((N, len(t)))  # velocity

# Initial condition: position and velocity are zero
x[:, 0] = x0
u[:, 0] = np.zeros(3)

# Loop variables
f = np.zeros(N)
J = np.zeros((N, N))

for k in range(len(t) - 1):  # march over time steps
    x_old = x[:, k]
    u_old = u[:, k]
    t_new = t[k + 1]

    # Guess solution
    x_new = x_old.copy() # x_new = x_old will NOT work!
    err = eps * 100  # initialize to a large value

    while err > eps:
        # Compute f
        for i in range(N):
            f[i] = m[i] / dt * ((x_new[i] - x_old[i]) / dt - u_old[i]) - \
              F0[i] * np.sin(omega[i] * t_new)
            if i == 0:
                f[i] += (K[i] + K[i + 1]) * x_new[i] - K[i + 1] * x_new[i + 1] + \
                        (b[i] + b[i + 1]) * (x_new[i] - x_old[i]) / dt - b[i + 1] * (x_new[i + 1] - x_old[i + 1]) / dt
            elif i == N - 1:
                f[i] += K[i] * x_new[i] - K[i] * x_new[i - 1] + b[i] * (x_new[i] - x_old[i]) / dt - \
                        b[i] * (x_new[i - 1] - x_old[i - 1]) / dt
            else:
                f[i] += (K[i] + K[i + 1]) * x_new[i] - K[i] * x_new[i - 1] - K[i + 1] * x_new[i + 1] + \
                        (b[i] + b[i + 1]) * (x_new[i] - x_old[i]) / dt - b[i] * (x_new[i - 1] - x_old[i - 1]) / dt - \
                        b[i + 1] * (x_new[i + 1] - x_old[i + 1]) / dt

        # Compute J (Jacobian)
        for i in range(N):
            for j in range(N):
                ki = K[i]
                kip1 = K[i + 1] if i < N - 1 else 0  # k_{i+1}
                bi = b[i]
                bip1 = b[i + 1] if i < N - 1 else 0  # b_{i+1}

                if i == j:
                    J[i, j] = m[i] / dt**2 + (ki + kip1) + (bi + bip1) / dt
                elif j == i - 1:
                    J[i, j] = -ki - bi / dt
                elif j == i + 1:
                    J[i, j] = -kip1 - bip1 / dt
                else:
                    J[i, j] = 0

        # Compute delta x and update x_new
        deltaX = np.linalg.solve(J, f)
        x_new -= deltaX
        err = np.linalg.norm(f)

    # Update velocity
    u_new = (x_new - x_old) / dt

    # Store solution
    x[:, k + 1] = x_new
    u[:, k + 1] = u_new

    # Plot intermediate results
    fig = plt.figure(1)
    plt.clf()
    plt.plot([0] + list(x_new), np.zeros(N + 1), 'ro-')
    plt.xlim([0, np.max(x0) * 1.5])
    plt.ylim([-1, 1])
    plt.title(f't = {t_new:.2f} sec')
    plt.axis('off')
    plt.draw()
    plt.pause(0.1)

# Final plot
colpos = np.array([[247, 148, 30], [0, 166, 81], [237, 28, 36]]) / 255  # colors

plt.figure(2)
plt.plot(t, x[0, :], color=colpos[0], linewidth=2, label='i=1')
plt.plot(t, x[1, :], color=colpos[1], linewidth=2, label='i=2')
plt.plot(t, x[2, :], color=colpos[2], linewidth=2, label='i=3')
plt.xlabel('Time, t [sec]')
plt.ylabel('Location, x_i [m]')
plt.legend(loc='best')
plt.grid(True)

# Save the plot as PDF
plt.savefig('Fig_multiMassSpringDamper.pdf')

# Show the plot
plt.show()