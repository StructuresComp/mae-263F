"""
Neural ODE demo: learn the spring stiffness k and the damping coefficient b of a forced
mass-spring-damper from measurements of its displacement,

    m x'' + b x' + k x = F0 sin(omega t).

1. Synthetic data: integrate the equation of motion with implicit Euler and Newton's
   method (as in ../massSpringDamper.py) using the true k and b with time step dtSim,
   and sample the displacement every dtData seconds.
2. Model: the same equation as a first-order system in y = [x, v], with k and b as
   trainable PyTorch parameters. The mass m, the forcing and the initial state are known.
3. Training: integrate the model over the measurement times with the solver chosen by
   SOLVER at the top of main() and backpropagate through it; Adam minimizes the mean
   squared error between the predicted and the measured displacement.
     'odeint'         - torchdiffeq's adaptive dopri5, i.e. (nearly) the exact ODE solution.
     'backward_euler' - implicit Euler with time step dtData, the same scheme as the data
                        generator. With dtSim = dtData it reproduces the data exactly.

Requires PyTorch and torchdiffeq (pip install torch torchdiffeq).

Run from anywhere, e.g.:  python src/neuralODE/neuralODE_massSpringDamper.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchdiffeq import odeint

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# Synthetic data: implicit Euler with Newton's method
# ---------------------------------------------------------------------------

def simulate_implicit_euler(m, k, b, F0, omega, x0, v0, dt, totalTime, xtol=1e-12, maxIter=10):
    """
    Integrate m x'' + b x' + k x = F0 sin(omega t) with implicit Euler. At every step,
    Newton's method solves f(x_new) = 0, where f is the equation of motion at t_new.
    Returns the time, position and velocity arrays.
    """
    nSteps = round(totalTime / dt)
    t = np.arange(nSteps + 1) * dt
    x = np.zeros(nSteps + 1)
    v = np.zeros(nSteps + 1)
    x[0], v[0] = x0, v0

    for i in range(nSteps):
        x_new = x[i]  # guess for Newton's method
        for _ in range(maxIter):
            f = (m / dt * ((x_new - x[i]) / dt - v[i]) + b * (x_new - x[i]) / dt
                 + k * x_new - F0 * np.sin(omega * t[i + 1]))
            df = m / dt**2 + b / dt + k
            dx = f / df
            x_new = x_new - dx
            if abs(dx) < xtol:
                break

        x[i + 1] = x_new
        v[i + 1] = (x_new - x[i]) / dt

    return t, x, v


# ---------------------------------------------------------------------------
# Model: right-hand side of the ODE with trainable k and b
# ---------------------------------------------------------------------------

class MassSpringDamper(nn.Module):
    """
    dy/dt for y = [x, v]: dx/dt = v, dv/dt = (F0 sin(omega t) - b v - k x) / m.
    k and b are stored as logarithms so that they stay positive during training.
    """

    def __init__(self, m, k_guess, b_guess, F0, omega):
        super().__init__()
        self.m = m
        self.F0 = F0
        self.omega = omega
        self.log_k = nn.Parameter(torch.log(torch.tensor(k_guess)))
        self.log_b = nn.Parameter(torch.log(torch.tensor(b_guess)))

    @property
    def k(self):
        return torch.exp(self.log_k)

    @property
    def b(self):
        return torch.exp(self.log_b)

    def forward(self, t, y):
        x, v = y[0], y[1]
        a = (self.F0 * torch.sin(self.omega * t) - self.b * v - self.k * x) / self.m
        return torch.stack([v, a])


# ---------------------------------------------------------------------------
# Predicted displacement: odeint or backward Euler
# ---------------------------------------------------------------------------

def backward_euler_prediction(model, y0, t):
    """
    Displacement at the times t from implicit Euler with time step t[i+1] - t[i], the
    same scheme as simulate_implicit_euler. The equation of motion is linear in x_new,
    so every step is solved exactly in closed form (Newton converges in one iteration).
    """
    x, v = y0[0], y0[1]
    positions = [x]

    for i in range(t.numel() - 1):
        h = t[i + 1] - t[i]
        force = model.F0 * torch.sin(model.omega * t[i + 1])
        x_new = ((model.m / h**2 * (x + h * v) + model.b / h * x + force)
                 / (model.m / h**2 + model.b / h + model.k))
        v = (x_new - x) / h
        x = x_new
        positions.append(x)

    return torch.stack(positions)


def predict_displacement(model, y0, t, solver):
    """Displacement at the times t from the solver 'odeint' or 'backward_euler'."""
    if solver == 'odeint':
        y = odeint(model, y0, t, method='dopri5', rtol=1e-4, atol=1e-6)
        return y[:, 0]
    if solver == 'backward_euler':
        return backward_euler_prediction(model, y0, t)
    raise ValueError(f"Unknown solver '{solver}': use 'odeint' or 'backward_euler'.")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, y0, t, x_measured, solver, nEpochs, learningRate, printEvery):
    """
    Fit k and b with Adam. The loss is the mean squared error between the predicted
    and the measured displacement, normalized by the largest measured displacement.
    Returns the loss, k and b at every epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    xScale = x_measured.abs().max()

    lossHistory = np.zeros(nEpochs)
    kHistory = np.zeros(nEpochs)
    bHistory = np.zeros(nEpochs)

    for epoch in range(nEpochs):
        optimizer.zero_grad()
        x_predicted = predict_displacement(model, y0, t, solver)
        loss = torch.mean(((x_predicted - x_measured) / xScale)**2)
        loss.backward()
        optimizer.step()

        lossHistory[epoch] = loss.item()
        kHistory[epoch] = model.k.item()
        bHistory[epoch] = model.b.item()

        if epoch % printEvery == 0 or epoch == nEpochs - 1:
            print(f'Epoch {epoch:4d} | loss = {loss.item():.4e} | '
                  f'k = {model.k.item():.4f} | b = {model.b.item():.4f}')

    return lossHistory, kHistory, bHistory


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_synthetic_data(t, x, v):
    """Position and velocity of the synthetic data against time, displayed before training starts."""
    _, ax = plt.subplots(2, 1, sharex=True, num=3)
    ax[0].plot(t, x, 'r-')
    ax[0].set_ylabel('Position, x [m]')
    ax[0].set_title('Synthetic Data')
    ax[1].plot(t, v, 'r-')
    ax[1].set_ylabel('Velocity, v [m/s]')
    ax[1].set_xlabel('Time, t [s]')
    for a in ax:
        a.grid(True)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.5)  # Display the figure without blocking; a shorter pause leaves the window blank during training


def plot_fit(t_sim, x_sim, t_data, x_data, x_initial, x_learned, solver):
    """Measured displacement, the true response, and the model before and after training."""
    plt.figure(1)
    plt.plot(t_sim, x_sim, 'k-', label='True response')
    plt.plot(t_data, x_data, 'r.', label='Measurements')
    plt.plot(t_data, x_initial, 'g--', label='Initial guess')
    plt.plot(t_data, x_learned, 'b-', label='Learned model')
    plt.xlabel('Time, t [s]')
    plt.ylabel('Displacement, x [m]')
    plt.title(f'Mass-Spring-Damper: Neural ODE Fit ({solver})')
    plt.legend()
    plt.grid(True)


def plot_training(lossHistory, kHistory, bHistory, k_true, b_true):
    """Loss, k and b against the epoch, with the true values as dashed lines."""
    _, ax = plt.subplots(3, 1, sharex=True, num=2)
    ax[0].semilogy(lossHistory, 'b-')
    ax[0].set_ylabel('Loss')
    ax[1].plot(kHistory, 'b-', label='Learned')
    ax[1].axhline(k_true, color='r', linestyle='--', label='True')
    ax[1].set_ylabel('k [N/m]')
    ax[1].legend()
    ax[2].plot(bHistory, 'b-')
    ax[2].axhline(b_true, color='r', linestyle='--')
    ax[2].set_ylabel('b [N s/m]')
    ax[2].set_xlabel('Epoch')
    for a in ax:
        a.grid(True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Solver used in training: 'odeint' or 'backward_euler'
    SOLVER = 'backward_euler'

    # Physical parameters: m and the forcing are known; k and b are to be learned
    m = 1.0             # mass [kg]
    k_true = 1.0        # spring stiffness [N/m]
    b_true = 1.0        # viscous damping coefficient [N s/m]
    F0 = 0.1            # driving amplitude [N]
    omega = 1.0         # driving frequency [rad/s]
    x0 = 0.0            # initial displacement [m]
    v0 = 0.0            # initial velocity [m/s]

    # Synthetic data
    totalTime = 500.0    # length of the record [s]
    dtSim = 0.1         # implicit Euler time step [s]; with dtSim = dtData, backward_euler reproduces the data exactly
    dtData = 0.1        # time between measurements [s]; also the backward_euler time step

    # Training
    k_guess = 2.0       # initial guess for k [N/m]
    b_guess = 5.0       # initial guess for b [N s/m]
    nEpochs = 200       # number of Adam iterations
    learningRate = 0.03
    printEvery = 100

    # 1. Synthetic data: simulate with the true k and b, sample every dtData
    t_sim, x_sim, v_sim = simulate_implicit_euler(m, k_true, b_true, F0, omega, x0, v0,
                                                  dtSim, totalTime)
    stride = round(dtData / dtSim)
    t_data = t_sim[::stride]
    x_data = x_sim[::stride]
    plot_synthetic_data(t_sim, x_sim, v_sim)

    t = torch.tensor(t_data)
    x_measured = torch.tensor(x_data)
    y0 = torch.tensor([x0, v0])

    # 2. Model with the initial guesses
    model = MassSpringDamper(m, k_guess, b_guess, F0, omega)
    with torch.no_grad():
        x_initial = predict_displacement(model, y0, t, SOLVER).numpy()

    # 3. Training
    print(f'Solver: {SOLVER}')
    lossHistory, kHistory, bHistory = train(model, y0, t, x_measured, SOLVER,
                                            nEpochs, learningRate, printEvery)

    print('\n===== Learned parameters =====')
    print(f'k: true = {k_true:.4f}, learned = {model.k.item():.4f}')
    print(f'b: true = {b_true:.4f}, learned = {model.b.item():.4f}')

    with torch.no_grad():
        x_learned = predict_displacement(model, y0, t, SOLVER).numpy()

    plot_fit(t_sim, x_sim, t_data, x_data, x_initial, x_learned, SOLVER)
    plot_training(lossHistory, kHistory, bHistory, k_true, b_true)
    plt.show()


if __name__ == '__main__':
    main()
