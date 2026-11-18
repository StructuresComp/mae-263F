# Mass–Spring–Damper Parameter Identification with a Neural ODE

A mass $m$ is attached to a spring of stiffness $k$ and a viscous damper with damping coefficient $b$. A sinusoidal force of amplitude $F_0$ and frequency $\omega$ drives it. Its displacement $x(t)$ obeys

$$
m\ddot{x} + b\dot{x} + kx = F_0\sin(\omega t).
$$

Our goal is to identify the spring stiffness $k$ and the damping coefficient $b$ from measurements of the displacement $x(t)$. The mass $m$, the forcing ($F_0$ and $\omega$), and the initial displacement and velocity are known. The code has no neural network. It uses the neural ODE approach of backpropagating through an ODE solver to fit the two physical parameters.

## First-order ODEs

With $v=\dot{x}$, the second-order equation becomes

$$
\begin{aligned}
\dot{x} &= v,\\
\dot{v} &= \frac{F_0\sin(\omega t)-bv-kx}{m}.
\end{aligned}
$$

## Code structure

`neuralODE_massSpringDamper.py` works in five steps:

1. **Synthetic data:** Simulate the equation of motion with the true $k$ and $b$ using backward (implicit) Euler and Newton's method, as in `../massSpringDamper.py`, with time step `dtSim`. Sample the displacement every `dtData` seconds. A figure of the position (top) and the velocity (bottom) of the data against time is displayed before training starts.
2. **ODE model:** Write the first-order equations as a PyTorch `nn.Module` with trainable $k$ and $b$, starting from `k_guess` and `b_guess`. They are stored as $\log k$ and $\log b$ so that they stay positive.
3. **Prediction:** Integrate the model over the measurement times with the solver chosen by `SOLVER` (see below).
4. **Training:** Adam minimizes the mean squared error between the predicted and the measured displacement, normalized by the largest measured displacement. The gradients come from backpropagation through the solver.
5. **Results:** Print the learned $k$ and $b$. Plot the measurements, the true response, and the model before and after training. A second figure shows the loss, $k$ and $b$ against the epoch.

## Choosing the solver

Set `SOLVER` at the top of `main()` to one of two values:

- `'odeint'`: torchdiffeq's adaptive Dormand–Prince solver (`dopri5`, `rtol=1e-4`, `atol=1e-6`). It gives a close approximation of the exact solution of the ODE.
- `'backward_euler'`: backward Euler with time step `dtData`, the same scheme as the data generator. The equation is linear in $x$, so each step is solved in closed form.

Comparing the two separates parameter-identification difficulties from numerical-discretization differences:

- **`dtSim = dtData` (default):** the data are exactly what `backward_euler` computes, so `backward_euler` recovers $k$ and $b$ exactly. Any error from `odeint` comes from the discretization error of the data generator: its numerical damping is absorbed into the learned $k$ and $b$.
- **Small `dtSim` (e.g. `1e-4`):** the data are close to the exact solution, and the roles reverse. `odeint` matches the data, and `backward_euler` carries the discretization error of its step `dtData`.

## Effect of the integration time step

With `SOLVER = 'odeint'`, the model is (nearly) the exact ODE, so any discretization error in the synthetic data shows up in the learned parameters. The table lists the learned values for the default parameters (true $k = 1$ N/m, $b = 1$ N s/m, `totalTime = 500` s). `dtSim` cannot exceed `dtData`, so `dtData` is raised with it in the last two rows.

| `dtSim` [s] | `dtData` [s] | Learned $k$ [N/m] | Learned $b$ [N s/m] |
|---|---|---|---|
| 0.001 | 0.1 | 1.0005 (+0.05%) | 1.001 (+0.1%) |
| 0.01 | 0.1 | 1.005 (+0.5%) | 1.010 (+1.0%) |
| 0.1 (default) | 0.1 | 1.056 (+5.6%) | 1.098 (+9.8%) |
| 0.2 | 0.2 | 1.123 (+12%) | 1.191 (+19%) |
| 0.5 | 0.5 | 1.385 (+38%) | 1.427 (+43%) |

With the default `dtSim = 0.1` s, the learned damping is $b = 1.098$ N s/m, 9.8% above the true value, and the learned stiffness is $k = 1.056$ N/m, 5.6% above. Both errors shrink in proportion to `dtSim`.

Backward Euler is numerically dissipative: in a steady oscillation at frequency $\omega$, its amplification factor per step, $1/|1 - i\omega\Delta t|$, is less than one, so it damps each state at an artificial rate of about $\tfrac{1}{2}\omega^2\Delta t$. In the equation of motion, this numerical dissipation acts as extra damping of about $m\omega^2\Delta t$ (0.1 N s/m for $\Delta t = 0.1$ s) and extra stiffness of about $\tfrac{1}{2}b\omega^2\Delta t$ (0.05 N/m), so the learned model compensates mainly with a larger $b$ and partly with a larger $k$.

The numerical dissipation acts on the steady forced response, so `totalTime` must be long compared with the decay time of the start-up transient, $2m/b = 2$ s. For an overdamped system ($b > 2\sqrt{mk}$), this decay time grows to about $b/k$, so a larger $b$ needs a longer record.

## Running the code

From the repository root, install the dependencies, including PyTorch and torchdiffeq, and run the script:

```bash
pip install -r requirements-neuralode.txt
python src/neuralODE/neuralODE_massSpringDamper.py
```

The solver, the physical parameters, the data settings (`totalTime`, `dtSim`, `dtData`), the initial guesses and the training settings (`nEpochs`, `learningRate`) are at the top of `main()`.
