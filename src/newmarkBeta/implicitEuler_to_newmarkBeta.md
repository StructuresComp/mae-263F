# From Implicit Euler to Newmark-Beta

## Newmark relations

With parameters $\beta$ and $\gamma$:

$$
\mathbf{q}_{n+1} = \mathbf{q}_n + \Delta t\,\mathbf{u}_n + \Delta t^2\left[\left(\tfrac{1}{2}-\beta\right)\mathbf{a}_n + \beta\,\mathbf{a}_{n+1}\right]
$$

$$
\mathbf{u}_{n+1} = \mathbf{u}_n + \Delta t\left[(1-\gamma)\,\mathbf{a}_n + \gamma\,\mathbf{a}_{n+1}\right]
$$

Solving the first relation for the new acceleration:

$$
\mathbf{a}_{n+1} = \frac{\mathbf{q}_{n+1} - \mathbf{q}_n - \Delta t\,\mathbf{u}_n}{\beta\,\Delta t^2} - \left(\frac{1}{2\beta} - 1\right)\mathbf{a}_n
$$

| Scheme | $\beta$ | $\gamma$ | Behavior |
|---|---|---|---|
| Average acceleration (trapezoidal) | 0.25 | 0.5 | Time-symmetric, no numerical damping |
| Fully implicit Newmark | 0.5 | 1.0 | Strong numerical damping |

## Changes to the simulation

`objfun` stays as implicit Euler. Newmark-Beta is a separate function, `objfun_Newmark`, which uses the helper `update_Newmark` for the relations above. The `useNewmark` setting in `main()` chooses between them.

1. **Track acceleration.** The state carried between time steps becomes $(\mathbf{q}, \mathbf{u}, \mathbf{a})$ instead of $(\mathbf{q}, \mathbf{u})$. Acceleration is not an extra unknown; Newton's method still solves only for the free DOFs of $\mathbf{q}_{n+1}$. Both functions return $\mathbf{a}_{n+1}$ along with $\mathbf{q}_{n+1}$, $\mathbf{u}_{n+1}$ and the convergence flag; `objfun` computes it as $\mathbf{a}_{n+1} = (\mathbf{u}_{n+1} - \mathbf{u}_n)/\Delta t$.

2. **Initial acceleration.** Before the time loop, solve the equation of motion at $t = 0$ on the free DOFs (zero on the fixed DOFs):

$$
\mathbf{M}\,\mathbf{a}_0 = \mathbf{F}^{\textrm{int}}(\mathbf{q}_0) + \mathbf{W}
$$

3. **Residual and Jacobian.** At each Newton iteration of `objfun_Newmark`, compute $\mathbf{a}_{n+1}$ from the current guess of $\mathbf{q}_{n+1}$, then

| | Implicit Euler (`objfun`) | Newmark-Beta (`objfun_Newmark`) |
|---|---|---|
| Residual | $\mathbf{M}\dfrac{\mathbf{q}_{n+1} - \mathbf{q}_n}{\Delta t^2} - \mathbf{M}\dfrac{\mathbf{u}_n}{\Delta t} - \mathbf{F}^{\textrm{int}} - \mathbf{W}$ | $\mathbf{M}\,\mathbf{a}_{n+1} - \mathbf{F}^{\textrm{int}} - \mathbf{W}$ |
| Jacobian | $\dfrac{\mathbf{M}}{\Delta t^2} - \dfrac{\partial \mathbf{F}^{\textrm{int}}}{\partial \mathbf{q}}$ | $\dfrac{\mathbf{M}}{\beta\,\Delta t^2} - \dfrac{\partial \mathbf{F}^{\textrm{int}}}{\partial \mathbf{q}}$ |

   where $\mathbf{F}^{\textrm{int}} = \mathbf{F}_b + \mathbf{F}_s$ is the elastic force.

4. **Update after convergence.** `objfun` keeps the implicit Euler velocity update $\mathbf{u}_{n+1} = (\mathbf{q}_{n+1} - \mathbf{q}_n)/\Delta t$, and `objfun_Newmark` uses the Newmark relations for $\mathbf{a}_{n+1}$ and $\mathbf{u}_{n+1}$ above. Then advance all three: $(\mathbf{q}_n, \mathbf{u}_n, \mathbf{a}_n) \leftarrow (\mathbf{q}_{n+1}, \mathbf{u}_{n+1}, \mathbf{a}_{n+1})$.

5. **Beam parameters.** For both methods, `totalMass` changes from 0.2 to 0.02 kg and `dt` from 5 to 2 ms. Without numerical damping, neither change is enough alone: at 0.2 kg and 2 ms, or 0.02 kg and 5 ms, the energy grows until Newton's method fails.

## What to expect

With $\beta = 0.25$, $\gamma = 0.5$ and no physical damping, the beam keeps oscillating about its static shape instead of settling, as it does with implicit Euler. This is the correct behavior for an undamped system. Between the first and last 2 s of a 10 s run, the tip's peak-to-peak vertical motion stays at 0.51 beam lengths, while with implicit Euler it falls from 0.49 to 0.005.
