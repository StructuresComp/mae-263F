"""
MAE 263F, Fall 2026 - MS Comprehensive Exam: extra assignment (TEMPLATE).

Write simulate() and submit this file as NewmarkBetaAssignment.py. You may add any other
functions to this file. Do not change main(). Use only numpy, matplotlib, and the Python
standard library.

A ball of mass m hangs from a vertical linear spring (stiffness k) under gravity, with
linear viscous drag (coefficient b). y is the vertical position (positive up), measured
from the position where the spring is unstretched:

    m y'' = -k y - b y' - m g,      y(0) = 0,  y'(0) = 0

Examples:
    python NewmarkBetaAssignment.py --method euler --k 0
    python NewmarkBetaAssignment.py --method newmark --k 10 --b 0
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')  # save figures without opening a window
import matplotlib.pyplot as plt


def simulate(method, m, k, b, g, dt, T, beta, gamma):
    """
    Simulate the ball from y(0) = 0, v(0) = 0 with method 'euler' (implicit Euler) or
    'newmark' (Newmark-Beta with beta, gamma; ignored for 'euler'). Solve each time step
    with the Newton-Raphson method.
    Returns numpy arrays t, y, v, E of length n + 1, with n = round(T / dt) and t[i] = i * dt,
    where E is the total energy (Eq. 2 of the handout).
    """
    # TODO
    raise NotImplementedError


def main():
    # Do not change main(): it reads the options, calls simulate(), prints one line, and saves the figure.
    p = argparse.ArgumentParser(description='Ball on a spring under gravity: implicit Euler vs. Newmark-Beta.')
    p.add_argument('--method', choices=['euler', 'newmark'], default='newmark')
    p.add_argument('--m', type=float, default=1.0, help='mass [kg]')
    p.add_argument('--k', type=float, default=10.0, help='spring stiffness [N/m]')
    p.add_argument('--b', type=float, default=0.0, help='viscous drag coefficient [N s/m]')
    p.add_argument('--g', type=float, default=9.81, help='gravity [m/s^2]')
    p.add_argument('--dt', type=float, default=0.01, help='time step [s]')
    p.add_argument('--T', type=float, default=20.0, help='total time [s]')
    p.add_argument('--beta', type=float, default=0.25)
    p.add_argument('--gamma', type=float, default=0.5)
    args = p.parse_args()

    t, y, v, E = simulate(args.method, args.m, args.k, args.b, args.g, args.dt, args.T, args.beta, args.gamma)

    print(f'method={args.method} m={args.m:g} k={args.k:g} b={args.b:g} dt={args.dt:g} T={args.T:g} '
          f'y_final={y[-1]:.6g} v_final={v[-1]:.6g} E_initial={E[0]:.6g} E_final={E[-1]:.6g}')

    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax[0].plot(t, y, label=args.method)
    if args.b == 0:  # exact solution (see the handout)
        if args.k == 0:
            y_exact = -0.5 * args.g * t**2
        else:
            y_exact = -args.m * args.g / args.k * (1.0 - np.cos(np.sqrt(args.k / args.m) * t))
        ax[0].plot(t, y_exact, 'k--', label='exact')
    ax[0].set_ylabel('y [m]')
    ax[0].legend()
    ax[1].plot(t, E)
    ax[1].set_xlabel('t [s]')
    ax[1].set_ylabel('E [J]')
    fig.suptitle(f'{args.method}: m={args.m:g}, k={args.k:g}, b={args.b:g}, dt={args.dt:g}')
    fig.tight_layout()
    if args.method == 'euler':
        fname = f'euler_k{args.k:g}_b{args.b:g}_dt{args.dt:g}.png'
    else:
        fname = f'newmark_beta{args.beta:g}_gamma{args.gamma:g}_k{args.k:g}_b{args.b:g}_dt{args.dt:g}.png'
    fig.savefig(fname, dpi=150)
    print(f'saved {fname}')


if __name__ == '__main__':
    main()
