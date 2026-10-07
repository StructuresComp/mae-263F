"""
Falling elastic beam with N nodes in a viscous fluid (Section 4.3, Assignment 2).

Implicit Euler time stepping with Newton's method. The gradients and Hessians of
the elastic energies (gradEb, hessEb, gradEs, hessEs) are in ../appendix.

Run from anywhere, e.g.:  python3 src/main_falling_beam_n_nodes.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Make ../appendix importable regardless of the current working directory
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'appendix'))

from gradEb import gradEb
from hessEb import hessEb
from gradEs import gradEs
from hessEs import hessEs


# ---------------------------------------------------------------------------
# Elastic forces: assemble over all bending and stretching springs
# ---------------------------------------------------------------------------

def getFb(q, EI, deltaL):
    """Bending force Fb = -dE_b/dq and its Jacobian Jb = -d2E_b/dq2."""
    ndof = q.size
    nv = ndof // 2
    Fb = np.zeros(ndof)
    Jb = np.zeros((ndof, ndof))

    for k in range(1, nv - 1):  # bending springs at the interior nodes
        ind = np.arange(2*k - 2, 2*k + 4)  # x_{k-1}, y_{k-1}, x_k, y_k, x_{k+1}, y_{k+1}
        Fb[ind] -= gradEb(*q[ind], 0, deltaL, EI)
        Jb[np.ix_(ind, ind)] -= hessEb(*q[ind], 0, deltaL, EI)

    return Fb, Jb


def getFs(q, EA, deltaL):
    """Stretching force Fs = -dE_s/dq and its Jacobian Js = -d2E_s/dq2."""
    ndof = q.size
    nv = ndof // 2
    Fs = np.zeros(ndof)
    Js = np.zeros((ndof, ndof))

    for k in range(nv - 1):  # stretching springs between nodes k and k+1
        ind = np.arange(2*k, 2*k + 4)  # x_k, y_k, x_{k+1}, y_{k+1}
        Fs[ind] -= gradEs(*q[ind], deltaL, EA)
        Js[np.ix_(ind, ind)] -= hessEs(*q[ind], deltaL, EA)

    return Fs, Js


# ---------------------------------------------------------------------------
# One implicit Euler step (Newton's method)
# ---------------------------------------------------------------------------

def objfun(q_guess, q_old, u_old, dt, tol, maximum_iter,
           m, mMat, EI, EA, W, C, deltaL):
    """Solve for q_new. Returns (q_new, flag); flag = -1 if Newton did not converge."""
    q_new = q_guess.copy()
    iter_count = 0
    error = tol * 10

    while error > tol:
        Fb, Jb = getFb(q_new, EI, deltaL)
        Fs, Js = getFs(q_new, EA, deltaL)

        # Viscous force
        Fv = -C @ (q_new - q_old) / dt
        Jv = -C / dt

        # Equations of motion and Jacobian
        f = m * (q_new - q_old) / dt**2 - m * u_old / dt - (Fb + Fs + W + Fv)
        J = mMat / dt**2 - (Jb + Js + Jv)

        q_new = q_new - np.linalg.solve(J, f)
        error = np.linalg.norm(f)

        iter_count += 1
        if iter_count > maximum_iter:
            return q_new, -1

    return q_new, 1


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(nv, dt, totalTime, plotShapes=False):
    """
    Simulate the falling beam with nv nodes (nv odd).

    Returns t, y_mid, v_mid, angle, q: time, vertical position and velocity of
    the middle node, turning angle at the middle node [deg], and the final DOF
    vector q = [x1, y1, x2, y2, ...].
    """
    if nv % 2 == 0 or nv < 3:
        raise ValueError('nv must be an odd number >= 3 so that a middle node exists.')

    ndof = 2 * nv

    # Geometry
    RodLength = 0.10
    deltaL = RodLength / (nv - 1)

    # Sphere radii: R_mid = 0.025 at the middle node, deltaL/10 elsewhere
    midNode = (nv - 1) // 2  # 0-based index of the middle node
    R = np.full(nv, deltaL / 10)
    R[midNode] = 0.025

    # Material and fluid properties
    rho_metal = 7000
    rho_fluid = 1000
    r0 = 1e-3          # cross-sectional radius of the beam
    Y = 1e9            # Young's modulus
    visc = 1000.0      # fluid viscosity
    EI = Y * np.pi * r0**4 / 4
    EA = Y * np.pi * r0**2

    # Newton solver settings
    maximum_iter = 100
    tol = EI / RodLength**2 * 1e-3

    # Mass, weight (with buoyancy), and viscous damping for each node
    m = np.repeat(4 / 3 * np.pi * R**3 * rho_metal, 2)  # same mass for x_k and y_k
    mMat = np.diag(m)

    W = np.zeros(ndof)
    W[1::2] = -4 / 3 * np.pi * R**3 * (rho_metal - rho_fluid) * 9.8  # along -y

    C = np.diag(np.repeat(6 * np.pi * visc * R, 2))

    # Initial conditions: straight beam along x, at rest
    q0 = np.zeros(ndof)
    q0[0::2] = np.arange(nv) * deltaL
    u = np.zeros(ndof)

    # Storage (index 0 is t = 0)
    Nsteps = round(totalTime / dt)
    t = np.arange(Nsteps + 1) * dt
    y_mid = np.zeros(Nsteps + 1)
    v_mid = np.zeros(Nsteps + 1)
    angle = np.zeros(Nsteps + 1)

    if plotShapes:
        plt.figure(1)
        plt.plot(q0[::2], q0[1::2], 'o-', label='t = 0 s')

    for timeStep in range(1, Nsteps + 1):
        q, flag = objfun(q0, q0, u, dt, tol, maximum_iter,
                         m, mMat, EI, EA, W, C, deltaL)
        if flag < 0:
            raise RuntimeError(f'Newton did not converge at t = {t[timeStep]:.4f} s')

        u = (q - q0) / dt
        q0 = q

        y_mid[timeStep] = q[2*midNode + 1]
        v_mid[timeStep] = u[2*midNode + 1]

        # Turning angle at the middle node
        e1 = q[2*midNode:2*midNode + 2] - q[2*midNode - 2:2*midNode]
        e2 = q[2*midNode + 2:2*midNode + 4] - q[2*midNode:2*midNode + 2]
        angle[timeStep] = np.degrees(np.arctan2(abs(e1[0]*e2[1] - e1[1]*e2[0]), e1 @ e2))

        if plotShapes and timeStep%10 == 0:
            plt.figure(1)
            plt.clf()
            plt.plot(q[::2], q[1::2], 'o-', label=f't = {t[timeStep]:g} s')
            plt.title(f't={t[timeStep]:.6f}')  # Format the title with the current time
            plt.axis('equal')  # Set equal scaling
            plt.xlabel('x [m]')
            plt.ylabel('y [m]')
            plt.draw()
            plt.pause(0.01)  # Display the figure without blocking

    return t, y_mid, v_mid, angle, q


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # Assignment 2, parts 1 and 2
    nv = 21          # number of nodes (odd)
    dt = 1e-2        # time step [s]
    totalTime = 50   # total simulation time [s]

    t, y_mid, v_mid, angle, q = simulate(nv, dt, totalTime, plotShapes=True)
    print(f'N = {nv}, dt = {dt}: terminal velocity of the middle node = {v_mid[-1]:.6f} m/s')

    plt.figure(1)
    plt.axis('equal')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'Shape of the beam (N = {nv})')
    plt.legend()
    plt.savefig('fallingBeam_shapes.png')

    plt.figure(2)
    plt.plot(t, y_mid)
    plt.xlabel('Time, t [s]')
    plt.ylabel('Vertical position of middle node, y [m]')
    plt.savefig('fallingBeam_position.png')

    plt.figure(3)
    plt.plot(t, v_mid)
    plt.xlabel('Time, t [s]')
    plt.ylabel('Vertical velocity of middle node, v [m/s]')
    plt.savefig('fallingBeam_velocity.png')

    plt.figure(4)
    plt.plot(q[::2], q[1::2], 'ko-')
    plt.axis('equal')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title(f'Final deformed shape, t = {totalTime} s')
    plt.savefig('fallingBeam_finalShape.png')

    plt.figure(5)
    plt.plot(t, angle, 'r')
    plt.xlabel('Time, t [s]')
    plt.ylabel('Turning angle at middle node [deg]')
    plt.savefig('fallingBeam_angle.png')
    plt.pause(0.01)  # Display figures 1-5 before the (slow) convergence study

    plt.show()
