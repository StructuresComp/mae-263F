"""
Predictor Corrector Method with beams: an elastic beam, clamped at one end, in contact
with a rigid planar ground.

Implicit Euler time stepping with Newton's method. The last node is moved back and
forth along x, so the beam buckles and touches the ground. Contact is handled with a
predictor-corrector step: a node that falls below the ground is fixed on it, and a node
on the ground is released when the ground would have to pull it down to hold it there.
The gradients and Hessians of the elastic energies (gradEb, hessEb, gradEs, hessEs) are
in ../../appendix.

Run from anywhere, e.g.:  python src/predictor_corrector/predictor_corrector_beam_and_planar_ground.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Make ../../appendix importable regardless of the current working directory
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'appendix'))

from gradEb import gradEb
from hessEb import hessEb
from gradEs import gradEs
from hessEs import hessEs


# ---------------------------------------------------------------------------
# Elastic forces: assemble over all bending and stretching springs
# ---------------------------------------------------------------------------

def getFb(q, EI, voronoiLength):
    """Bending force Fb = -dE_b/dq and its Jacobian Jb = -d2E_b/dq2."""
    ndof = q.size
    nv = ndof // 2
    Fb = np.zeros(ndof)
    Jb = np.zeros((ndof, ndof))

    for k in range(1, nv - 1):  # bending springs at the interior nodes
        ind = np.arange(2*k - 2, 2*k + 4)  # x_{k-1}, y_{k-1}, x_k, y_k, x_{k+1}, y_{k+1}
        Fb[ind] -= gradEb(*q[ind], 0, voronoiLength[k], EI)
        Jb[np.ix_(ind, ind)] -= hessEb(*q[ind], 0, voronoiLength[k], EI)

    return Fb, Jb


def getFs(q, EA, refLength):
    """Stretching force Fs = -dE_s/dq and its Jacobian Js = -d2E_s/dq2."""
    ndof = q.size
    nv = ndof // 2
    Fs = np.zeros(ndof)
    Js = np.zeros((ndof, ndof))

    for k in range(nv - 1):  # stretching springs between nodes k and k+1
        ind = np.arange(2*k, 2*k + 4)  # x_k, y_k, x_{k+1}, y_{k+1}
        Fs[ind] -= gradEs(*q[ind], refLength[k], EA)
        Js[np.ix_(ind, ind)] -= hessEs(*q[ind], refLength[k], EA)

    return Fs, Js


# ---------------------------------------------------------------------------
# Reference (undeformed) lengths
# ---------------------------------------------------------------------------

def computeReferenceLength(q):
    """Length of each of the N - 1 edges in configuration q."""
    nv = q.size // 2
    refLength = np.zeros(nv - 1)
    for k in range(nv - 1):
        refLength[k] = np.linalg.norm(q[2*k + 2:2*k + 4] - q[2*k:2*k + 2])

    return refLength


def computeVoronoiLength(q):
    """Voronoi length of each node in configuration q (zero at the two end nodes)."""
    nv = q.size // 2
    voronoiLength = np.zeros(nv)
    for k in range(1, nv - 1):  # interior nodes only
        voronoiLength[k] = 0.5 * (np.linalg.norm(q[2*k:2*k + 2] - q[2*k - 2:2*k]) +
                                  np.linalg.norm(q[2*k + 2:2*k + 4] - q[2*k:2*k + 2]))

    return voronoiLength


# ---------------------------------------------------------------------------
# One implicit Euler step (Newton's method) with fixed DOFs
# ---------------------------------------------------------------------------

def objfun(q_guess, q_old, u_old, dt, qtol, maxIter,
           m, mMat, EI, EA, W, refLength, voronoiLength, freeIndex):
    """
    Solve for q_new, updating only the free DOFs; the fixed DOFs keep the values they
    have in q_guess. Returns (q_new, f, flag): f is the residual of the equations of
    motion, which at a fixed DOF is the reaction force holding it in place;
    flag = -1 if Newton did not converge within maxIter iterations.
    """
    q_new = q_guess.copy()
    iter_count = 0
    error = qtol * 10

    while error > qtol:
        Fb, Jb = getFb(q_new, EI, voronoiLength)
        Fs, Js = getFs(q_new, EA, refLength)

        # Equations of motion and Jacobian
        f = m * (q_new - q_old) / dt**2 - m * u_old / dt - (Fb + Fs + W)
        J = mMat / dt**2 - (Jb + Js)

        # Newton update of the free DOFs only
        f_free = f[freeIndex]
        J_free = J[np.ix_(freeIndex, freeIndex)]
        dq_free = np.linalg.solve(J_free, f_free)
        q_new[freeIndex] = q_new[freeIndex] - dq_free
        error = np.linalg.norm(dq_free)

        iter_count += 1
        if error > qtol and iter_count >= maxIter:
            return q_new, f, -1

    return q_new, f, 1


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rod_and_ground(q, ndof, t, RodLength, groundPosition):
    """Plot the beam and the ground line at time t."""
    plt.figure(1)
    plt.clf()
    plt.plot(q[0:ndof:2], q[1:ndof:2], 'ro-', label='Beam')
    plt.axhline(y=groundPosition, color='k', linestyle='-', linewidth=2, label='Ground')
    plt.title(f'Time={t:.2f} seconds')  # Format the title with the current time
    plt.xlim([0, 2.0 * RodLength])
    plt.ylim([-RodLength, RodLength])
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.grid(True)
    plt.draw()
    plt.pause(0.01)  # Display the figure without blocking


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Geometry: straight beam along x with a tiny random perturbation along y
    N = 20              # number of nodes
    ndof = 2 * N        # number of DOFs
    RodLength = 0.1     # length of the beam [m]
    q = np.zeros(ndof)
    q[0::2] = np.arange(N) * RodLength / (N - 1)
    q[1::2] = np.random.rand(N) * 1e-6

    # Reference lengths of the edges and Voronoi lengths of the nodes
    refLength = computeReferenceLength(q)
    voronoiLength = computeVoronoiLength(q)

    # Material properties
    Y = 1e9             # Young's modulus [Pa]
    h = 0.001           # cross-sectional radius of the beam [m]
    EA = Y * np.pi * h**2
    EI = Y * np.pi / 4 * h**4

    # Mass and weight: the total mass is split equally among the nodes
    totalMass = 0.2                     # [kg]
    m = np.full(ndof, totalMass / N)    # same mass for x_k and y_k
    mMat = np.diag(m)
    g = np.array([0.0, -9.81])          # gravity [m/s^2]
    W = m * np.tile(g, N)               # weight of every node, along -y

    # Ground: horizontal line y = groundPosition with normal +y
    groundPosition = -0.25 * RodLength
    contact_normal = np.array([0.0, 1.0])

    # Fixed DOFs: the first two nodes are clamped forever. The last node is fixed too
    # and is moved to a new target location at every time step.
    fixedIndex = [0, 1, 2, 3, ndof - 2, ndof - 1]

    # Time stepping and Newton solver settings
    dt = 0.005                      # time step [s]
    totalTime = 10.0                # total simulation time [s]
    qtol = RodLength / N * 1e-3     # tolerance on the position change [m]
    maxIter = 10                    # maximum number of Newton iterations

    # Initial conditions: at rest
    q_old = q.copy()
    u_old = np.zeros(ndof)

    for time_iter, t in enumerate(np.arange(0, totalTime, dt)):

        # Slowly move the last node: factor is a triangle wave in tau = t/totalTime
        # that goes from 1 (tau = 0) to 0.5 (tau = 0.5) and back to 1 (tau = 1)
        tau = t / totalTime
        factor = 0.5 * (1 + abs(2*tau - 1))
        targetNodeLocation = np.array([RodLength * factor, 0.0])

        # Predictor: the fixed DOFs keep the values they have in the guess
        q_guess = q_old.copy()
        q_guess[ndof - 2:ndof] = targetNodeLocation
        freeIndex = np.setdiff1d(np.arange(ndof), fixedIndex)
        q_new, f, flag = objfun(q_guess, q_old, u_old, dt, qtol, maxIter,
                                m, mMat, EI, EA, W, refLength, voronoiLength, freeIndex)
        if flag < 0:
            print(f'Simulation did not converge at time {t} seconds.')
            break

        # Contact check (the first two nodes and the last node are always fixed)
        corrector_flag = False
        for i in range(2, N - 1):
            if 2*i + 1 in fixedIndex:
                # Node on the ground: release its y-motion if the reaction force points
                # along -y (at a fixed DOF, the residual f is the reaction force)
                if np.dot(f[2*i:2*i + 2], contact_normal) < 0:
                    corrector_flag = True
                    fixedIndex.remove(2*i + 1)
            elif q_new[2*i + 1] < groundPosition:
                # Free node below the ground: fix its y-coordinate on the ground
                corrector_flag = True
                fixedIndex.append(2*i + 1)
                q_new[2*i + 1] = groundPosition

        # Corrector: solve the same time step again with the updated fixed DOFs
        if corrector_flag:
            freeIndex = np.setdiff1d(np.arange(ndof), fixedIndex)
            q_new, f, flag = objfun(q_new, q_old, u_old, dt, qtol, maxIter,
                                    m, mMat, EI, EA, W, refLength, voronoiLength, freeIndex)

        u_old = (q_new - q_old) / dt
        q_old = q_new.copy()

        if time_iter % 50 == 0:
            plot_rod_and_ground(q_new, ndof, t, RodLength, groundPosition)

    plt.show()


if __name__ == '__main__':
    main()
