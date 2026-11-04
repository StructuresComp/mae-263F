"""
Cantilever beam: an elastic beam, clamped at one end and free at the other, released
from rest under gravity.

Time stepping with Newton's method, using either implicit Euler (objfun) or the
Newmark-Beta method with parameters beta and gamma (objfun_Newmark). The gradients and
Hessians of the elastic energies (gradEb, hessEb, gradEs, hessEs) are in ../../appendix.

Run from anywhere, e.g.:  python src/newmarkBeta/cantilever_newmarkBeta.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Make ../../appendix importable regardless of the current working directory
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'appendix'))

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
    have in q_guess. Returns (q_new, u_new, a_new, flag): flag = -1 if Newton did not
    converge within maxIter iterations.
    """
    q_new = q_guess.copy() # Position
    u_new = u_old.copy() # Velocity
    a_new = np.zeros_like(u_old) # Acceleration
    
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
            return q_new, u_new, a_new, -1

    # Update velocity and acceleration
    u_new = (q_new - q_old) / dt # New vecocity
    a_new = (u_new - u_old) / dt # New acceleration
    return q_new, u_new, a_new, 1

#
# Objective function for the Newmark-Beta method
#

def update_Newmark(q_new, q_old, u_old, a_old, beta, gamma, dt):
    """Update velocity and acceleration using the Newmark-Beta method."""
    a_new = (0.5 / beta) * (2.0 / dt * ( (q_new - q_old) / dt - u_old) - (1.0 - 2.0 * beta) * a_old)
    u_new = u_old + (1.0 - gamma) * dt * a_old + gamma * dt * a_new
    return u_new, a_new

def objfun_Newmark(q_guess, q_old, u_old, a_old, beta, gamma, dt, qtol, maxIter,
           m, mMat, EI, EA, W, refLength, voronoiLength, freeIndex):
    q_new = q_guess.copy() # Position
    u_new = u_old.copy() # Velocity
    a_new = a_old.copy() # Acceleration

    iter_count = 0
    error = qtol * 10

    while error > qtol:
        Fb, Jb = getFb(q_new, EI, voronoiLength)
        Fs, Js = getFs(q_new, EA, refLength)

        # Equations of motion and Jacobian
        u_new, a_new = update_Newmark(q_new, q_old, u_old, a_old, beta, gamma, dt)
        f = m * a_new - (Fb + Fs + W)
        J = mMat / (beta * dt**2) - (Jb + Js)

        # Newton update of the free DOFs only
        f_free = f[freeIndex]
        J_free = J[np.ix_(freeIndex, freeIndex)]
        dq_free = np.linalg.solve(J_free, f_free)
        q_new[freeIndex] = q_new[freeIndex] - dq_free
        error = np.linalg.norm(dq_free)

        iter_count += 1
        if error > qtol and iter_count >= maxIter:
            return q_new, u_new, a_new, -1

    # Update velocity and acceleration
    u_new, a_new = update_Newmark(q_new, q_old, u_old, a_old, beta, gamma, dt)
    return q_new, u_new, a_new, 1

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rod(q, ndof, t, RodLength):
    """Plot the beam at time t."""
    plt.figure(1)
    plt.clf()
    plt.plot(q[0:ndof:2], q[1:ndof:2], 'ro-', label='Beam')
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

    use_Newmark = True  # Use Newmark-Beta method if True, else use implicit Euler
    beta = 0.25
    gamma = 0.5

    # Geometry: straight beam along x
    N = 20              # number of nodes
    ndof = 2 * N        # number of DOFs
    RodLength = 0.1     # length of the beam [m]
    q = np.zeros(ndof)
    q[0::2] = np.arange(N) * RodLength / (N - 1)

    # Reference lengths of the edges and Voronoi lengths of the nodes
    refLength = computeReferenceLength(q)
    voronoiLength = computeVoronoiLength(q)

    # Material properties
    Y = 1e9             # Young's modulus [Pa]
    h = 0.001           # cross-sectional radius of the beam [m]
    EA = Y * np.pi * h**2
    EI = Y * np.pi / 4 * h**4

    # Mass and weight: the total mass is split equally among the nodes
    totalMass = 0.02                    # [kg]
    m = np.full(ndof, totalMass / N)    # same mass for x_k and y_k
    mMat = np.diag(m)
    g = np.array([0.0, -9.81])          # gravity [m/s^2]
    W = m * np.tile(g, N)               # weight of every node, along -y

    # Fixed DOFs: the first two nodes are clamped
    fixedIndex = [0, 1, 2, 3]
    freeIndex = np.setdiff1d(np.arange(ndof), fixedIndex)

    # Time stepping and Newton solver settings
    dt = 0.002                      # time step [s]
    totalTime = 10.0                # total simulation time [s]
    qtol = RodLength / N * 1e-3     # tolerance on the position change [m]
    maxIter = 10                    # maximum number of Newton iterations

    # Initial conditions: at rest, with the acceleration from the equations of motion
    q_old = q.copy() # Old position
    u_old = np.zeros(ndof) # Old velocity
    a_old = np.zeros(ndof) # Old acceleration
    Fb, _ = getFb(q_old, EI, voronoiLength)
    Fs, _ = getFs(q_old, EA, refLength)
    a_old[freeIndex] = ((Fb + Fs + W) / m)[freeIndex]

    time_array = np.arange(0, totalTime, dt)
    endZ = np.zeros(time_array.size)    # y-coordinate of the last node at each time step

    for time_iter, t in enumerate(time_array):

        # Guess for Newton's method: the fixed DOFs keep these values
        q_guess = q_old.copy()
        if use_Newmark:
            q_new, u_new, a_new, flag = objfun_Newmark(q_guess, q_old, u_old, a_old, beta, gamma, dt, qtol, maxIter,
                                            m, mMat, EI, EA, W, refLength, voronoiLength, freeIndex)
        else: # Do not use Newmark-Beta method
            q_new, u_new, a_new, flag = objfun(q_guess, q_old, u_old, dt, qtol, maxIter,
                                            m, mMat, EI, EA, W, refLength, voronoiLength, freeIndex)
        if flag < 0:
            print(f'Simulation did not converge at time {t} seconds.')
            break

        # Update for the next time step
        a_old = a_new.copy()
        u_old = u_new.copy()
        q_old = q_new.copy()

        endZ[time_iter] = q_new[-1]

        if time_iter % 50 == 0:
            plot_rod(q_new, ndof, t, RodLength)

    # y-coordinate of the last node vs. time (q_new is the state at the end of the step)
    plt.figure(2)
    plt.plot(time_array + dt, endZ, 'ro-')
    plt.xlabel('Time, t [sec]')
    plt.ylabel('y-coord of the last node, $\\delta_y$ [m]')
    plt.show()


if __name__ == '__main__':
    main()
