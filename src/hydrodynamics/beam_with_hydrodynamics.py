"""
Hydrodynamics demo: an elastic cantilever beam, clamped at one end and free at the other,
released from rest under gravity in a fluid.

Implicit Euler time stepping with Newton's method. The fluid applies a drag force at
every node, from either the low Reynolds number model (Resistive Force Theory) or the
high Reynolds number model (quadratic drag); pick the model and its parameters at the
top of main(). The gradients and Hessians of the elastic energies (gradEb, hessEb,
gradEs, hessEs) are in ../../appendix.

Run from anywhere, e.g.:  python src/hydrodynamics/beam_with_hydrodynamics.py
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
    """Voronoi length of each node in configuration q (half an edge at the two end nodes)."""
    nv = q.size // 2
    voronoiLength = np.zeros(nv)
    for k in range(1, nv - 1):  # interior nodes
        voronoiLength[k] = 0.5 * (np.linalg.norm(q[2*k:2*k + 2] - q[2*k - 2:2*k]) +
                                  np.linalg.norm(q[2*k + 2:2*k + 4] - q[2*k:2*k + 2]))
    voronoiLength[0] = 0.5 * np.linalg.norm(q[2:4] - q[0:2])
    voronoiLength[nv - 1] = 0.5 * np.linalg.norm(q[2*nv - 2:2*nv] - q[2*nv - 4:2*nv - 2])

    return voronoiLength


# ---------------------------------------------------------------------------
# Hydrodynamic forces: drag on every node from its velocity relative to the fluid
# ---------------------------------------------------------------------------

def computeTangent(q):
    """Unit tangent at each node: along the edge at the two end nodes, and the normalized
    average of the two adjacent edge tangents at the interior nodes."""
    nv = q.size // 2
    edgeTangent = np.zeros((nv - 1, 2))
    for k in range(nv - 1):
        edge = q[2*k + 2:2*k + 4] - q[2*k:2*k + 2]
        edgeTangent[k] = edge / np.linalg.norm(edge)

    tangent = np.zeros((nv, 2))
    tangent[0] = edgeTangent[0]
    tangent[nv - 1] = edgeTangent[nv - 2]
    for k in range(1, nv - 1):
        t = edgeTangent[k - 1] + edgeTangent[k]
        tangent[k] = t / np.linalg.norm(t)

    return tangent


def getFh_lowRe(q, q_old, dt, Ct, Cp, voronoiLength, u_f):
    """
    Low Reynolds number drag (Resistive Force Theory) Fh and its Jacobian Jh. The
    tangential and normal drag coefficients are Ct and Cp; the node tangents are frozen
    in the Jacobian.
    """
    ndof = q.size
    nv = ndof // 2
    Fh = np.zeros(ndof)
    Jh = np.zeros((ndof, ndof))
    tangent = computeTangent(q)

    for k in range(nv):
        ind = np.arange(2*k, 2*k + 2)               # x_k, y_k
        u_rel = (q[ind] - q_old[ind]) / dt - u_f    # velocity relative to the fluid
        Pt = np.outer(tangent[k], tangent[k])       # projection along the tangent
        Pn = np.eye(2) - Pt                         # projection normal to the tangent
        C = Ct * Pt + Cp * Pn
        Fh[ind] = -voronoiLength[k] * C @ u_rel
        Jh[np.ix_(ind, ind)] = -voronoiLength[k] / dt * C

    return Fh, Jh


def getFh_highRe(q, q_old, dt, rho_f, Cd, r0, voronoiLength, u_f, eps=1e-6):
    """
    High Reynolds number (quadratic) drag Fh and its Jacobian Jh, for a fluid of density
    rho_f, drag coefficient Cd and a rod of radius r0. The speed is smoothed with eps
    [m/s] so that the Jacobian stays finite when a node moves with the fluid.
    """
    ndof = q.size
    nv = ndof // 2
    Fh = np.zeros(ndof)
    Jh = np.zeros((ndof, ndof))

    for k in range(nv):
        ind = np.arange(2*k, 2*k + 2)               # x_k, y_k
        u_rel = (q[ind] - q_old[ind]) / dt - u_f    # velocity relative to the fluid
        speed = np.sqrt(u_rel @ u_rel + eps**2)     # smoothed |u_rel|
        A = 2 * r0 * voronoiLength[k]               # projected area of the node
        c = 0.5 * rho_f * Cd * A
        Fh[ind] = -c * speed * u_rel
        Jh[np.ix_(ind, ind)] = -c / dt * (speed * np.eye(2) + np.outer(u_rel, u_rel) / speed)

    return Fh, Jh


def getFh(q, q_old, dt, voronoiLength, hydro):
    """Hydrodynamic force Fh and its Jacobian Jh for the model chosen in hydro['model']."""
    if hydro['model'] == 'lowRe':
        return getFh_lowRe(q, q_old, dt, hydro['Ct'], hydro['Cp'], voronoiLength, hydro['u_f'])
    if hydro['model'] == 'highRe':
        return getFh_highRe(q, q_old, dt, hydro['rho_f'], hydro['Cd'], hydro['r0'],
                            voronoiLength, hydro['u_f'])
    if hydro['model'] == 'none':
        return np.zeros(q.size), np.zeros((q.size, q.size))
    raise ValueError(f"Unknown hydrodynamic model '{hydro['model']}': use 'none', 'lowRe' or 'highRe'.")


# ---------------------------------------------------------------------------
# One implicit Euler step (Newton's method) with fixed DOFs
# ---------------------------------------------------------------------------

def objfun(q_guess, q_old, u_old, dt, qtol, maxIter,
           m, mMat, EI, EA, W, refLength, voronoiLength, freeIndex, hydro):
    """
    Solve for q_new, updating only the free DOFs; the fixed DOFs keep the values they
    have in q_guess. Returns (q_new, flag): flag = -1 if Newton did not converge within
    maxIter iterations.
    """
    q_new = q_guess.copy()
    iter_count = 0
    error = qtol * 10

    while error > qtol:
        Fb, Jb = getFb(q_new, EI, voronoiLength)
        Fs, Js = getFs(q_new, EA, refLength)
        Fh, Jh = getFh(q_new, q_old, dt, voronoiLength, hydro)

        # Equations of motion and Jacobian
        f = m * (q_new - q_old) / dt**2 - m * u_old / dt - (Fb + Fs + Fh + W)
        J = mMat / dt**2 - (Jb + Js + Jh)

        # Newton update of the free DOFs only
        f_free = f[freeIndex]
        J_free = J[np.ix_(freeIndex, freeIndex)]
        dq_free = np.linalg.solve(J_free, f_free)
        q_new[freeIndex] = q_new[freeIndex] - dq_free
        error = np.linalg.norm(dq_free)

        iter_count += 1
        if error > qtol and iter_count >= maxIter:
            return q_new, -1

    return q_new, 1


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rod(q, ndof, t, RodLength, hydroModel):
    """Plot the beam at time t."""
    plt.figure(1)
    plt.clf()
    plt.plot(q[0:ndof:2], q[1:ndof:2], 'ro-', label='Beam')
    plt.title(f'Hydrodynamics: {hydroModel}, Time={t:.2f} seconds')
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
    # Hydrodynamics: pick a model and set its parameters
    hydroModel = 'lowRe'                # 'none', 'lowRe' (Resistive Force Theory) or 'highRe' (quadratic drag)
    u_f = np.array([0.0, 0.0])          # background fluid velocity [m/s]
    mu = 1.0                            # lowRe: dynamic viscosity [Pa s]
    rho_f = 1000.0                      # highRe: fluid density [kg/m^3]
    Cd = 1.1                            # highRe: drag coefficient

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

    # Resistive Force Theory drag coefficients of a slender rod [Pa s]
    Ct = 2 * np.pi * mu / (np.log(2 * RodLength / h) - 0.5)    # tangential
    Cp = 4 * np.pi * mu / (np.log(2 * RodLength / h) + 0.5)    # normal
    hydro = {'model': hydroModel, 'u_f': u_f, 'Ct': Ct, 'Cp': Cp,
             'rho_f': rho_f, 'Cd': Cd, 'r0': h}

    # Mass and weight: the total mass is split equally among the nodes
    totalMass = 0.2                     # [kg]
    m = np.full(ndof, totalMass / N)    # same mass for x_k and y_k
    mMat = np.diag(m)
    g = np.array([0.0, -9.81])          # gravity [m/s^2]
    W = m * np.tile(g, N)               # weight of every node, along -y

    # Fixed DOFs: the first two nodes are clamped
    fixedIndex = [0, 1, 2, 3]
    freeIndex = np.setdiff1d(np.arange(ndof), fixedIndex)

    # Time stepping and Newton solver settings
    dt = 0.005                      # time step [s]
    totalTime = 10.0                # total simulation time [s]
    qtol = RodLength / N * 1e-3     # tolerance on the position change [m]
    maxIter = 10                    # maximum number of Newton iterations

    # Initial conditions: at rest
    q_old = q.copy()
    u_old = np.zeros(ndof)

    for time_iter, t in enumerate(np.arange(0, totalTime, dt)):

        # Guess for Newton's method: the fixed DOFs keep these values
        q_guess = q_old.copy()
        q_new, flag = objfun(q_guess, q_old, u_old, dt, qtol, maxIter,
                             m, mMat, EI, EA, W, refLength, voronoiLength, freeIndex, hydro)
        if flag < 0:
            print(f'Simulation did not converge at time {t} seconds.')
            break

        u_old = (q_new - q_old) / dt
        q_old = q_new.copy()

        if time_iter % 50 == 0:
            plot_rod(q_new, ndof, t, RodLength, hydroModel)

    plt.show()


if __name__ == '__main__':
    main()
