"""
Functions for the Discrete Elastic Shells (plates and shells) simulator.

Copyright M. Khalid Jawed (khalidjm@seas.ucla.edu). License: CC BY-NC
You should use this code at your own risk.

Python version of meshMe.m (edges and hinges), grad_hess_elastic.m (elastic forces),
and update.m (one time step) in "Plate Simulation_v1". The gradients and Hessians of
the elastic energies (gradEs_hessEs_Shell and gradEb_hessEb_Shell) are in the appendix
folder (../../appendix).
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Make ../../appendix importable regardless of the current working directory
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'appendix'))

from gradEs_hessEs_Shell import gradEs_hessEs_Shell
from gradEb_hessEb_Shell import gradEb_hessEb_Shell, getTheta


# ---------------------------------------------------------------------------
# Mesh connectivity: edges (stretching springs) and hinges (bending springs)
# ---------------------------------------------------------------------------

def getEdgesAndHinges(elements):
    """
    Find the edges and the hinges (bending elements) of a triangle mesh.

    Parameters:
    - elements: (nt, 3) integer array; row k contains the three node indices of
      triangle k, ordered counterclockwise (as returned by mesh_plate).

    Returns:
    - edges: (ne, 2) integer array; row k is (node0, node1) of edge k, node0 < node1.
    - hinges: (nh, 4) integer array; row k is (node0, node1, node2, node3) of hinge k.
      node0-node1 is the edge shared by two triangles; node2 and node3 are the other
      nodes of these two triangles (see the figure in gradEb_hessEb_Shell.py):
      node2 is on the left of node0 -> node1 and node3 is on the right.

    Note: the ordering of node2 and node3 only changes the sign of the hinge angle.
    Since thetaBar is computed with the same ordering, the energy does not change.
    """
    # For each edge (node0, node1) with node0 < node1, store the third node of the
    # triangle on its left and the third node of the triangle on its right.
    leftNode = {}
    rightNode = {}
    for element in elements:
        for i in range(3):
            na = element[i]
            nb = element[(i + 1) % 3]
            nc = element[(i + 2) % 3] # In a counterclockwise triangle, nc is on the left of na -> nb
            if na < nb:
                leftNode[(na, nb)] = nc
            else:
                rightNode[(nb, na)] = nc

    edgeList = sorted(set(leftNode.keys()) | set(rightNode.keys()))
    edges = np.array(edgeList, dtype=int)

    # Interior edges (shared by two triangles) are hinges; boundary edges are not.
    hingeList = []
    for (n0, n1) in edgeList:
        if (n0, n1) in leftNode and (n0, n1) in rightNode:
            hingeList.append([n0, n1, leftNode[(n0, n1)], rightNode[(n0, n1)]])
    hinges = np.array(hingeList, dtype=int).reshape(-1, 4)

    return edges, hinges


def getThetaBar(q, hinges):
    """
    Natural (undeformed) hinge angle of each hinge, computed from the DOF vector q.
    """
    nh = hinges.shape[0]
    thetaBar = np.zeros(nh)
    for kHinge in range(nh):
        node0, node1, node2, node3 = hinges[kHinge, :]
        thetaBar[kHinge] = getTheta(q[3*node0:3*node0+3], q[3*node1:3*node1+3],
                                    q[3*node2:3*node2+3], q[3*node3:3*node3+3])
    return thetaBar


# ---------------------------------------------------------------------------
# Elastic forces (assembled over all edges and hinges)
# ---------------------------------------------------------------------------

def getFs(q, edges, refLen, ks):

    """
Inputs:
q: DOF vector of size 3*nv (nv is the number of nodes)
edges: ne X 2 matrix - the two nodes of each edge.
refLen: vector of size "ne" containing the undeformed reference length of each edge.
ks: vector of size "ne" containing the stretching stiffness of each edge.

Outputs:
Fs: Stretching force vector of size 3*nv
Js: Stretching force Jacobian of size (3*nv) X (3*nv)
    """

    ndof = len(q)
    Fs = np.zeros(ndof)
    Js = np.zeros((ndof,ndof))
    for kEdge in range(edges.shape[0]):
        node0 = edges[kEdge, 0]
        node1 = edges[kEdge, 1]
        x0 = q[3*node0 : 3*node0+3]
        x1 = q[3*node1 : 3*node1+3]
        ind = np.array([3*node0, 3*node0+1, 3*node0+2,
                        3*node1, 3*node1+1, 3*node1+2])

        dF, dJ = gradEs_hessEs_Shell(x0, x1, refLen[kEdge], ks[kEdge])

        Fs[ind] = Fs[ind] - dF
        Js[np.ix_(ind, ind)] = Js[np.ix_(ind, ind)] - dJ

    return Fs,Js


def getFb(q, hinges, thetaBar, kb):

    """
Inputs:
q: DOF vector of size 3*nv (nv is the number of nodes)
hinges: nh X 4 matrix - the four nodes of each hinge.
thetaBar: vector of size "nh" containing the natural hinge angle of each hinge.
kb: scalar bending stiffness.

Outputs:
Fb: Bending force vector of size 3*nv
Jb: Bending force Jacobian of size (3*nv) X (3*nv)
    """

    ndof = len(q)
    Fb = np.zeros(ndof)
    Jb = np.zeros((ndof,ndof))
    for kHinge in range(hinges.shape[0]):
        node0, node1, node2, node3 = hinges[kHinge, :]
        x0 = q[3*node0 : 3*node0+3]
        x1 = q[3*node1 : 3*node1+3]
        x2 = q[3*node2 : 3*node2+3]
        x3 = q[3*node3 : 3*node3+3]
        ind = np.array([3*node0, 3*node0+1, 3*node0+2,
                        3*node1, 3*node1+1, 3*node1+2,
                        3*node2, 3*node2+1, 3*node2+2,
                        3*node3, 3*node3+1, 3*node3+2])

        dF, dJ = gradEb_hessEb_Shell(x0, x1, x2, x3, thetaBar[kHinge], kb)

        Fb[ind] = Fb[ind] - dF
        Jb[np.ix_(ind, ind)] = Jb[np.ix_(ind, ind)] - dJ

    return Fb,Jb


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def set_axes_equal(ax):
    """
    Set equal aspect ratio for a 3D plot in Matplotlib.
    This function adjusts the limits of the plot to make sure
    that the scale is equal along all three axes.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])


def plotShell(q, edges, fixedNodes, ctime):
    """
    Function to plot the shell: the edges in black and the fixed nodes in red.

    Parameters:
    - q: DOF vector of size 3*nv.
    - edges: ne X 2 matrix - the two nodes of each edge.
    - fixedNodes: array with the indices of the fixed (clamped) nodes.
    - ctime: Current time for title.
    """

    x = q[0::3]
    y = q[1::3]
    z = q[2::3]

    # All the edges as one line; NaN breaks the line between two edges
    nanColumn = np.full(edges.shape[0], np.nan)
    X = np.column_stack((x[edges[:, 0]], x[edges[:, 1]], nanColumn)).flatten()
    Y = np.column_stack((y[edges[:, 0]], y[edges[:, 1]], nanColumn)).flatten()
    Z = np.column_stack((z[edges[:, 0]], z[edges[:, 1]], nanColumn)).flatten()

    fig = plt.figure(1)
    plt.clf()  # Clear the figure
    ax = fig.add_subplot(111, projection='3d')

    # Plot the edges
    ax.plot3D(X, Y, Z, 'k-', linewidth=0.8)

    # Plot the fixed nodes with red triangles
    ax.plot3D(x[fixedNodes], y[fixedNodes], z[fixedNodes], 'r^')

    # Set the title with current time
    ax.set_title(f't={ctime:.2f}')

    # Set axes labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Set equal scaling and a 3D view
    set_axes_equal(ax)
    plt.draw()
    plt.pause(0.01)  # Display the figure without blocking


# ---------------------------------------------------------------------------
# Objective function: one implicit Euler step solved with Newton's method
# ---------------------------------------------------------------------------

def objfun(qGuess, q0, u, freeIndex, dt, tol, maximum_iter,
           massVector, mMat,
           edges, refLen, ks,
           hinges, thetaBar, kb,
           Fg, C):
    """
    Returns q, u, flag; flag = 1 if Newton's method converged and 0 otherwise.
    C: viscous damping coefficient (viscous force = - C * velocity on every DOF).
    """
    q = qGuess
    ndof = len(q)
    iter = 0
    error = 10 * tol

    while error > tol and iter < maximum_iter:
        Fb, Jb = getFb(q, hinges, thetaBar, kb)
        Fs, Js = getFs(q, edges, refLen, ks)

        # Viscous force
        Fv = - C * (q - q0) / dt
        Jv = - C / dt * np.eye(ndof)

        Forces = Fb + Fs + Fg + Fv
        JForces = Jb + Js + Jv
        f = massVector * ((q - q0) / dt**2 - u / dt) - Forces
        J = mMat / dt**2 - JForces
        f_free = f[freeIndex]
        J_free = J[np.ix_(freeIndex, freeIndex)]
        dq_free = np.linalg.solve(J_free, f_free)

        q[freeIndex] = q[freeIndex] - dq_free
        error = np.sum(np.abs(f_free))
        iter += 1

    print(f'Iter={iter}, error={error:.6e}')
    u = (q - q0) / dt
    flag = 1 if error <= tol else 0
    return q, u, flag
