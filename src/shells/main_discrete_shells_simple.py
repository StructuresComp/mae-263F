"""
Discrete Elastic Shells: simple example with four nodes (two triangles sharing one edge).
Three nodes are clamped; the fourth node deforms under gravity.

Copyright M. Khalid Jawed (khalidjm@seas.ucla.edu). License: CC BY-NC
You should use this code at your own risk.

Script version of Main Discrete Shells Simple.ipynb (originally a Colab notebook).
The gradients and Hessians of the elastic energies (gradEs_hessEs_Shell and
gradEb_hessEb_Shell) are in the appendix folder (../../appendix).

Run from anywhere, e.g.:  python src/shells/main_discrete_shells_simple.py
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


def plotShell(q, ctime):
    """
    Function to plot the four-node shell (two triangles sharing the edge x0-x1).

    Parameters:
    - q: DOF vector of size 12, q = [x0, x1, x2, x3].
    - ctime: Current time for title.
    """

    x0 = q[0:3]
    x1 = q[3:6]
    x2 = q[6:9]
    x3 = q[9:12]

    fig = plt.figure(1)
    plt.clf()  # Clear the figure
    ax = fig.add_subplot(111, projection='3d')

    # Plot the two triangles: x0 -> x1 -> x2 -> x0 -> x3 -> x1
    X = np.array([x0[0], x1[0], x2[0], x0[0], x3[0], x1[0]])
    Y = np.array([x0[1], x1[1], x2[1], x0[1], x3[1], x1[1]])
    Z = np.array([x0[2], x1[2], x2[2], x0[2], x3[2], x1[2]])
    ax.plot3D(X, Y, Z, 'ko-')

    # Plot the first node with a red triangle
    ax.plot3D([X[0]], [Y[0]], [Z[0]], 'r^')

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

def objfun(qGuess, q0, u, freeIndex, dt, tol,
           massVector, mMat,
           ks, lk, edges,
           kb, thetaBar, hinges,
           Fg, visc):
    q = qGuess
    ndof = len(q)
    iter = 0
    error = 10 * tol

    while error > tol:

        # Bending force and its Jacobian (loop over hinges)
        Fb = np.zeros(ndof)
        Jb = np.zeros((ndof,ndof))
        for kHinge in range(hinges.shape[0]):
            node0 = hinges[kHinge,0]
            node1 = hinges[kHinge,1]
            node2 = hinges[kHinge,2]
            node3 = hinges[kHinge,3]
            x0 = q[3 * node0:3 * node0 + 3]
            x1 = q[3 * node1:3 * node1 + 3]
            x2 = q[3 * node2:3 * node2 + 3]
            x3 = q[3 * node3:3 * node3 + 3]
            ind = [3 * node0, 3 * node0 + 1, 3 * node0 + 2,
                   3 * node1, 3 * node1 + 1, 3 * node1 + 2,
                   3 * node2, 3 * node2 + 1, 3 * node2 + 2,
                   3 * node3, 3 * node3 + 1, 3 * node3 + 2]
            dF, dJ = gradEb_hessEb_Shell(x0, x1, x2, x3, thetaBar[kHinge], kb)
            Fb[ind] = Fb[ind] - dF
            Jb[np.ix_(ind, ind)] = Jb[np.ix_(ind, ind)] - dJ

        # Stretching force and its Jacobian (loop over edges)
        Fs = np.zeros(ndof)
        Js = np.zeros((ndof,ndof))
        for kEdge in range(edges.shape[0]):
            node0 = edges[kEdge, 0]
            node1 = edges[kEdge, 1]
            x0 = q[3 * node0:3 * node0 + 3]
            x1 = q[3 * node1:3 * node1 + 3]
            ind = [3 * node0, 3 * node0 + 1, 3 * node0 + 2,
                   3 * node1, 3 * node1 + 1, 3 * node1 + 2]
            dF, dJ = gradEs_hessEs_Shell(x0, x1, lk[kEdge], ks[kEdge])
            Fs[ind] = Fs[ind] - dF
            Js[np.ix_(ind, ind)] = Js[np.ix_(ind, ind)] - dJ

        # Viscous force
        Fv = - visc * (q - q0) / dt
        Jv = - visc / dt * np.eye(ndof)

        Forces = Fb + Fs + Fg + Fv
        JForces = Jb + Js + Jv

        f = np.dot(mMat / dt, (q - q0) / dt - u) - Forces
        J = mMat / dt**2 - JForces

        f_free = f[freeIndex]
        J_free = J[np.ix_(freeIndex, freeIndex)]
        dq_free = np.linalg.solve(J_free, f_free)

        q[freeIndex] = q[freeIndex] - dq_free
        error = np.sum(np.abs(f_free))
        iter += 1

    print(f'Iter={iter}, error={error:.6f}')
    u = (q - q0) / dt
    return q, u


# ---------------------------------------------------------------------------
# Main Discrete Shells
#
# Degrees of freedom and nodes: the shell is divided into nv nodes, and the
# DOF vector has size ndof = 3*nv: q = [x_0, y_0, z_0, x_1, y_1, z_1, ...].
# ---------------------------------------------------------------------------

# Inputs
x0 = np.array([0.0,  0.0, 0.0])
x1 = np.array([0.01,  0.0, 0.0])
x2 = np.array([0.005,  0.01, 0.0])
x3 = np.array([0.005, -0.01, 0.0])

q0 = np.concatenate((x0, x1, x2, x3))
plotShell(q0, 0)

nv = 4
ndof = nv * 3

visc = 0 # 0.001  # Viscosity

# Create edges and hinges
# Each row of "edges" is (node0, node1). Each row of "hinges" is (node0, node1, node2, node3),
# where node0-node1 is the shared edge (see the figure in gradEb_hessEb_Shell.py).
edges = np.array([(0, 1), (1, 2), (0, 2), (0, 3), (1, 3)])
hinges = np.array([(0, 1, 2, 3)])

# Elastic stiffness parameters

# Material and geometric parameters
Y = 1.0e6  # Young's Modulus
h = 0.001 # Thickness

# Stiffness variables
kb = 2.0 / np.sqrt(3.0) * Y * h**3.0 / 12.0  # Bending stiffness (Newton-meter)

lk = np.zeros(edges.shape[0]) # Reference length of each edge
ks = np.zeros_like(lk) # Stretching stiffness (Newton-meter)
for kEdge in range(edges.shape[0]):
  node0 = edges[kEdge, 0]
  node1 = edges[kEdge, 1]
  xa = q0[3 * node0:3 * node0 + 3]
  xb = q0[3 * node1:3 * node1 + 3]
  lk[kEdge] = np.linalg.norm(xb - xa)
  ks[kEdge] = np.sqrt(3.0) / 2.0 * Y * h * (lk[kEdge])**2.0

# Time parameters
totalTime = 5  # Simulation time
dt = 0.01  # Time step size
plotEvery = 10  # Plot the shell every plotEvery time steps

# Tolerance
tol = kb / (0.01) * 1e-3 # Approximate

# Mass vector and matrix

# Mass
rho = 10  # Density
totalM = 0.01  # Total mass
dm = totalM / nv  # mass per nodes
massVector = np.zeros(ndof)
for c in range(nv):  # Loop over nodes
    ind = [3*c, 3*c+1, 3*c+2]
    massVector[ind] = dm

mMat = np.diag(massVector)  # ndof x ndof sized mass matrix

# External Force
# Since this does not change with time, we define it here instead of within the time loop.

# Gravity
g = np.array([0, 0, -9.81])
Fg = np.zeros(ndof)
for c in range(nv):  # loop over the nodes
    ind = [3*c, 3*c+1, 3*c+2]
    Fg[ind] = massVector[ind] * g

# Initial DOF vector: q0 is already defined
u = np.zeros_like(q0)  # Initialize velocity

# Natural (undeformed) hinge angle of each hinge
thetaBar = np.zeros(hinges.shape[0])
for kHinge in range(hinges.shape[0]):
  node0, node1, node2, node3 = hinges[kHinge, :]
  thetaBar[kHinge] = getTheta(q0[3 * node0:3 * node0 + 3], q0[3 * node1:3 * node1 + 3],
                              q0[3 * node2:3 * node2 + 3], q0[3 * node3:3 * node3 + 3])

# Set up boundary conditions
# We define the fixed and free DOFs. If the simulation asks that the boundary conditions vary with time (e.g., someone is holding a shell and all on a sudden drops it), we have to define it later within the time stepping loop.

# Fixed and free DOFs
fixedIndex = np.arange(0, 9) # x0, x1, x2 are fixed
freeIndex = np.arange(9, ndof) # Only x3 is free

# Time stepping loop

Nsteps = round(totalTime / dt)  # Number of time steps
ctime = 0  # Current time
endZ = np.zeros(Nsteps)  # Store z-coordinate of last node

for timeStep in range(Nsteps):
  print('Current time=%f' % ctime)

  # Call the objective function to update q and u
  qGuess = q0.copy()  # Start with the current guess
  q, u = objfun(qGuess, q0, u, freeIndex, dt, tol,
                massVector, mMat,
                ks, lk, edges,
                kb, thetaBar, hinges,
                Fg, visc)

  ctime += dt  # Increment time

  # Update q0 with the new q
  q0 = q.copy()

  # Store the z-coordinate of the last node
  endZ[timeStep] = q[-1]

  # Every plotEvery time steps, plot the shell
  if timeStep % plotEvery == 0:
    plotShell(q, ctime)

# Visualization after the loop
plotShell(q, ctime)  # Final shape of the shell (t = totalTime)
plt.savefig('shell_final_shape.png')

plt.figure(2)
time_array = np.arange(1, Nsteps + 1) * dt
plt.plot(time_array, endZ, 'ro-')
plt.box(True)
plt.xlabel('Time, t [sec]')
plt.ylabel('z-coord of last node, $\\delta_z$ [m]')
plt.savefig('endZ_vs_time.png')
plt.pause(0.01)  # Display figures 1-2

plt.show()
