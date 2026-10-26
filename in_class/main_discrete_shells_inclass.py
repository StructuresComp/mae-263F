"""
Discrete Elastic Shells: simple example with four nodes (two triangles sharing one edge).
In class exercise.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Make ../appendix importable regardless of the current working directory
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'appendix'))

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
def objfun( qGuess, q0, u, freeIndex, dt, tol,
                  massVector, mMat,
                  ks, lk, edges,
                  kb, thetaBar, hinges,
                  Fg):
    q = qGuess # Initial guess
    ndof = len(q)
    iter = 0
    error = 10.0 * tol
    
    while error > tol:
        # Bending
        Fb = np.zeros(ndof)
        Jb = np.zeros((ndof, ndof))
        for kHinge in range(hinges.shape[0]): # Loop over all the hinges
            node0 = hinges[kHinge, 0] # Node number for the first node
            node1 = hinges[kHinge, 1] # Node number for the second node
            node2 = hinges[kHinge, 2] # Node number for the third node
            node3 = hinges[kHinge, 3] # Node number for the fourth node
            x0 = q [ 3 * node0: 3*node0 + 3 ]
            x1 = q [ 3 * node1: 3*node1 + 3 ]
            x2 = q [ 3 * node2: 3*node2 + 3 ]
            x3 = q [ 3 * node3: 3*node3 + 3 ]
            ind = [3*node0, 3*node0 + 1, 3*node0 + 2,
                   3*node1, 3*node1 + 1, 3*node1 + 2,
                   3*node2, 3*node2 + 1, 3*node2 + 2,
                   3*node3, 3*node3 + 1, 3*node3 + 2 ]
            dF, dJ = gradEb_hessEb_Shell( x0, x1, x2, x3, thetaBar, kb)
            Fb[ind] = Fb[ind] - dF
            Jb[np.ix_(ind, ind)] = Jb[np.ix_(ind, ind)] - dJ

        # Stretching
        Fs = np.zeros(ndof)
        Js = np.zeros((ndof, ndof))
        for kEdges in range(edges.shape[0]): # Loop over all the edges
            node0 = edges[kEdges, 0] # Node number for the first node
            node1 = edges[kEdges, 1] # Node number for the second node
            x0 = q [ 3 * node0: 3*node0 + 3 ]
            x1 = q [ 3 * node1: 3*node1 + 3 ]
            ind = [3*node0, 3*node0 + 1, 3*node0 + 2,
                   3*node1, 3*node1 + 1, 3*node1 + 2]
            dF, dJ = gradEs_hessEs_Shell( x0, x1, lk[kEdge], ks[kEdge])
            Fs[ind] = Fs[ind] - dF
            Js[np.ix_(ind, ind)] = Js[np.ix_(ind, ind)] - dJ
        
        # Elastic
        F_elastic = Fb + Fs
        J_elastic = Jb + Js

        # Assemble my residual/force vector and Jacobian
        f = np.dot( mMat / dt, (q-q0)/dt - u ) - F_elastic - Fg
        J = mMat / dt ** 2.0 - J_elastic

        f_free = f[freeIndex] # Extract the free part
        J_free = J[np.ix_(freeIndex, freeIndex)]
        dq_free = np.linalg.solve(J_free, f_free) # Correction
        q[freeIndex] = q[freeIndex] - dq_free # Update guess using Newton
        error = np.linalg.norm(f_free)
        iter += 1

    u = (q - q0) / dt # Velocity
    print(f'iter = {iter}, error={error:0.6f}')
    return q, u



# ---------------------------------------------------------------------------
# Main Discrete Shells
#
# ---------------------------------------------------------------------------

# Input nodes (mesh)
x0 = np.array([0.0, 0.0, 0.0])
x1 = np.array([0.01, 0.0, 0.0])
x2 = np.array([0.005, 0.01, 0.0])
x3 = np.array([0.005, -0.01, 0.0])

q0 = np.concatenate((x0, x1, x2, x3))
plotShell(q0, 0)

nv = 4 # Number of nodes or vertices
ndof = 3 * nv

# Create edges and hinges
edges = np.array([ (0,1), (1,2), (0,2), (0,3), (1,3) ]) # Map from edges to nodes
hinges = np.array([ (0, 1, 2, 3) ])

# Stiffness
Y = 1.0e6 # Young's Modulus
h = 0.001 # thickness
# (1) Bending stiffness: kb
kb = 2.0 / np.sqrt(3.0) * Y * h**3.0 / 12.0
# (2) Stretching stiffness: ks
lk= np.zeros( edges.shape[0])
ks = np.zeros( edges.shape[0])
for kEdge in range(edges.shape[0]):
    node0 = edges[kEdge, 0]
    node1 = edges[kEdge, 1]
    ind = np.arange(3*node0, 3*node0 + 3)
    x0 = q0[ind]
    ind = np.arange(3*node1, 3*node1 + 3)
    x1 = q0[ind]
    lk[kEdge] = np.linalg.norm( x1 - x0 )
    ks[kEdge] = np.sqrt(3.0) / 2.0 * Y * h * (lk[kEdge])**2.0

# Time parameters
totalTime = 5.0 # seconds
dt = 0.01 # Time step size
plotEvery = 10 # Plot every few iterations

# Tolerance
tol = kb / (0.01) * 1.0e-3

# Mass
rho = 10 # density
totalM = 0.01 # total mass
dm = totalM / nv # mass per node
massVector = np.zeros(ndof)
for c in range(nv): # loop over the nodes
    ind = np.arange(3*c, 3*c + 3)
    massVector[ind] = dm
mMat = np.diag(massVector)

# External force: Gravity
g = np.array([0, 0, -9.81])
Fg = np.zeros(ndof)
for c in range(nv): # Loop over each node
    ind = np.arange(3*c, 3*c + 3)
    Fg[ind] = massVector[ind] * g

# Initial DOF
u = np.zeros_like(q0) # Velocity -- initially all zeros

# Boundary conditions
fixedIndex = np.arange(0, 9) # First three nodes are fixed
freeIndex = np.arange(9, ndof) # All the other nodes (i.e., fourth node) is free

# Time stepping parameters
Nsteps = round(totalTime / dt) # Number of steps
ctime = 0.0 # Current time
endZ = np.zeros(Nsteps) # Container to store the z-coordinate of the free node
thetaBar = 0 # Undeformed theta = 0 for a plate
for timeStep in range(Nsteps):
    print('Current time = %f' % ctime)

    qGuess = q0.copy() # Initial guess
    q, u = objfun( qGuess, q0, u, freeIndex, dt, tol,
                  massVector, mMat,
                  ks, lk, edges,
                  kb, thetaBar, hinges,
                  Fg)
    ctime += dt # Update time
    q0 = q.copy() # Update old position
    endZ[timeStep] = q[-1] # z-coordinate of the last node

    if timeStep % plotEvery == 0:
        plotShell(q, ctime)

# Visualization
plotShell(q, ctime)

# Plot the deflection (endZ)
plt.figure(2)
time_array = np.arange(1, Nsteps+1) * dt
plt.plot(time_array, endZ, 'ro-')
plt.xlabel('Time, t [sec]')
plt.ylabel('z-coordinate of last node, $\\delta_z$ [m]')
plt.pause(0.01)
plt.show()
