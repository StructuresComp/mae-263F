"""
Discrete Elastic Rods (DER): a naturally curved rod, clamped at one end, deforming under gravity.

Copyright M. Khalid Jawed (khalidjm@seas.ucla.edu). License: CC BY-NC
You should use this code at your own risk.

Script version of modular_main_discrete_elastic_rods.py (originally a Colab notebook).
The helper functions are in der_functions.py (this folder); the gradients and Hessians
of the elastic energies are in Materials/appendix.

Run from anywhere, e.g.:  python src/der/main_discrete_elastic_rods.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from der_functions import (computeTangent, assignOrthonormalAdaptedFrame, computeSpaceParallel,
                           computeMaterialDirectors, getKappa, plotrod_simple, objfun)


# ---------------------------------------------------------------------------
# Main DER
#
# Degrees of freedom and nodes: the rod is divided into nv nodes, i.e., ne = nv - 1
# edges, and the DOF vector has size ndof = 4*nv - 1:
# q = [x_1, y_1, z_1, theta^1, x_2, y_2, z_2, theta^2, ..., x_nv, y_nv, z_nv].
# ---------------------------------------------------------------------------

# Inputs
nv = 50 # number of nodes (N = 50 and dt = 0.01 reproduce Figure 8.1 of the notes)
ne = nv - 1  # number of edges
ndof = 3 * nv + ne  # number of DOF = 4*nv - 1

# Geometric parameters
RodLength = 0.2
natR = 0.02  # Natural radius
r0 = 0.001  # Cross-sectional radius

# Matrix containing the nodes
nodes = np.zeros((nv, 3))
if natR == 0:
  for c in range(nv):
    nodes[c, 0] = RodLength * c / (nv-1)
    nodes[c, 1] = 0
    nodes[c, 2] = 0
else:
  dTheta = (RodLength / natR) * (1 / ne)
  for c in range(nv):
    nodes[c, 0] = natR * np.cos(c * dTheta)
    nodes[c, 1] = natR * np.sin(c * dTheta)
    nodes[c, 2] = 0

# Elastic stiffness parameters
# We will compute the elastic stiffness values (bending, twisting, and stretching), i.e., EI, GJ, and EA, based on the material properties and cross-sectional radius. We are assuming a rod with circular cross-section.

# Material parameters
Y = 10e6  # Young's Modulus
nu = 0.5  # Poisson's ratio
G = Y / (2 * (1 + nu))  # Shear modulus

# Stiffness variables
EI = Y * np.pi * r0**4 / 4  # Bending stiffness
GJ = G * np.pi * r0**4 / 2  # Shearing stiffness
EA = Y * np.pi * r0**2  # Stretching stiffness

# Time parameters

# Time parameters
totalTime = 5  # Simulation time
dt = 0.01  # Time step size
plotEvery = 10  # Plot the rod every plotEvery time steps

# Tolerance
tol = EI / RodLength**2 * 1e-3

# Mass vector and matrix

# Mass
rho = 1000  # Density
totalM = np.pi * r0**2 * RodLength * rho
dm = totalM / ne  # mass per edge
massVector = np.zeros(ndof)
for c in range(nv):  # Loop over nodes
    ind = [4*c, 4*c+1, 4*c+2]
    if c == 0:
        massVector[ind] = dm/2
    elif c == nv - 1:
        massVector[ind] = dm/2
    else:
        massVector[ind] = dm

for c in range(ne):  # Loop over edges
    massVector[4*c+3] = 1/2 * dm * r0**2
mMat = np.diag(massVector)  # ndof x ndof sized mass matrix

# External Force
# Since this does not change with time, we define it here instead of within the time loop.

# Gravity
g = np.array([0, 0, -9.81])
Fg = np.zeros(ndof)
for c in range(nv):  # loop over the nodes
    ind = [4*c, 4*c+1, 4*c+2]
    Fg[ind] = massVector[ind] * g

# Initial DOF vector

# Initial DOF vector
q0 = np.zeros(ndof)
for c in range(nv):
    ind = [4*c, 4*c+1, 4*c+2]
    q0[ind] = nodes[c, :]

u = np.zeros_like(q0)  # Initialize velocity

plotrod_simple(q0, 0)

# Compute the reference (undeformed) length of each edge and the Voronoi length (undeformed) associated with each node

# Reference length (edge length)
refLen = np.zeros(ne)
for c in range(ne):  # loop over the edges
    dx = nodes[c+1, :] - nodes[c, :]
    refLen[c] = np.linalg.norm(dx)

# Voronoi length (Length associated with each node)
voronoiRefLen = np.zeros(nv)
for c in range(nv):  # loop over the nodes
    if c == 0:
        voronoiRefLen[c] = 0.5 * refLen[c]
    elif c == nv - 1:
        voronoiRefLen[c] = 0.5 * refLen[c-1]
    else:
        voronoiRefLen[c] = 0.5 * (refLen[c-1] + refLen[c])

# Set up reference frame (a1, a2) and material frame (m1, m2)

# Reference frame (Space parallel transport at t=0)
tangent = computeTangent(q0)
a1_first, a2_first = assignOrthonormalAdaptedFrame(tangent[0, :])  # Arbitrary adapted frame on the first edge
a1, a2 = computeSpaceParallel(a1_first, q0)

# Material frame
theta = q0[3::4]  # Extract theta
m1, m2 = computeMaterialDirectors(a1, a2, theta)

# Compute natural curvature, natural twist, and reference twist
# Typically, reference twist and natural twist are both zeros. However, it is not always the case. Natural curvature is zero only if the rod is straight.

# Reference twist
refTwist = np.zeros(nv)

# Natural curvature
kappaBar = getKappa(q0, m1, m2)

# Natural twist
twistBar = np.zeros(nv)

# Set up boundary conditions
# We define the fixed and free DOFs. If the simulation asks that the boundary conditions vary with time (e.g., someone is holding a rod and all on a sudden drops it), we have to define it later within the time stepping loop.

# Fixed and free DOFs
fixedIndex = np.arange(0, 7)
freeIndex = np.arange(7, ndof)

# Time stepping loop

Nsteps = round(totalTime / dt)  # Number of time steps
ctime = 0  # Current time
endZ = np.zeros(Nsteps)  # Store z-coordinate of last node

for timeStep in range(Nsteps):
  print('Current time=%f' % ctime)

  # Call the objective function to update q, u, a1, and a2
  qGuess = q0.copy()  # Start with the current guess
  q, u, a1, a2 = objfun(qGuess, q0, u, a1, a2, freeIndex, dt, tol, refTwist,
                        massVector, mMat,
                        EA, refLen,
                        EI, GJ, voronoiRefLen, kappaBar, twistBar,
                        Fg)

  ctime += dt  # Increment time

  # Update q0 with the new q
  q0 = q.copy()

  # Store the z-coordinate of the last node
  endZ[timeStep] = q[-1]

  # Every plotEvery time steps, plot the rod
  if timeStep % plotEvery == 0:
    # theta = q[3::4]  # Extract the theta values
    # m1, m2 = computeMaterialDirectors(a1, a2, theta)
    # plotrod(q, a1, a2, m1, m2, ctime)
    plotrod_simple(q, ctime)

# Visualization after the loop
plotrod_simple(q, ctime)  # Final shape of the rod (t = totalTime)
plt.savefig('rod_final_shape.png')

plt.figure(2)
time_array = np.arange(1, Nsteps + 1) * dt
plt.plot(time_array, endZ, 'ro-')
plt.box(True)
plt.xlabel('Time, t [sec]')
plt.ylabel('z-coord of last node, $\\delta_z$ [m]')
plt.savefig('endZ_vs_time.png')
plt.pause(0.01)  # Display figures 1-2

plt.show()
