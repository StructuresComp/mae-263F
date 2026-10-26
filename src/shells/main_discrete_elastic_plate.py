"""
Discrete Elastic Plate: a cantilever plate, clamped at x = 0, deforming under gravity.
The tip deflection is compared with the Euler-Bernoulli beam solution.

Copyright M. Khalid Jawed (khalidjm@seas.ucla.edu). License: CC BY-NC
You should use this code at your own risk.

Python version of Main_Discrete_Elastic_Plate.m ("Plate Simulation_v1").
The mesh is from mesh_plate.py and the helper functions are in shell_functions.py
(this folder); the gradients and Hessians of the elastic energies
(gradEs_hessEs_Shell and gradEb_hessEb_Shell) are in the appendix folder (../../appendix).

Run from anywhere, e.g.:  python src/shells/main_discrete_elastic_plate.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mesh_plate import mesh_plate
from shell_functions import getEdgesAndHinges, getThetaBar, plotShell, objfun


# ---------------------------------------------------------------------------
# Main Discrete Elastic Plate
#
# Degrees of freedom and nodes: the plate is divided into nv nodes, and the
# DOF vector has size ndof = 3*nv: q = [x_0, y_0, z_0, x_1, y_1, z_1, ...].
# ---------------------------------------------------------------------------

# Geometric parameters
l = 0.1    # Length of the plate (along x)
w = 0.01   # Width of the plate (along y)
h = 0.002  # Thickness of the plate

# Create the mesh
# The stiffness values below assume equilateral triangles. With this mesh size,
# there are 4 rows of (nearly) equilateral triangles across the width.
max_mesh_size = w / (4 * np.sqrt(3) / 2)
nodes, elements = mesh_plate(l, w, max_mesh_size)
nv = nodes.shape[0]  # Number of nodes
ndof = 3 * nv  # Number of DOF

# Edges (stretching springs) and hinges (bending springs)
edges, hinges = getEdgesAndHinges(elements)
print(f'Number of nodes = {nv}, edges = {edges.shape[0]}, hinges = {hinges.shape[0]}')

# Elastic stiffness parameters

# Material parameters
Y = 1.0e8  # Young's Modulus (1e7 in the MATLAB code; see the note at the end of this file)
rho = 1000  # Density

# Stiffness variables
kb = 2.0 / np.sqrt(3.0) * Y * h**3.0 / 12.0  # Bending stiffness of each hinge (Newton-meter)

refLen = np.zeros(edges.shape[0])  # Reference (undeformed) length of each edge
for kEdge in range(edges.shape[0]):
    node0 = edges[kEdge, 0]
    node1 = edges[kEdge, 1]
    refLen[kEdge] = np.linalg.norm(nodes[node1, :] - nodes[node0, :])

# Stretching stiffness of each edge (Newton). gradEs_hessEs_Shell uses the energy
# Es = 1/2 * ks * (|e|/refLen - 1)^2 * refLen, so ks has units of force; this is
# the spring constant sqrt(3)/2 * Y * h (Newton/meter) times refLen.
ks = np.sqrt(3.0) / 2.0 * Y * h * refLen

# Time parameters
totalTime = 1.0  # Simulation time
dt = 0.01  # Time step size
plotEvery = 10  # Plot the plate every plotEvery time steps

# Newton's method parameters
tol = 1e-6  # Tolerance on the force residual (Newton)
maximum_iter = 100  # Maximum number of iterations

# Viscous damping (viscous force = - C * velocity)
visc = 1.0  # Viscosity
C = visc * np.mean(refLen)

# Mass vector and matrix

# Mass: the mass of each triangle (rho * h * area) is split equally among its three nodes
massNode = np.zeros(nv)
for element in elements:
    p0, p1, p2 = nodes[element[0]], nodes[element[1]], nodes[element[2]]
    area = 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
    massNode[element] += rho * h * area / 3
massVector = np.repeat(massNode, 3)  # Same mass for x, y, and z of each node
mMat = np.diag(massVector)  # ndof x ndof sized mass matrix

# External Force
# Since this does not change with time, we define it here instead of within the time loop.

# Gravity
g = np.array([0, 0, -9.81])
Fg = np.zeros(ndof)
for c in range(nv):  # loop over the nodes
    ind = [3*c, 3*c+1, 3*c+2]
    Fg[ind] = massVector[ind] * g

# Initial DOF vector (flat plate in the z = 0 plane)
q0 = np.zeros(ndof)
q0[0::3] = nodes[:, 0]
q0[1::3] = nodes[:, 1]

u = np.zeros_like(q0)  # Initialize velocity

# Natural (undeformed) hinge angles. These are zero for a flat plate.
thetaBar = getThetaBar(q0, hinges)

# Set up boundary conditions
# Clamped end: fix the nodes at x = 0 and the nodes connected to them by an edge
# (fixing two "rows" of nodes also fixes the slope, as in the MATLAB code).

isFixedNode = np.zeros(nv, dtype=bool)
for kEdge in range(edges.shape[0]):
    node0 = edges[kEdge, 0]
    node1 = edges[kEdge, 1]
    if nodes[node0, 0] == 0 or nodes[node1, 0] == 0:
        isFixedNode[node0] = True
        isFixedNode[node1] = True

fixedNodes = np.where(isFixedNode)[0]
freeNodes = np.where(~isFixedNode)[0]

# Fixed and free DOFs
fixedIndex = np.sort(np.concatenate((3*fixedNodes, 3*fixedNodes+1, 3*fixedNodes+2)))
freeIndex = np.sort(np.concatenate((3*freeNodes, 3*freeNodes+1, 3*freeNodes+2)))

# Nodes at the free end (x = l)
tipNodes = np.where(np.isclose(nodes[:, 0], l))[0]

plotShell(q0, edges, fixedNodes, 0)

# Time stepping loop

Nsteps = round(totalTime / dt)  # Number of time steps
ctime = 0  # Current time
tipZ = np.zeros(Nsteps)  # Store the z-coordinate of the free end (average over the tip nodes)

for timeStep in range(Nsteps):
  print('Current time=%f' % ctime)

  # Call the objective function to update q and u
  qGuess = q0.copy()  # Start with the current guess
  q, u, flag = objfun(qGuess, q0, u, freeIndex, dt, tol, maximum_iter,
                      massVector, mMat,
                      edges, refLen, ks,
                      hinges, thetaBar, kb,
                      Fg, C)
  if flag != 1:
    print('Newton\'s method did not converge. Stopping the simulation.')
    break

  ctime += dt  # Increment time

  # Update q0 with the new q
  q0 = q.copy()

  # Store the z-coordinate of the free end
  tipZ[timeStep] = np.mean(q[3*tipNodes+2])

  # Every plotEvery time steps, plot the plate
  if timeStep % plotEvery == 0:
    plotShell(q, edges, fixedNodes, ctime)

# Comparison with Euler-Bernoulli beam theory
# Cantilever under a uniform load (weight per unit length) qLoad: tip deflection = qLoad * l^4 / (8 * EI)
qLoad = rho * w * h * 9.81  # Weight per unit length
EI = Y * w * h**3 / 12  # Bending stiffness of the beam
deltaEB = qLoad * l**4 / (8 * EI)
deltaSim = - np.mean(q0[3*tipNodes+2])  # From the last converged time step
print(f'Tip deflection: simulation = {deltaSim:.6e} m, Euler-Bernoulli = {deltaEB:.6e} m, '
      f'ratio = {deltaSim / deltaEB:.4f}')

# Visualization after the loop
plotShell(q, edges, fixedNodes, ctime)  # Final shape of the plate (t = totalTime)
plt.savefig('plate_final_shape.png')

plt.figure(2)
time_array = np.arange(1, Nsteps + 1) * dt
plt.plot(time_array, tipZ, 'ro-', label='Simulation')
plt.plot(time_array, -deltaEB * np.ones(Nsteps), 'k--', label='Euler-Bernoulli (static)')
plt.box(True)
plt.legend()
plt.xlabel('Time, t [sec]')
plt.ylabel('z-coord of the free end, $\\delta_z$ [m]')
plt.savefig('tipZ_vs_time.png')
plt.pause(0.01)  # Display figures 1-2

plt.show()

# Note on Young's modulus: with Y = 1e7 (as in the MATLAB code), the tip deflection is
# about 37% of the length. Euler-Bernoulli (linear) beam theory is not valid for such a
# large deflection. With Y = 1e8, the deflection is about 4% of the length.
