import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'der'))

from der_functions import (computeTangent, assignOrthonormalAdaptedFrame, computeSpaceParallel,
                           computeMaterialDirectors, getKappa, plotrod_simple,
                           computeTimeParallel, getRefTwist, getFs, getFb, getFt)


# Objective function
def objfun(qGuess, q0, u, a1, a2, freeIndex,
                          dt, tol, refTwist, massVector, mMat,
                          EA, refLen,
                          EI, GJ, voronoiRefLen,
                          kappaBar, twistBar,
                          Fg):
    q = qGuess # Initial guess for Newton Raphson
    iter = 0
    error = 10.0 * tol
    while error > tol:
        # Frames
        a1Iterate, a2Iterate = computeTimeParallel(a1, q0, q) # Parallel transport a1,a2 from q0 to q
        tangent = computeTangent(q)
        refTwist_iterate = getRefTwist(a1Iterate, tangent, refTwist)
        theta = q[3::4]
        m1, m2 = computeMaterialDirectors(a1Iterate, a2Iterate, theta)

        # Force and Jacobian
        Fb, Jb = getFb(q, m1, m2, kappaBar, EI, voronoiRefLen) # Return bending force vector of size ndof and Hessian
        Ft, Jt = getFt(q, refTwist_iterate, twistBar, GJ, voronoiRefLen) # Twisting
        Fs, Js = getFs(q, EA, refLen)

        elasticForces = Fb + Ft + Fs
        elasticJ = Jb + Jt + Js

        f = np.dot( mMat/ dt, (q-q0)/dt - u ) - elasticForces - Fg
        J = mMat / dt**2 - elasticJ

        # Update/Correction
        f_free = f[freeIndex]
        J_free = J[np.ix_(freeIndex, freeIndex)]
        dq_free = np.linalg.solve(J_free, f_free)
        q[freeIndex] = q[freeIndex] - dq_free
        error = np.linalg.norm(f_free)
        iter += 1
    print(f'Iter ={iter}, error={error:.6f}')
    u = (q - q0) / dt
    a1 = a1Iterate
    a2 = a2Iterate
    return q, u, a1, a2

# Main DER

# Geometry
nv = 50 # Nodes
ne = nv - 1 # Edges
ndof = 4 * nv - 1 # Total number of DOF

RodLength = 0.2 # meter
natR = 0.02 # m
r0 = 0.001 # m - cross sectional radius

# Create nodes matrix of size nv x 3
nodes = np.zeros((nv, 3))
if natR == 0:
    for c in range(nv):
        nodes[c, 0] = c * RodLength / (nv - 1) # x-coordinate of c-th node
        nodes[c,1] = 0.0 # y-coordinate of c-th node
        nodes[c,2] = 0.0 # z-coordinate of c-th node
else:
    dTheta = (RodLength / natR) / ne
    for c in range(nv):
        nodes[c, 0] = natR * np.cos(c * dTheta) # x-coordinate of c-th node
        nodes[c, 1] = natR * np.sin(c * dTheta) # y-coordinate of c-th node
        nodes[c, 2] = 0.0 # z-coordinate of c-th node

# Stiffness parameters
Y = 10e6 # Pa
nu = 0.5 # Poisson's ratio
G = Y / (2 * (1 + nu)) # Shear modulus
EI = Y * np.pi * r0**4 / 4 # Bending stiffness
GJ = G * np.pi * r0**4 / 2 # Shearing stiffness
EA = Y * np.pi * r0**2 # Stretching stiffness

# Time parameters
totalTime = 5 # seconds
dt = 0.01 # seconds
plotEvery = 10 # Plot every plotEvery time steps
tol = EI / RodLength**2 * 1e-3 # Tolerance

# Mass
rho = 1000 # kg/m^3
totalM = (np.pi * r0**2 * RodLength) * rho # Total mass of the rod
dm = totalM / ne # Mass per edge
massVector = np.zeros(ndof)
for c in range(nv): # Loop over nodes
    ind= [4*c, 4*c+1, 4*c+2]
    if c == 0 or c == nv-1: # First or last node
        massVector[ind] = dm/2.0
    else:
        massVector[ind] = dm
for c in range(ne): # Loop over edges (angular mass)
    ind = 4 * c + 3
    massVector[ind] = 1.0/2.0 * dm * r0**2.0
mMat = np.diag(massVector)

# External force - gravity
g = np.array([0, 0, -9.8])
Fg = np.zeros(ndof)
for c in range(nv):
    ind = [4*c, 4*c+1, 4*c+2]
    Fg[ind] = massVector[ind] * g

# Initial DOF vector
q0 = np.zeros(ndof) # Position
for c in range(nv):
    ind = [4*c, 4*c+1, 4*c+2] # Location of the c-th node
    q0[ind] = nodes[c,:] # Place the c-th node
u = np.zeros(ndof) # Velocity

plotrod_simple(q0, 0)

# Reference length for each edge (edge length)
refLen = np.zeros(ne)
for c in range(ne): # loop over edges
    dx = nodes[c+1,:] - nodes[c,:] # Vector between two subsequent nodes
    refLen[c] = np.linalg.norm(dx)

# Voronoi length
voronoiRefLen = np.zeros(nv)
for c in range(nv): # loop over nodes
    if c == 0:
        voronoiRefLen[c] = 0.5 * refLen[c] # half of edge after
    elif c == nv-1:
        voronoiRefLen[c] = 0.5 * refLen[c-1] # half of edge before
    else:
        voronoiRefLen[c] = 0.5 * (refLen[c-1] + refLen[c])

# Reference frame
tangent = computeTangent(q0) # Return an array of size ne x 3, given the DOF vector
a1_first, a2_first = assignOrthonormalAdaptedFrame(tangent[0,:]) # Get orthonormal adapted frame on the first edge
a1, a2 = computeSpaceParallel(a1_first, q0) # Given the first frame and the DOF vector, return the frame on each edge
# a1 and a2 have size ne x 3 (One vector on each edge)

# Material frame: Angle from reference frame to material frame is theta for each edge
theta = q0[3::4] # Every fourth number starting with index 3 (Should be zero here)
m1, m2 = computeMaterialDirectors(a1, a2, theta)

# Reference twist
refTwist = np.zeros(nv)

# Natural curvature
kappaBar = getKappa(q0, m1, m2) # Returns an array of size nv x 2

# Natural twist
twistBar = np.zeros(nv)

# Boundary conditions
fixedIndex = np.arange(0,7)
freeIndex = np.arange(7, ndof)

# Time stepping loop
Nsteps = round(totalTime / dt)
ctime = 0 # Current time
endZ = np.zeros(Nsteps) # Container to store z-coordinate of last node

for timeStep in range(Nsteps):
    print('Current time = %f' % ctime)

    # Call simulator / objfun
    qGuess = q0.copy()
    q, u, a1, a2 = objfun(qGuess, q0, u, a1, a2, freeIndex,
                          dt, tol, refTwist, massVector, mMat,
                          EA, refLen,
                          EI, GJ, voronoiRefLen,
                          kappaBar, twistBar,
                          Fg)

    ctime += dt # Update time

    # Update old position
    q0 = q.copy()
    # Store z-coordinate of the last node
    endZ[timeStep] = q[-1]

    if timeStep % plotEvery == 0:
        plotrod_simple(q, ctime)

# Plotting
plt.figure(2)
time_array = np.arange(1, Nsteps+1) * dt
plt.plot(time_array, endZ, 'ro-')
plt.xlabel('Time, t [sec]')
plt.ylabel('z-coord of the last node, $\\delta_z$ [m]')
plt.pause(0.01)
plt.show()