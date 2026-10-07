import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'appendix'))

import numpy as np
import matplotlib.pyplot as plt

from gradEb import gradEb
from hessEb import hessEb
from gradEs import gradEs
from hessEs import hessEs

# Elastic forces: assemble over all bending and stretching springs
def getFb(q, EI, deltaL):
    """
    q = [x_0, y_0, x_1, y_1, ..., x_{nv-1}, y_{nv-1}]^T
    EI = Bending stiffness [Nm^2]
    deltaL = Rest length of edges
    """

    ndof = q.size
    nv = ndof // 2 # Number of nodes / vertices

    # Initialize bending force and Jacobian
    Fb = np.zeros(ndof)
    Jb = np.zeros((ndof, ndof))


    # loop over all nodes starting from the second and ending at the second to last node
    for k in range(1, nv - 1):
        idx = np.arange(2*k - 2, 2*k + 4) # Indices for the current bending spring
        Fb[idx] -= gradEb(*q[idx], 0, deltaL, EI) # Compute bending force (negative gradient of bending energy)
        Jb[np.ix_(idx, idx)] -= hessEb(*q[idx], 0, deltaL, EI) # Compute bending Jacobian (negative Hessian of bending energy)

    return Fb, Jb

def getFs(q, EA, deltaL):
    """
    q = [x_0, y_0, x_1, y_1, ..., x_{nv-1}, y_{nv-1}]^T
    EA = Stretching stiffness [N]
    deltaL = Rest length of edges
    """

    ndof = q.size
    nv = ndof // 2 # Number of nodes / vertices

    # Initialize stretching force and Jacobian
    Fs = np.zeros(ndof)
    Js = np.zeros((ndof, ndof))

    # loop over all nodes starting from the first and ending at the second to last node
    for k in range(0, nv - 1):
        idx = np.arange(2*k, 2*k + 4) # Indices for the current stretching spring
        Fs[idx] -= gradEs(*q[idx], deltaL, EA) # Compute stretching force (negative gradient of stretching energy)
        Js[np.ix_(idx, idx)] -= hessEs(*q[idx], deltaL, EA) # Compute stretching Jacobian (negative Hessian of stretching energy)

    return Fs, Js

# Implicit Euler step using Newton's method
def objfun(q_guess, q_old, u_old, dt, tol, maximum_iter,
           m, mMat, EI, EA, W, C, deltaL):
    """
    Solve for q_new using Newton's method.
    Returns (q_new, flag); flag = -1 if Newton did not converge.
    """
    q_new = q_guess.copy() # Guess solution
    iter_count = 0 # Number of iterations; intialize to 0
    error = tol * 10 # Initialize error to be larger than tolerance

    while error > tol:
        # (1) Inertia, (2) Elastic, and (3) External (damping and buoyancy)
        # Item (2)
        Fb, Jb = getFb(q_new, EI, deltaL) # Get bending force and Jacobian
        Fs, Js = getFs(q_new, EA, deltaL) # Get stretching force and Jacobian
        Felastic = Fb + Fs # Total elastic force
        Jelastic = Jb + Js # Total elastic Jacobian

        # Item (1)
        Fi = m * (q_new - q_old) / dt**2  - m * u_old / dt # Inertia force
        Ji = mMat / dt**2 # Inertia Jacobian

        # Item (3)
        # Viscous force
        Fv = - C @ (q_new - q_old) / dt # Viscous force
        Jv = - C / dt # Viscous Jacobian

        # Residual and Jacobian
        f = Fi - Felastic - Fv - W # Residual
        J = Ji - Jelastic - Jv # Jacobian

        # Newton's update
        q_new = q_new - np.linalg.solve(J, f) # Update guess solution
        error = np.linalg.norm(f) # Compute error

        iter_count += 1 # Increment iteration count
        if iter_count > maximum_iter: # Check for convergence
            return q_new, -1 # Return with flag -1 if not converged
    return q_new, 1 # Return with flag 1 if converged

# Simulation loop
def simulate(nv, dt, totalTime, plotShapes=False):
    """
    Simulate the falling beam with nv nodes, time step dt, and total simulation time totalTime.
    If plotShapes is True, plot the shape of the beam at specified time intervals.
    Return the time array, midpoint y-coordinates + velocities
    """
    ndof = 2 * nv # Number of degrees of freedom

    # PART 1
    # Geometry
    RodLength = 0.10 # m
    deltaL = RodLength / (nv - 1) # Rest length of edges
    midNode = (nv - 1) // 2 # Index of the middle node
    R = np.full(nv, deltaL / 10)
    R[midNode] = 0.025 # Sphere radii: R_mid = 0.025 at the middle node, deltaL/10 elsewhere

    # Material
    rho_metal = 7000 # kg/m^3
    rho_fluid = 1000 # kg/m^3
    r0 = 1e-3 # m, radius of the rod
    Y = 1e9 # Pa, Young's modulus
    visc = 1000.0 # Pa.s, fluid viscosity
    # Stiffness quantities
    EI = Y * np.pi * r0**4 / 4 # Bending stiffness [Nm^2]
    EA = Y * np.pi * r0**2 # Stretching stiffness [N]

    # Time stepping
    maximum_iter = 100 # Maximum number of Newton iterations
    tol = EI / RodLength**2 * 1e-6 # Tolerance for Newton's method

    # PART 2: Some simple calculation
    # Mass + Weight
    m = np.repeat( 4.0 / 3.0 * np.pi * R**3 * rho_metal, 2) # Mass of each node (2 DOF per node)
    mMat = np.diag(m) # Mass matrix
    W = np.zeros(ndof) # Weight vector
    W[1::2] = - 4.0 / 3.0 * np.pi * R**3 * (rho_metal - rho_fluid)  * 9.81 # Weight in y-direction

    # Damping matrix
    Cvector = np.repeat(6.0 * np.pi * visc * R, 2)
    C = np.diag(Cvector)

    # PART 3: Initial conditions
    q0 = np.zeros(ndof) # Initial position vector
    q0[0::2] = np.arange(nv) * deltaL # x-coordinates of nodes
    u = np.zeros(ndof) # Initial velocity vector

    # PART 4: Time stepping loop
    Nsteps = round(totalTime / dt) # Number of time steps
    t = np.arange(Nsteps + 1) * dt # Time array
    y_mid = np.zeros(Nsteps + 1) # y-coordinate of the middle node
    v_mid = np.zeros(Nsteps + 1) # y-velocity of the middle

    for timeStep in range(1, Nsteps + 1):
        q, flag = objfun(q0, q0, u, dt, tol, maximum_iter,
                         m, mMat, EI, EA, W, C, deltaL) # Solve for new position
        if flag == -1:
            raise RuntimeError(f'Newton did not converge at time step {timeStep}.')

        u = (q - q0) / dt # Update velocity

        q0 = q.copy() # Update position

        y_mid[timeStep] = q[2*midNode + 1] # y-coordinate of the middle node
        v_mid[timeStep] = u[2*midNode + 1] # y-velocity of the middle node

        # Plot
        if plotShapes and timeStep % 10 == 0:
            plt.figure(1)
            plt.clf()
            plt.plot(q[0::2], q[1::2], 'o-')
            plt.title(f'Time = {t[timeStep]:.3f} s')
            plt.xlabel('x [m]')
            plt.ylabel('y [m]')
            plt.axis('equal')
            plt.draw()
            plt.pause(0.01)
    return t, y_mid, v_mid

# Main
if __name__ == "__main__":
    nv = 21 # Number of nodes
    dt = 0.01 # Time step
    totalTime = 50.0 # Total simulation time
    plotShapes = True # Whether to plot the shapes

    # Call the simulation function
    t, y_mid, v_mid = simulate(nv, dt, totalTime, plotShapes)

    # Plot position and velocity of the middle node over time
    plt.figure(2)
    plt.plot(t, y_mid)
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('Middle Node Position Over Time')
    plt.show()

    # Plot position and velocity of the middle node over time
    plt.figure(3)
    plt.plot(t, v_mid)
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title('Middle Node Velocity Over Time')
    plt.show()
