"""
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



# ---------------------------------------------------------------------------
# Main Discrete Shells
#
# Degrees of freedom and nodes: the shell is divided into nv nodes, and the
# DOF vector has size ndof = 3*nv: q = [x_0, y_0, z_0, x_1, y_1, z_1, ...].
# ---------------------------------------------------------------------------

