"""
Mesh a rectangular plate with triangles, using only numpy and matplotlib.

Script version of Lecture 14 - Mesh_Plate.ipynb (originally a Colab notebook that
used pygmsh/Gmsh). Here, the nodes are placed on a triangular lattice and
matplotlib.tri (Delaunay triangulation) connects them into triangles. Away from the
left and right sides, the triangles are (nearly) equilateral.

Run from anywhere, e.g.:  python src/shells/mesh_plate.py
Or import it:             from mesh_plate import mesh_plate, plot_mesh
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


# ---------------------------------------------------------------------------
# Create and plot the mesh
# ---------------------------------------------------------------------------

def mesh_plate(l, w, max_mesh_size):
    """
    Triangle mesh of the rectangle [0, l] x [0, w].

    Parameters:
    - l: Length of the rectangle (along x).
    - w: Width of the rectangle (along y).
    - max_mesh_size: Maximum edge length of the triangles.

    Returns:
    - nodes: (nv, 2) array; row c is the (x, y) position of node c.
    - elements: (nt, 3) integer array; row k contains the three node indices of
      triangle k, ordered counterclockwise (i.e., the normal points along +z).
    """
    # Number of intervals along x and number of rows along y.
    # Row spacing is sqrt(3)/2 of the node spacing, as in an equilateral lattice.
    # Rounding up guarantees that every edge is at most max_mesh_size long.
    nx = max(1, int(np.ceil(l / max_mesh_size - 1e-9)))
    ny = max(1, int(np.ceil(w / (max_mesh_size * np.sqrt(3) / 2) - 1e-9)))
    dx = l / nx  # Node spacing along x
    dy = w / ny  # Row spacing along y

    # Place the nodes row by row (bottom to top). Odd rows are shifted by dx/2 and
    # also get one node on each side, so that the sides x = 0 and x = l are straight.
    nodes = []
    for j in range(ny + 1):
        y = j * dy
        if j % 2 == 0:
            xs = np.arange(nx + 1) * dx
        else:
            xs = np.concatenate(([0.0], (np.arange(nx) + 0.5) * dx, [l]))
        for x in xs:
            nodes.append([x, y])
    nodes = np.array(nodes)

    # Connect the nodes into triangles (Delaunay triangulation)
    triangulation = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
    elements = triangulation.triangles.astype(int)

    # Make sure every triangle is ordered counterclockwise (normal along +z)
    for k in range(elements.shape[0]):
        p0, p1, p2 = nodes[elements[k, 0]], nodes[elements[k, 1]], nodes[elements[k, 2]]
        signedArea = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if signedArea < 0:
            elements[k, [1, 2]] = elements[k, [2, 1]]

    return nodes, elements


def plot_mesh(nodes, elements):
    plt.figure(figsize=(8, 6))

    # Plot each triangle element
    for element in elements:
        triangle = nodes[element, :]
        # Draw lines for each edge of the triangle
        for i in range(3):
            x_values = triangle[[i, (i + 1) % 3], 0]  # Connect vertices i and (i+1)%3
            y_values = triangle[[i, (i + 1) % 3], 1]
            plt.plot(x_values, y_values, color='black')  # Draw triangle edge

    # Plot nodes
    plt.scatter(nodes[:, 0], nodes[:, 1], color='red', label='Nodes', zorder=5)

    # Set plot properties
    plt.title('Mesh Visualization')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.pause(0.01)  # Display the figure without blocking


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # Define your parameters
    l = 1.0  # Length of the rectangle
    w = 0.5  # Width of the rectangle
    max_mesh_size = 0.1

    # Generate mesh
    nodes, elements = mesh_plate(l, w, max_mesh_size)
    print(f'Number of nodes = {nodes.shape[0]}, number of triangles = {elements.shape[0]}')

    # Plot the mesh
    plot_mesh(nodes, elements)

    plt.show()
