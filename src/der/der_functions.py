"""
Functions for the Discrete Elastic Rods (DER) simulator.

Copyright M. Khalid Jawed (khalidjm@seas.ucla.edu). License: CC BY-NC
You should use this code at your own risk.

These functions were moved, unchanged, out of modular_main_discrete_elastic_rods.py
(originally a Colab notebook). The gradients and Hessians of the elastic energies,
computekappa, rotateAxisAngle, and crossMat are in the appendix folder
(Materials/appendix) that accompanies the course notes.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Make Materials/appendix importable regardless of the current working directory
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'appendix'))

from rotateAxisAngle import rotateAxisAngle
from computekappa import computekappa
from gradEs_hessEs import gradEs_hessEs
from gradEb_hessEb import gradEb_hessEb
from gradEt_hessEt import gradEt_hessEt_panetta as gradEt_hessEt


# ---------------------------------------------------------------------------
# Miscellaneous functions: signedAngle, parallel_transport
# ---------------------------------------------------------------------------

def signedAngle(u = None,v = None,n = None):
    # This function calculates the signed angle between two vectors, "u" and "v",
    # using an optional axis vector "n" to determine the direction of the angle.
    #
    # Parameters:
    #   u: numpy array-like, shape (3,), the first vector.
    #   v: numpy array-like, shape (3,), the second vector.
    #   n: numpy array-like, shape (3,), the axis vector that defines the plane
    #      in which the angle is measured. It determines the sign of the angle.
    #
    # Returns:
    #   angle: float, the signed angle (in radians) from vector "u" to vector "v".
    #          The angle is positive if the rotation from "u" to "v" follows
    #          the right-hand rule with respect to the axis "n", and negative otherwise.
    #
    # The function works by:
    # 1. Computing the cross product "w" of "u" and "v" to find the vector orthogonal
    #    to both "u" and "v".
    # 2. Calculating the angle between "u" and "v" using the arctan2 function, which
    #    returns the angle based on the norm of "w" (magnitude of the cross product)
    #    and the dot product of "u" and "v".
    # 3. Using the dot product of "n" and "w" to determine the sign of the angle.
    #    If this dot product is negative, the angle is adjusted to be negative.
    #
    # Example:
    #   signedAngle(np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
    #   This would return a positive angle (π/2 radians), as the rotation
    #   from the x-axis to the y-axis is counterclockwise when viewed along the z-axis.
    w = np.cross(u,v)
    angle = np.arctan2( np.linalg.norm(w), np.dot(u,v) )
    if (np.dot(n,w) < 0):
        angle = - angle

    return angle


def parallel_transport(u = None,t1 = None,t2 = None):

    # This function parallel transports a vector u from tangent t1 to t2
    # Input:
    # t1 - vector denoting the first tangent
    # t2 - vector denoting the second tangent
    # u - vector that needs to be parallel transported
    # Output:
    # d - vector after parallel transport

    b = np.cross(t1,t2)
    if (np.linalg.norm(b) == 0):
        d = u
    else:
        b = b / np.linalg.norm(b)
        # The following four lines may seem unnecessary but can sometimes help
        # with numerical stability
        b = b - np.dot(b,t1) * t1
        b = b / np.linalg.norm(b)
        b = b - np.dot(b,t2) * t2
        b = b / np.linalg.norm(b)
        n1 = np.cross(t1,b)
        n2 = np.cross(t2,b)
        d = np.dot(u,t1) * t2 + np.dot(u,n1) * n2 + np.dot(u,b) * b

    return d


# ---------------------------------------------------------------------------
# Tangent, adapted frame, material frame, and reference frame (space and time parallel transport)
# ---------------------------------------------------------------------------

def computeTangent(x = None):
    ne = int( (len(x) + 1) / 4 - 1 )
    tangent = np.zeros((ne,3))
    for c in np.arange(ne):
        dx = x[4*(c+1):4*(c+1)+3] - x[4*c:4*c+3]
        tangent[c,:] = dx / np.linalg.norm(dx)
    return tangent


def computeMaterialDirectors(d1 = None,d2 = None, theta0 = None):
    ne = len(theta0)
    m1 = np.zeros((ne,3))
    m2 = np.zeros((ne,3))
    for c in np.arange(ne):
        cs = np.cos(theta0[c])
        ss = np.sin(theta0[c])
        d1_l = d1[c,0:3]
        d2_l = d2[c,0:3]
        m1[c,0:3] = cs * d1_l + ss * d2_l
        m2[c,0:3] = - ss * d1_l + cs * d2_l

    return m1, m2


def computeTimeParallel(d1_old = None,x0 = None,x = None):
    ne = int( (len(x) + 1) / 4 - 1 )
    tangent = np.zeros((ne,3))
    for c in np.arange(ne):
        dt = x0[4*(c+1):4*(c+1)+3] - x0[4*c:4*c+3]
        dt = dt / np.linalg.norm(dt)
        tangent[c,:] = dt

    d1 = np.zeros((ne,3))
    d2 = np.zeros((ne,3))
    for c in np.arange(ne):
        edge = x[4*(c+1):4*(c+1)+3] - x[4*c:4*c+3]
        t = edge / np.linalg.norm(edge)
        t = np.transpose(t)
        d1_l = parallel_transport(d1_old[c,:], tangent[c,:], t)
        d1_l = d1_l - np.dot(d1_l,t) * t
        d1_l = d1_l / np.linalg.norm(d1_l)
        d1[c,:] = d1_l
        d2[c,:] = np.cross(t,d1_l)
    return d1,d2


def assignOrthonormalAdaptedFrame(t = None, randomize = False, rng = None):
    # This function assigns an orthonormal adapted frame (d1, d2, t) to an edge
    # with unit tangent "t" (Section 7.1 of the notes). It is used at t = 0 to
    # start the space-parallel reference frame on the first edge (Section 7.3):
    # choose any unit vector d1 orthogonal to t, and d2 = t x d1.
    #
    # Parameters:
    #   t: numpy array-like, shape (3,), the unit tangent of the edge.
    #   randomize: bool, optional (default False). If False, the frame is
    #              deterministic (same output every call). If True, d1 is rotated
    #              about t by a random angle in [0, 2*pi), so every call returns a
    #              different (but still orthonormal and adapted) frame.
    #   rng: numpy.random.Generator, optional. Random number generator used when
    #        randomize is True, e.g. np.random.default_rng(seed) for reproducible
    #        results. If None, a new generator is created.
    #
    # Returns:
    #   d1: numpy array, shape (3,), first director (unit vector, orthogonal to t).
    #   d2: numpy array, shape (3,), second director, d2 = t x d1.
    #
    # The function works by:
    # 1. Taking the cross product of t with an "arbitrary" vector, [0, 0, -1].
    #    The result is perpendicular to t.
    # 2. If t is (anti)parallel to the arbitrary vector, the cross product is
    #    (close to) zero; in that case, [0, 1, 0] is used instead.
    # 3. Normalizing d1. If randomize is True, rotating d1 about t by a random
    #    angle (rotateAxisAngle keeps d1 orthogonal to t and of unit length).
    # 4. Computing d2 = t x d1.
    #
    # Example:
    #   assignOrthonormalAdaptedFrame(np.array([1, 0, 0]))
    #   returns d1 = [0, 1, 0] and d2 = [0, 0, 1].
    d1 = np.cross(t, np.array([0, 0, -1]))  # perpendicular to both t and [0, 0, -1]
    if np.linalg.norm(d1) < 1e-6:  # t is (anti)parallel to [0, 0, -1]
        d1 = np.cross(t, np.array([0, 1, 0]))
    d1 = d1 / np.linalg.norm(d1)  # Normalize
    if randomize:
        if rng is None:
            rng = np.random.default_rng()
        d1 = rotateAxisAngle(d1, t, rng.uniform(0, 2 * np.pi))
    d2 = np.cross(t, d1)
    return d1, d2

def computeSpaceParallel(d1_first = None,x = None):
  ne = int( (len(x) + 1) / 4 - 1 )
  tangent = np.zeros((ne,3))
  for c in np.arange(ne):
    dt = x[4*(c+1):4*(c+1)+3] - x[4*c:4*c+3]
    dt = dt / np.linalg.norm(dt)
    tangent[c,:] = dt

  d1 = np.zeros((ne,3))
  d2 = np.zeros((ne,3))

  # First edge
  t0 = tangent[0,:]
  d1[0,:] = d1_first
  d2[0,:] = np.cross(t0, d1_first)

  for c in np.arange(1,ne):
    t = tangent[c,:]
    d1_first = parallel_transport(d1_first, t0, t)
    d1_first = d1_first - np.dot(d1_first, t) * t
    d1_first = d1_first / np.linalg.norm(d1_first)
    d1[c,:] = d1_first
    d2[c,:] = np.cross(t,d1_first)
    t0 = t

  return d1,d2


# ---------------------------------------------------------------------------
# Reference twist and curvature
# ---------------------------------------------------------------------------

def computeReferenceTwist(u1, u2, t1, t2, refTwist = None):
    # This function computes the reference twist angle between two vectors "u1" and "u2",
    # given two tangent directions "t1" and "t2", and an optional initial guess for the twist.
    # It adjusts the guess to align "u1" with "u2" when parallel transported along "t1" and "t2".
    #
    # Parameters:
    #   u1: numpy array-like, shape (3,), the initial vector at position "t1".
    #   u2: numpy array-like, shape (3,), the target vector at position "t2".
    #   t1: numpy array-like, shape (3,), the initial tangent direction.
    #   t2: numpy array-like, shape (3,), the target tangent direction.
    #   refTwist: float, an optional initial guess for the reference twist angle (in radians).
    #             If not provided, it defaults to zero.
    #
    # Returns:
    #   refTwist: float, the adjusted reference twist angle (in radians) that aligns "u1"
    #             with "u2" after parallel transport and rotation.
    #
    # The function works by:
    # 1. Checking if "refTwist" is None and setting it to zero if no initial guess is provided.
    #    This ensures a default starting point for the twist calculation.
    # 2. Parallel transporting "u1" along "t1" to "t2" using the parallel_transport function.
    #    This adjusts "u1" so that it is aligned correctly in the direction of "t2".
    # 3. Rotating the parallel transported vector "ut" around "t2" by the current "refTwist" angle
    #    using the rotateAxisAngle function. This accounts for the initial twist guess.
    # 4. Computing the angle between the rotated "ut" and "u2" using the signedAngle function,
    #    with "t2" as the axis to determine the direction of the angle.
    # 5. Adjusting "refTwist" by adding the signed angle, thus refining the twist angle to
    #    align "u1" with "u2" after transport and rotation.
    #
    # Example:
    #   computeReferenceTwist(np.array([1, 0, 0]), np.array([0, 1, 0]),
    #                         np.array([0, 0, 1]), np.array([0, 0, 1]))
    #   This would compute the reference twist needed to align the x-axis with the y-axis
    #   after transporting along the same tangent direction, starting with an initial guess of 0.
    if refTwist is None:
      refTwist = 0
    ut = parallel_transport(u1, t1, t2)
    ut = rotateAxisAngle(ut, t2, refTwist)
    refTwist = refTwist + signedAngle(ut, u2, t2)
    return refTwist


def getRefTwist(d1, tangent, refTwist = None):
    # This function calculates the reference twist angles along a series of nodes of an elastic rod.
    # It uses the first director "d1" and the tangent directions between each consecutive node to determine
    # the twist required to align the directors while accounting for potential angle ambiguity.
    #
    # Parameters:
    #   d1: numpy array-like, shape (n, 3), representing the first director of the material frame at each node.
    #       Each row corresponds to a director vector at a particular node, defining orientation.
    #   tangent: numpy array-like, shape (n, 3), the tangent vectors along the rod at each node.
    #            These vectors represent the direction of the rod between each pair of nodes.
    #   refTwist: numpy array-like, shape (n,), optional initial guess for the reference twist angles
    #             between each pair of nodes. If not provided, the initial guess is set to zero for all nodes.
    #
    # Returns:
    #   refTwist: numpy array-like, shape (n,), the computed reference twist angles.
    #             It holds the adjusted twist values for each node, ensuring consistent twist transitions
    #             between nodes.
    #
    # The function works by:
    # 1. Determining the number of elements ("ne") from the shape of "d1".
    # 2. Initializing the "refTwist" array if it is not provided, setting the initial guess to zero.
    # 3. Iterating over each pair of consecutive nodes (from 1 to "ne"), and for each pair:
    #    - Extracting the director vectors "u0" (previous) and "u1" (current).
    #    - Extracting the tangent vectors "t0" (previous) and "t" (current).
    #    - Computing the reference twist for each pair using the `computeReferenceTwist` function.
    #      This ensures the twist between nodes is smooth and resolves any ambiguity.
    # 4. Storing the computed reference twist for each node in the "refTwist" array.
    #
    # This function is particularly important when simulating elastic rods with large twist angles,
    # where angles like "alpha" and "2*pi + alpha" can represent different physical configurations.
    # By iteratively refining the twist angles, the function ensures the correct twist is applied
    # throughout the rod.
    #
    # Example:
    #   getRefTwist(d1_array, tangent_array)
    #   This computes the reference twists for a given rod configuration where "d1_array"
    #   represents the directors and "tangent_array" represents the tangents at each node.
    ne = d1.shape[0]
    if refTwist is None:
      refTwist = 0
    for c in np.arange(1,ne):
        u0 = d1[c-1,0:3]
        u1 = d1[c,  0:3]
        t0 = tangent[c-1,0:3]
        t =  tangent[c,  0:3]
        refTwist[c] = computeReferenceTwist(u0, u1, t0, t, refTwist[c])
    return refTwist


def getKappa(q, m1, m2):
    """
    Compute the curvature kappa for each node along an elastic rod.

    Parameters:
    - q: Numpy array of size (ndof, ) containing the degrees of freedom (DOF) values,
      where each node is represented by four consecutive entries in the array (x, y, z, theta).
    - m1: Numpy array of size (ne, 3) containing the first director vector for each edge,
      defining the material direction of the rod.
    - m2: Numpy array of size (ne, 3) containing the second director vector for each edge,
      orthogonal to m1, also defining the material direction of the rod.

    Returns:
    - kappa: Numpy array of size (nv, 2) containing the computed curvatures for each node.
      Each row represents a node, where:
        - kappa[:, 0]: Curvature component in the direction of the average of material directors 2
          for edges before and after the node.
        - kappa[:, 1]: Curvature component in the direction of the negative average of material directors 1
          for edges before and after the node.

    The function works as follows:
    1. **Determine the number of nodes (nv) and edges (ne)**:
       - `nv` is calculated as the number of degrees of freedom divided by 4 (since each node has 4 DOF: x, y, z, and theta).
       - `ne` is one less than `nv` as the number of edges is always one less than the number of nodes.

    2. **Initialize the kappa array**:
       - A (nv, 2) array is created to store curvature values for each node.

    3. **Loop through each edge**:
       - The loop starts from the second edge and iterates to the last edge (`range(2, ne)`).
       - For each edge, extract the positions of the nodes (node0, node1, node2) based on their indices in the `q` array.
       - Extract the corresponding material directors `m1` and `m2` for the edges before and after the turning node.
       - Compute the local curvature at the current node using the `computekappa` function.

    4. **Store the computed curvature values**:
       - The computed curvature values from `computekappa` are assigned to the appropriate row in the `kappa` array.

    5. **Return the kappa array**:
       - After looping through all edges, the function returns the complete kappa array containing curvature values for each node.

    Example usage:
        kappa_values = getKappa(q, m1, m2)
        This will return the curvature values computed for the elastic rod defined by the DOF in `q`
        and the material directors `m1` and `m2`.
    """
    nv = (len(q) + 1) // 4  # nv is number of nodes
    ne = nv - 1  # ne is number of edges

    kappa = np.zeros((nv, 2))  # Initialize kappa array

    for c in range(2, nv):  # Loop over edges (from second to last)

        # Extract node positions from q
        node0 = q[4*c-8:4*c-5]
        node1 = q[4*c-4:4*c-1]
        node2 = q[4*c+0:4*c+3]

        # Extract m1 and m2 for the current and previous edges
        m1e = m1[c-2,:].flatten()  # m1 vector on c-1 th edge
        # Another option is m1e = np.squeeze(np.array(m1[c-2, :]))
        m2e = m2[c-2,:].flatten()  # m2 vector on c-1 th edge
        m1f = m1[c-1,:].flatten()  # m1 vector on c th edge
        m2f = m2[c-1,:].flatten()  # m2 vector on c th edge

        # Compute local curvature at each node
        kappa_local = computekappa(node0, node1, node2, m1e, m2e, m1f, m2f)

        # Store the curvature values
        kappa[c-1, 0] = kappa_local[0]
        kappa[c-1, 1] = kappa_local[1]

    return kappa


# ---------------------------------------------------------------------------
# Elastic forces along the arclength (assembled over all springs)
# ---------------------------------------------------------------------------

def getFs(x, EA, refLen):

    """
Inputs:
x: DOF vector of size 4*nv-1 (nv is the number of nodes)
      ne = nv - 1 is the number of edges.
refTwist: vector of size nv containing reference twist at each node.
twistBar: vector of size nv containing undeformed twist at each node.
      This undeformed twist is typically zero.
EA: scalar stretching stiffness. This can easily be converted into a vector
refLen: vector of size "ne" containing the undeformed reference length
      of each node.

Outputs:
Fs: Stretching force vector of size 4*nv-1
Js: Stretching force Jacobian of size (4*nv-1) X (4*nv-1)
    """

    ndof = len(x)
    nv = int( ( ndof + 1) / 4 ) # Number of vertices
    ne = nv - 1

    Fs = np.zeros(ndof)
    Js = np.zeros((ndof,ndof))
    for c in np.arange(0,ne):

        node0 = x[4*c : 4*c+3]
        node1 = x[4*(c+1) : 4*(c+1)+3]
        ind = np.array([4*c, 4*c+1, 4*c+2, 4*c+4, 4*c+5, 4*c+6])

        l_k = refLen[c]

        dF, dJ = gradEs_hessEs(node0, node1, l_k, EA)

        for kx in np.arange(len(ind)):
            Fs[ind[kx]] = Fs[ind[kx]] - dF[kx]

        for kx in np.arange(len(ind)):
            for ky in np.arange(len(ind)):
                Js[ind[kx], ind[ky]] = Js[ind[kx], ind[ky]] - dJ[kx, ky]

    return Fs,Js


def getFb(x, m1, m2, kappaBar, EI, voronoiRefLen):

    """
Inputs:
x: DOF vector of size 4*nv-1 (nv is the number of nodes)
      ne = nv - 1 is the number of edges.
m1: ne X 3 matrix - first material director.
m2: ne X 3 matrix - second material director.
kappaBar: nv X 2 matrix - natural curvature at each node. Note
      that the natural curvature at each node is a vector of size 2.
EI: scalar bending stiffness. This can easily be converted into a vector
voronoiRefLen: vector of size "nv" containing the voronoi length
      (undeformed) of each node. At each node, voronoi length is equal
      to the average of the reference edge length of the previous and
      next edges.

Outputs:
Fb: Bending force vector of size 4*nv-1
Jb: Bending force Jacobian of size (4*nv-1) X (4*nv-1)
    """

    ndof = len(x)
    nv = int( ( ndof + 1) / 4 ) # Number of vertices
    ne = nv - 1

    Fb = np.zeros(ndof)
    Jb = np.zeros((ndof,ndof))
    for c in np.arange(1,ne):

        node0 = x[4*(c-1) : 4*(c-1)+3]
        node1 = x[4*c : 4*c+3]
        node2 = x[4*(c+1) : 4*(c+1)+3]

        m1e = m1[ c-1, 0:3]
        m2e = m2[ c-1, 0:3]
        m1f = m1[ c, 0:3]
        m2f = m2[ c, 0:3]

        l_k = voronoiRefLen[c]
        kappaBar_node = kappaBar[c, 0:2]

        dF, dJ = gradEb_hessEb(node0, node1, node2, m1e, m2e, m1f, m2f, kappaBar_node, l_k, EI)

        ci = 4*(c-1)
        cf = 4*(c+1)+3

        Fb[ci:cf] = Fb[ci:cf] - dF
        Jb[ci:cf, ci:cf] = Jb[ci:cf, ci:cf] - dJ

    return Fb,Jb


def getFt(x, refTwist, twistBar, GJ, voronoiRefLen):

    """
Inputs:
x: DOF vector of size 4*nv-1 (nv is the number of nodes)
      ne = nv - 1 is the number of edges.
refTwist: vector of size nv containing reference twist at each node.
twistBar: vector of size nv containing undeformed twist at each node.
      This undeformed twist is typically zero.
GJ: scalar twisting stiffness. This can easily be converted into a vector
voronoiRefLen: vector of size "nv" containing the voronoi length
      (undeformed) of each node. At each node, voronoi length is equal
      to the average of the reference edge length of the previous and
      next edges.

Outputs:
Ft: Twisting force vector of size 4*nv-1
Jt: Twisting force Jacobian of size (4*nv-1) X (4*nv-1)
    """

    ndof = len(x)
    nv = int( ( ndof + 1) / 4 ) # Number of vertices
    ne = nv - 1

    Ft = np.zeros(ndof)
    Jt = np.zeros((ndof,ndof))
    for c in np.arange(1,ne):

        node0 = x[4*(c-1) : 4*(c-1)+3]
        node1 = x[4*c : 4*c+3]
        node2 = x[4*(c+1) : 4*(c+1)+3]

        theta_e = x[4*c-1]
        theta_f = x[4*c+3]

        l_k = voronoiRefLen[c]
        refTwist_node = refTwist[c]
        twistBar_node = twistBar[c]

        dF, dJ = gradEt_hessEt(node0, node1, node2, theta_e, theta_f,
            refTwist_node, twistBar_node, l_k, GJ)

        ci = 4*(c-1)
        cf = 4*(c+1)+3

        Ft[ci:cf] = Ft[ci:cf] - dF
        Jt[ci:cf, ci:cf] = Jt[ci:cf, ci:cf] - dJ

    return Ft,Jt


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


def plotrod(q, a1, a2, m1, m2, ctime):
    """
    Function to plot the rod with the position and directors.

    Parameters:
    - q: Position vector (DOF vector).
    - a1, a2: Reference frames (director vectors).
    - m1, m2: Material directors.
    - ctime: Current time for title.
    """

    nv = (len(q) + 1) // 4
    x1 = q[0::4]
    x2 = q[1::4]
    x3 = q[2::4]

    # Compute the length of the rod
    L = np.sum(np.sqrt((x1[1:] - x1[:-1])**2 +
                       (x2[1:] - x2[:-1])**2 +
                       (x3[1:] - x3[:-1])**2))

    # Scale the director vectors by 0.1 * L
    a1 *= 0.1 * L
    a2 *= 0.1 * L
    m1 *= 0.1 * L
    m2 *= 0.1 * L

    # Create figure and set up 3D plotting
    fig = plt.figure(1)
    plt.clf()  # Clear the figure
    ax = fig.add_subplot(111, projection='3d')

    # Plot the rod as black circles connected by lines
    ax.plot3D(x1, x2, x3, 'ko-')

    # Plot the first node with a red triangle
    ax.plot3D([x1[0]], [x2[0]], [x3[0]], 'r^')

    # Plot the directors along the rod
    for c in range(nv - 1):
        xa = q[4 * c : 4 * c + 3]
        xb = q[4 * c + 4 : 4 * c + 7]
        xp = (xa + xb) / 2  # Midpoint between xa and xb

        # Plot the a1, a2, m1, m2 vectors at the midpoint
        ax.plot3D([xp[0], xp[0] + a1[c, 0]], [xp[1], xp[1] + a1[c, 1]],
                  [xp[2], xp[2] + a1[c, 2]], 'b--', linewidth=2)
        ax.plot3D([xp[0], xp[0] + a2[c, 0]], [xp[1], xp[1] + a2[c, 1]],
                  [xp[2], xp[2] + a2[c, 2]], 'c--', linewidth=2)
        ax.plot3D([xp[0], xp[0] + m1[c, 0]], [xp[1], xp[1] + m1[c, 1]],
                  [xp[2], xp[2] + m1[c, 2]], 'r-', linewidth=2)
        ax.plot3D([xp[0], xp[0] + m2[c, 0]], [xp[1], xp[1] + m2[c, 1]],
                  [xp[2], xp[2] + m2[c, 2]], 'g-', linewidth=2)

    # Add legend
    ax.legend(['a1', 'a2', 'm1', 'm2'])

    # Set the title with current time
    ax.set_title(f't={ctime:.2f}')

    # Set axes labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Set equal scaling using the custom function
    set_axes_equal(ax)

    plt.draw()
    plt.pause(0.01)  # Display the figure without blocking


def plotrod_simple(q, ctime):
    """
    Function to plot the rod with the position and directors.

    Parameters:
    - q: Position vector (DOF vector).
    - ctime: Current time for title.
    """

    x1 = q[0::4]
    x2 = q[1::4]
    x3 = q[2::4]

    fig = plt.figure(1)
    plt.clf()  # Clear the figure
    ax = fig.add_subplot(111, projection='3d')

    # Plot the rod as black circles connected by lines
    ax.plot3D(x1, x2, x3, 'ko-')

    # Plot the first node with a red triangle
    ax.plot3D([x1[0]], [x2[0]], [x3[0]], 'r^')

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

def objfun(qGuess, q0, u, a1, a2,
           freeIndex, dt, tol,
           refTwist, massVector, mMat,
           EA, refLen, EI, GJ, voronoiRefLen,
           kappaBar, twistBar, Fg):
    q = qGuess
    iter = 0
    error = 10 * tol

    while error > tol:
        a1Iterate, a2Iterate = computeTimeParallel(a1, q0, q)
        tangent = computeTangent(q)
        refTwist_iterate = getRefTwist(a1Iterate, tangent, refTwist)
        theta = q[3::4]
        m1, m2 = computeMaterialDirectors(a1Iterate, a2Iterate, theta)

        Fb, Jb = getFb(q, m1, m2, kappaBar, EI, voronoiRefLen)
        Ft, Jt = getFt(q, refTwist_iterate, twistBar, GJ, voronoiRefLen)
        Fs, Js = getFs(q, EA, refLen)

        Forces = Fb + Ft + Fs + Fg
        JForces = Jb + Jt + Js
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
    a1 = a1Iterate
    a2 = a2Iterate
    return q, u, a1, a2
