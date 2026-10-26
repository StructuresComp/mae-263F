import numpy as np

#         x2
#         /\
#        /  \
#     e1/    \e3
#      /  t0  \
#     /        \
#    /    e0    \
#  x0------------x1
#    \          /
#     \   t1   /
#      \      /
#     e2\    /e4
#        \  /
#         \/
#         x3
#
# Edge orientation: e0,e1,e2 point away from x0
#                      e3,e4 point away from x1

# In the original code, there are probaly TWO sign errors in the expressions for m_h3 and m_h4.
# [Original code: https://github.com/shift09/plates-shells/blob/master/src/bending.cpp]
# I indicated those two corrections by writing the word "CORRECTION" next
# to them.

def gradEb_hessEb_Shell(x0 = None,x1 = None,x2 = None,x3 = None,thetaBar = None,kb = None):

# Inputs:
# x0, x1, x2, x3: array of 3 - positions of the four nodes of the hinge (see figure above).
#                 The hinge (shared) edge is x0-x1.
#                 Alternatively, x0 can be an array of 12 = [x0, x1, x2, x3]; then x1, x2, x3 are not needed.
#
# thetaBar: scalar - natural (undeformed) hinge angle
# kb: scalar - bending stiffness
#
# Outputs:
# dF: array of 12 - gradient of the bending energy E = 0.5 * kb * (theta-thetaBar)^2
# dJ: 12x12 array - hessian of the bending energy

    if np.size(x0) == 12: # Let us allow another type of inputs. In this case, x0 contains all the info.
        x0, x1, x2, x3 = splitNodes(x0)

    # E = 0.5 * kb * (theta-thetaBar)^2
    # F = dE/dx = 2 * (theta-thetaBar) * gradTheta
    theta = getTheta(x0, x1, x2, x3)
    grad = gradTheta(x0, x1, x2, x3)
    dF = 0.5 * kb * (2 * (theta - thetaBar) * grad)

    # E = 0.5 * kb * (theta-thetaBar)^2
    # F = 0.5 * kb * (2 (theta-thetaBar) d theta/dx)
    # J = dF/dx = 0.5 * kb * [ 2 (d theta / dx) transpose(d theta/dx) +
    #       2 (theta-thetaBar) (d^2 theta/ dx^2 ) ]
    hess = hessTheta(x0, x1, x2, x3)
    dJ = 0.5 * kb * ( 2 * np.outer(grad, grad) + 2 * (theta - thetaBar) * hess )

    return dF,dJ


# ---------------------------------------------------------------------------
# Hinge angle (theta), its gradient, and its Hessian
# ---------------------------------------------------------------------------

def getTheta(x0 = None,x1 = None,x2 = None,x3 = None):

# Inputs: same as gradEb_hessEb_Shell (four nodes, or one array of 12)
# Output:
# theta: scalar - signed angle between the normals of the two triangles (x0,x1,x2) and (x0,x3,x1)

    if np.size(x0) == 12:
        x0, x1, x2, x3 = splitNodes(x0)

    m_e0 = x1 - x0
    m_e1 = x2 - x0
    m_e2 = x3 - x0

    n0 = np.cross(m_e0, m_e1)
    n1 = np.cross(m_e2, m_e0)

    theta = signedAngle(n0, n1, m_e0)

    return theta


def gradTheta(x0 = None,x1 = None,x2 = None,x3 = None):

# Inputs: same as gradEb_hessEb_Shell (four nodes, or one array of 12)
# Output:
# gradTheta: array of 12 - gradient of the hinge angle theta

    if np.size(x0) == 12:
        x0, x1, x2, x3 = splitNodes(x0)

    m_e0 = x1 - x0
    m_e1 = x2 - x0
    m_e2 = x3 - x0
    m_e3 = x2 - x1
    m_e4 = x3 - x1

    m_cosA1 = np.dot(m_e0, m_e1) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e1))
    m_cosA2 = np.dot(m_e0, m_e2) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e2))
    m_cosA3 = -np.dot(m_e0, m_e3) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e3))
    m_cosA4 = -np.dot(m_e0, m_e4) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e4))

    m_sinA1 = np.linalg.norm(np.cross(m_e0, m_e1)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e1))
    m_sinA2 = np.linalg.norm(np.cross(m_e0, m_e2)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e2))
    m_sinA3 = -np.linalg.norm(np.cross(m_e0, m_e3)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e3))
    m_sinA4 = -np.linalg.norm(np.cross(m_e0, m_e4)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e4))

    m_nn1 = np.cross(m_e0, m_e3)
    m_nn1 = m_nn1 / np.linalg.norm(m_nn1)
    m_nn2 = -np.cross(m_e0, m_e4)
    m_nn2 = m_nn2 / np.linalg.norm(m_nn2)

    m_h1 = np.linalg.norm(m_e0) * m_sinA1
    m_h2 = np.linalg.norm(m_e0) * m_sinA2
    m_h3 = -np.linalg.norm(m_e0) * m_sinA3  # CORRECTION
    m_h4 = -np.linalg.norm(m_e0) * m_sinA4  # CORRECTION
    m_h01 = np.linalg.norm(m_e1) * m_sinA1
    m_h02 = np.linalg.norm(m_e2) * m_sinA2

    gradTheta = np.zeros(12)
    gradTheta[0:3] = m_cosA3 * m_nn1 / m_h3 + m_cosA4 * m_nn2 / m_h4
    gradTheta[3:6] = m_cosA1 * m_nn1 / m_h1 + m_cosA2 * m_nn2 / m_h2
    gradTheta[6:9] = -m_nn1 / m_h01
    gradTheta[9:12] = -m_nn2 / m_h02

    return gradTheta


def hessTheta(x0 = None,x1 = None,x2 = None,x3 = None):

# Inputs: same as gradEb_hessEb_Shell (four nodes, or one array of 12)
# Output:
# hessTheta: 12x12 array - hessian of the hinge angle theta

    if np.size(x0) == 12:
        x0, x1, x2, x3 = splitNodes(x0)

    m_e0 = x1 - x0
    m_e1 = x2 - x0
    m_e2 = x3 - x0
    m_e3 = x2 - x1
    m_e4 = x3 - x1

    m_cosA1 = np.dot(m_e0, m_e1) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e1))
    m_cosA2 = np.dot(m_e0, m_e2) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e2))
    m_cosA3 = -np.dot(m_e0, m_e3) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e3))
    m_cosA4 = -np.dot(m_e0, m_e4) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e4))

    m_sinA1 = np.linalg.norm(np.cross(m_e0, m_e1)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e1))
    m_sinA2 = np.linalg.norm(np.cross(m_e0, m_e2)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e2))
    m_sinA3 = -np.linalg.norm(np.cross(m_e0, m_e3)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e3))
    m_sinA4 = -np.linalg.norm(np.cross(m_e0, m_e4)) / (np.linalg.norm(m_e0) * np.linalg.norm(m_e4))

    m_nn1 = np.cross(m_e0, m_e3)
    m_nn1 = m_nn1 / np.linalg.norm(m_nn1)
    m_nn2 = -np.cross(m_e0, m_e4)
    m_nn2 = m_nn2 / np.linalg.norm(m_nn2)

    m_h1 = np.linalg.norm(m_e0) * m_sinA1
    m_h2 = np.linalg.norm(m_e0) * m_sinA2
    m_h3 = -np.linalg.norm(m_e0) * m_sinA3  # CORRECTION
    m_h4 = -np.linalg.norm(m_e0) * m_sinA4  # CORRECTION
    m_h01 = np.linalg.norm(m_e1) * m_sinA1
    m_h02 = np.linalg.norm(m_e2) * m_sinA2

    # Intermediate vectors for the Hessian
    m_m1 = np.cross(m_nn1, m_e1) / np.linalg.norm(m_e1)
    m_m2 = -np.cross(m_nn2, m_e2) / np.linalg.norm(m_e2)
    m_m3 = -np.cross(m_nn1, m_e3) / np.linalg.norm(m_e3)
    m_m4 = np.cross(m_nn2, m_e4) / np.linalg.norm(m_e4)
    m_m01 = -np.cross(m_nn1, m_e0) / np.linalg.norm(m_e0)
    m_m02 = np.cross(m_nn2, m_e0) / np.linalg.norm(m_e0)

    # Hessian components
    M331 = m_cosA3 / (m_h3 ** 2) * np.outer(m_m3, m_nn1)
    M311 = m_cosA3 / (m_h3 * m_h1) * np.outer(m_m1, m_nn1)
    M131 = m_cosA1 / (m_h1 * m_h3) * np.outer(m_m3, m_nn1)
    M3011 = m_cosA3 / (m_h3 * m_h01) * np.outer(m_m01, m_nn1)
    M111 = m_cosA1 / (m_h1 ** 2) * np.outer(m_m1, m_nn1)
    M1011 = m_cosA1 / (m_h1 * m_h01) * np.outer(m_m01, m_nn1)

    M442 = m_cosA4 / (m_h4 ** 2) * np.outer(m_m4, m_nn2)
    M422 = m_cosA4 / (m_h4 * m_h2) * np.outer(m_m2, m_nn2)
    M242 = m_cosA2 / (m_h2 * m_h4) * np.outer(m_m4, m_nn2)
    M4022 = m_cosA4 / (m_h4 * m_h02) * np.outer(m_m02, m_nn2)
    M222 = m_cosA2 / (m_h2 ** 2) * np.outer(m_m2, m_nn2)
    M2022 = m_cosA2 / (m_h2 * m_h02) * np.outer(m_m02, m_nn2)

    B1 = 1 / np.linalg.norm(m_e0) ** 2 * np.outer(m_nn1, m_m01)
    B2 = 1 / np.linalg.norm(m_e0) ** 2 * np.outer(m_nn2, m_m02)

    N13 = 1 / (m_h01 * m_h3) * np.outer(m_nn1, m_m3)
    N24 = 1 / (m_h02 * m_h4) * np.outer(m_nn2, m_m4)
    N11 = 1 / (m_h01 * m_h1) * np.outer(m_nn1, m_m1)
    N22 = 1 / (m_h02 * m_h2) * np.outer(m_nn2, m_m2)
    N101 = 1 / (m_h01 ** 2) * np.outer(m_nn1, m_m01)
    N202 = 1 / (m_h02 ** 2) * np.outer(m_nn2, m_m02)

    hessTheta = np.zeros((12, 12))

    hessTheta[0:3, 0:3] = mmt(M331) - B1 + mmt(M442) - B2
    hessTheta[0:3, 3:6] = M311 + M131.T + B1 + M422 + M242.T + B2
    hessTheta[0:3, 6:9] = M3011 - N13
    hessTheta[0:3, 9:12] = M4022 - N24
    hessTheta[3:6, 3:6] = mmt(M111) - B1 + mmt(M222) - B2
    hessTheta[3:6, 6:9] = M1011 - N11
    hessTheta[3:6, 9:12] = M2022 - N22
    hessTheta[6:9, 6:9] = -mmt(N101)
    hessTheta[9:12, 9:12] = -mmt(N202)

    # The Hessian is symmetric
    hessTheta[3:6, 0:3] = hessTheta[0:3, 3:6].T
    hessTheta[6:9, 0:3] = hessTheta[0:3, 6:9].T
    hessTheta[9:12, 0:3] = hessTheta[0:3, 9:12].T
    hessTheta[6:9, 3:6] = hessTheta[3:6, 6:9].T
    hessTheta[9:12, 3:6] = hessTheta[3:6, 9:12].T

    return hessTheta


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def splitNodes(x):
    # Split an array of 12 into the four node positions x0, x1, x2, x3
    x = np.asarray(x, dtype=float).flatten()
    return x[0:3], x[3:6], x[6:9], x[9:12]


def signedAngle(u = None,v = None,n = None):
    # Signed angle from vector u to vector v; the sign is positive if the rotation
    # from u to v follows the right-hand rule about the axis n.
    w = np.cross(u,v)
    angle = np.arctan2( np.linalg.norm(w), np.dot(u,v) )
    if (np.dot(n,w) < 0):
        angle = - angle

    return angle


def mmt(matrix):
    # matrix + transpose(matrix)
    return matrix + matrix.T
