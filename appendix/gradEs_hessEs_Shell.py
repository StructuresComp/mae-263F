## Copyright M. Khalid Jawed (khalidjm@seas.ucla.edu)
# You should use this code at your own risk. Copy and redistribution is not
# permitted. Written permission is required.

import numpy as np

def gradEs_hessEs_Shell(node0 = None,node1 = None,l_k = None,EA = None):

# Inputs:
# node0: array of 3 - position of the first node
# node1: array of 3 - position of the last node
#
# l_k: reference length (undeformed) of the edge
# EA: scalar - stretching stiffness - Young's modulus times area
#
# Outputs:
# dF: array of 6 - gradient of the stretching energy between node0 and node 1.
# dJ: 6x6 array - hessian of the stretching energy between node0 and node 1.

    ## Gradient of Es
    node0 = np.asarray(node0, dtype=float).flatten()
    node1 = np.asarray(node1, dtype=float).flatten()
    edge = node1 - node0 # edge vector (array of 3)
    edgeLen = np.linalg.norm(edge)
    tangent = edge / edgeLen
    epsX = edgeLen / l_k - 1
    dF_unit = EA * tangent * epsX

    dF = np.zeros(6)
    dF[0:3] = - dF_unit
    dF[3:6] =   dF_unit

    ## Hessian of Es
    Id3 = np.eye(3)
    M = EA * ( \
        (1 / l_k - 1 / edgeLen) * Id3 + \
        1 / edgeLen * np.outer(edge, edge) / edgeLen ** 2 \
        ) # Note np.outer(edge, edge) is 3x3

    dJ = np.zeros((6,6))
    dJ[0:3,0:3] = M
    dJ[3:6,3:6] = M
    dJ[0:3,3:6] = - M
    dJ[3:6,0:3] = - M

    return dF,dJ
