import numpy as np

def computekappa(node0 = None,node1 = None,node2 = None,m1e = None,m2e = None,m1f = None,m2f = None):

# Inputs:
# node0: array of 3 - position of the node prior to the "turning" node
# node1: array of 3 - position of the "turning" node
# node2: array of 3 - position of the node after the "turning" node

# m1e: array of 3 - material director 1 of the edge prior to turning
# m2e: array of 3 - material director 2 of the edge prior to turning
# m1f: array of 3 - material director 1 of the edge after turning
# m2f: array of 3 - material director 2 of the edge after turning

# Outputs:
# kappa: array of 2 - curvature at the turning node

    t0 = (node1 - node0) / np.linalg.norm(node1 - node0)
    t1 = (node2 - node1) / np.linalg.norm(node2 - node1)
    kb = 2.0 * np.cross(t0,t1) / (1.0 + np.dot(t0,t1))

    kappa = np.zeros(2)
    kappa1 = 0.5 * np.dot(kb,m2e + m2f)

    kappa2 = - 0.5 * np.dot(kb,m1e + m1f)
    kappa[0] = kappa1
    kappa[1] = kappa2
    return kappa
