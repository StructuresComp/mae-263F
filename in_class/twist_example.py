import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'der'))

import numpy as np

from der_functions import (parallel_transport, signedAngle, rotateAxisAngle)

rng = np.random.default_rng(4)   # change the seed (or use None) to get other frames


def assignOrthonormalAdaptedFrame(tangent, randomize=False, rng=None):
    d1 = np.cross(tangent, np.array([0.0, 0.0, -1.0]))
    if np.linalg.norm(d1) < 1e-10:
        d1 = np.cross(tangent, np.array([0.0, 1.0, 0.0]))
    d1 = d1 / np.linalg.norm(d1)
    # Solely for randomization and demonstration
    if randomize:
        if rng is None:
            rng = np.random.default_rng()
        d1 = rotateAxisAngle(d1, tangent, rng.uniform(0.0, 2.0 * np.pi))
    d2 = np.cross(tangent, d1)
    return d1, d2

# Given nodes
x1 = np.array([0.00, 0.00, 0.00])
x2 = np.array([0.50, 0.00, 0.00])
x3 = np.array([0.75, 0.25, 0.00])

t1 = (x2 - x1) / np.linalg.norm(x2 - x1)   # tangent on the first edges
t2 = (x3 - x2) / np.linalg.norm(x3 - x2)   # tangent on the second edges

# Material frame on the first edge: a random orthonormal adapted frame
m1_1, m2_1 = assignOrthonormalAdaptedFrame(t1, randomize=True, rng=rng)

tau_true = np.pi / 4 # True value of twist between the first and second edges
m1_2 = rotateAxisAngle(parallel_transport(m1_1, t1, t2), t2, tau_true)
m2_2 = np.cross(t2, m1_2)

print('Given:')
print(f'  t1 = {t1},  t2 = {t2}')
print(f'  m1 on edge 1 = {m1_1}')
print(f'  m1 on edge 2 = {m1_2}')
print(f'  (twist used to build the frames: tau_true = {tau_true:.6f} rad)\n')

# TODO: Compute the twist of the rod defined above using two different methods