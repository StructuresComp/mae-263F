import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'der'))

import numpy as np

from der_functions import (parallel_transport, signedAngle, rotateAxisAngle,
                           computeReferenceTwist)

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

def wrapToPi(angle):
    # Map an angle in radians to the range [-pi, pi]
    return angle - 2.0 * np.pi * np.ceil((angle - np.pi) / (2.0 * np.pi))

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
# October 19, 2026

# Method 1: Twist without invoking reference frame
P_m1_1 = parallel_transport(m1_1, t1, t2)
tau_method1 = signedAngle(P_m1_1, m1_2, t2) # "discrete integrated twist"

print(f'METHOD 1: Twist without invoking reference frame: tau = {tau_method1:.6f} rad')

# Method 2: Twist with invoking reference frame

# Create arbitrary reference frame on the first edge
a1_1, a2_1 = assignOrthonormalAdaptedFrame(t1, randomize=True, rng=rng)
a1_2, a2_2 = assignOrthonormalAdaptedFrame(t2, randomize=True, rng=rng)
print("Method 2: Twist with invoking reference frame: ")
print(f'  a1 on edge 1 = {a1_1}')
print(f'  a1 on edge 2 = {a1_2}')

theta1 = signedAngle(a1_1, m1_1, t1)
theta2 = signedAngle(a1_2, m1_2, t2)
print(f'  theta1 = {theta1:.6f} rad, theta2 = {theta2:.6f} rad')

# Compute "reference twist" between the two reference frames
# P_a1_1 = parallel_transport(a1_1, t1, t2)
# m_ref = signedAngle(P_a1_1, a1_2, t2)
m_ref = computeReferenceTwist(a1_1, a1_2, t1, t2)  # This function is assumed to compute the reference twist

tau_method2 = (theta2 - theta1) + m_ref
tau_method2 = wrapToPi(tau_method2)  # Map to [-pi, pi]
print(f'METHOD 2: Twist with invoking reference frame: tau = {tau_method2:.6f} rad')
