"""
Exercise: integrated discrete twist between two edges, computed in two ways.

Copyright M. Khalid Jawed (khalidjm@seas.ucla.edu). License: CC BY-NC
You should use this code at your own risk.

Background: Chapter 7 of the course notes (MAE_263F___Discrete_Simulation_Notes.pdf).
  - Orthonormal adapted frame ............ Section 7.1
  - Signed angle, parallel transport ..... Sections 7.1-7.2 (Eq. 7.1)
  - Integrated discrete twist ............ Section 7.2
  - Twist with a reference frame ......... Section 7.5 (Eq. 7.4)

Setup: three nodes x1, x2, x3 and two edges e1 = x2 - x1, e2 = x3 - x2, with tangents
t1 and t2. Each edge carries a material frame (m1, m2, t). We build the material
frames so that the twist between them is known (tau_true), and then recover it with:

  Method 1 (no reference frame, Section 7.2):
      tau = signedAngle( P(m1^1), m1^2, t2 ),
      where P(.) = parallel_transport(., t1, t2).

  Method 2 (with a reference frame, Eq. 7.4):
      Assign an arbitrary (random) adapted reference frame (a1, a2, t) to each edge.
      theta^k   = signedAngle( a1^k, m1^k, t^k )               (k = 1, 2)
      delta_ref = signedAngle( P(a1^1), a1^2, t2 )              (reference twist)
      tau       = theta^2 - theta^1 + delta_ref

The reference frame is arbitrary, so theta^1, theta^2, and delta_ref change every
time it is re-assigned, but tau does not.

Run:  python exercise_twist_two_methods.py
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from der_functions import (assignOrthonormalAdaptedFrame, parallel_transport, signedAngle,
                           computeReferenceTwist)
from der_functions import rotateAxisAngle   # from Materials/appendix


def wrapToPi(angle):
    # Map an angle to (-pi, pi]. Every signed angle lies in (-pi, pi], so the sum in
    # Method 2 can differ from the true twist by a multiple of 2*pi; wrapping removes it.
    return angle - 2 * np.pi * np.ceil((angle - np.pi) / (2 * np.pi))


# ---------------------------------------------------------------------------
# Problem preparation: nodes, tangents, and material frames
# ---------------------------------------------------------------------------

rng = np.random.default_rng(4)   # change the seed (or use None) to get other frames

# Three nodes and two edges (the first three nodes of Assignment 1, Section 7.2)
x1 = np.array([0.00, 0.00, 0.00])
x2 = np.array([0.50, 0.00, 0.00])
x3 = np.array([0.75, 0.25, 0.00])

t1 = (x2 - x1) / np.linalg.norm(x2 - x1)   # tangent on the first edge
t2 = (x3 - x2) / np.linalg.norm(x3 - x2)   # tangent on the second edge

# Material frame on the first edge: a random orthonormal adapted frame
m1_1, m2_1 = assignOrthonormalAdaptedFrame(t1, randomize=True, rng=rng)

# Material frame on the second edge: parallel transport the first one to t2, then
# rotate it about t2 by tau_true. By construction, the twist between the two
# material frames is tau_true.
tau_true = np.pi / 4
m1_2 = rotateAxisAngle(parallel_transport(m1_1, t1, t2), t2, tau_true)
m2_2 = np.cross(t2, m1_2)

print('Given:')
print(f'  t1 = {t1},  t2 = {t2}')
print(f'  m1 on edge 1 = {m1_1}')
print(f'  m1 on edge 2 = {m1_2}')
print(f'  (twist used to build the frames: tau_true = {tau_true:.6f} rad)\n')


# ---------------------------------------------------------------------------
# Method 1: twist without a reference frame (Section 7.2)
# ---------------------------------------------------------------------------

P_m1_1 = parallel_transport(m1_1, t1, t2)      # P(m1^1): material director 1, moved to edge 2
tau_method1 = signedAngle(P_m1_1, m1_2, t2)    # signed angle from P(m1^1) to m1^2 about t2

print(f'Method 1 (no reference frame):  tau = {tau_method1:.6f} rad\n')


# ---------------------------------------------------------------------------
# Method 2: twist with a reference frame (Section 7.5, Eq. 7.4)
# Repeat with several random reference frames to show that tau does not depend on them.
# ---------------------------------------------------------------------------

print('Method 2 (with a random reference frame on each edge):')
print(f'  {"trial":>5}  {"theta1":>10}  {"theta2":>10}  {"ref. twist":>10}  {"sum":>10}  {"tau":>10}')

for trial in range(1, 6):
    # Arbitrary reference frames (a1, a2, t) on the two edges
    a1_1, a2_1 = assignOrthonormalAdaptedFrame(t1, randomize=True, rng=rng)
    a1_2, a2_2 = assignOrthonormalAdaptedFrame(t2, randomize=True, rng=rng)

    # Twist angles: signed angle from the reference frame to the material frame
    theta1 = signedAngle(a1_1, m1_1, t1)
    theta2 = signedAngle(a1_2, m1_2, t2)

    # Reference twist: signed angle from P(a1^1) to a1^2 about t2.
    # computeReferenceTwist (used in the DER code) does exactly this; with an initial
    # guess of 0 it equals signedAngle(parallel_transport(a1_1, t1, t2), a1_2, t2).
    refTwist = computeReferenceTwist(a1_1, a1_2, t1, t2, 0.0)

    # Eq. 7.4
    tau_sum = theta2 - theta1 + refTwist
    tau_method2 = wrapToPi(tau_sum)

    print(f'  {trial:>5}  {theta1:10.6f}  {theta2:10.6f}  {refTwist:10.6f}  {tau_sum:10.6f}  {tau_method2:10.6f}')

    assert np.isclose(tau_method2, tau_method1), 'Methods 1 and 2 disagree'

assert np.isclose(tau_method1, tau_true), 'Method 1 does not recover tau_true'
print('\nBoth methods recover tau_true; theta1, theta2, and the reference twist depend on the')
print('(arbitrary) reference frame, but their combination in Eq. 7.4 does not.')
print('In some trials the raw sum differs from tau by 2*pi: each signed angle lies in (-pi, pi],')
print('so the sum is only defined up to a multiple of 2*pi. (In the DER time loop, this is avoided')
print('by passing the previous reference twist as the initial guess to computeReferenceTwist.)')
