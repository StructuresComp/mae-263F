"""
MAE 263F, Fall 2026 - Homework 4: reference solution (instructor only).

Discrete elastic plates and shells:
  Q1     edges and hinges of a small triangle mesh
  Q2     hinge angle and bending energy of a four-node hinge (plate)
  Q3-Q6  bending energy after moving one coordinate of x0, x1, x2, x3 by 0.01
  Q7     finite-difference gradient (12 numbers)
  Q8     gradient from gradEb_hessEb_Shell (12 numbers)
  Q9-Q11 conceptual questions on Newmark-Beta and predictor-corrector (no computation)

Run with a single command (takes about one second):
    python Homework4_reference.py

It uses getEdgesAndHinges (../codes/shell_functions.py) and getTheta / gradEb_hessEb_Shell
(appendix/gradEb_hessEb_Shell.py, as in the course repository).
"""

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Imports from the class code
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
for path in [os.path.join(HERE, '..', 'codes'),
             os.path.join(HERE, '..', 'src', 'shells'),                  # this repository
             os.path.join(HERE, '..', '..', '..', 'Codes', 'appendix'),  # course folder layout
             os.path.join(HERE, '..', 'appendix')]:
    if os.path.isdir(path):
        sys.path.append(path)

from gradEb_hessEb_Shell import gradEb_hessEb_Shell, getTheta
from shell_functions import getEdgesAndHinges


def bendingEnergy(q, thetaBar, kb):
    """E_b = 0.5 * kb * (theta - thetaBar)^2 for one hinge; q = [x0, x1, x2, x3] (size 12)."""
    return 0.5 * kb * (getTheta(q) - thetaBar)**2


# ---------------------------------------------------------------------------
# Q1: edges and hinges of a 3 x 3 node mesh (8 triangles)
# ---------------------------------------------------------------------------
# Nodes: node = 3*j + i at (x, y) = (i, j), i, j = 0, 1, 2.
# Each square is cut by the diagonal from its lower-left to its upper-right corner.
def squareMesh():
    elements = []
    for j in range(2):
        for i in range(2):
            a = 3 * j + i        # lower left
            b = a + 1            # lower right
            c = a + 4            # upper right
            d = a + 3            # upper left
            elements.append([a, b, c])  # counterclockwise
            elements.append([a, c, d])  # counterclockwise
    return np.array(elements, dtype=int)


if __name__ == '__main__':
    # Q1
    edges, hinges = getEdgesAndHinges(squareMesh())
    print(f'Q1: edges = {edges.shape[0]}, hinges = {hinges.shape[0]}   -> 16 edges, 8 hinges')

    # Hinge used in Q2-Q8 (lengths in m), kb = 1 N m
    x0 = np.array([0.0, 0.0, 0.0])
    x1 = np.array([1.0, 0.0, 0.0])
    x2 = np.array([0.5, 1.0, 0.0])
    x3 = np.array([0.5, -1.0, 1.0])
    q = np.concatenate((x0, x1, x2, x3))
    kb = 1.0

    # Q2: hinge angle and bending energy (plate, thetaBar = 0)
    theta = getTheta(q)
    E0 = bendingEnergy(q, 0.0, kb)
    print(f'Q2: theta = {theta:.6f} rad = {theta / np.pi:.4f} pi, E_b = {E0:.5f} J   -> -pi/4, 0.308 J')

    # Q3-Q6: perturb each coordinate of each node by eps = 0.01 (one at a time)
    eps = 0.01
    Ep = np.zeros(12)
    for i in range(12):
        qp = q.copy()
        qp[i] += eps
        Ep[i] = bendingEnergy(qp, 0.0, kb)
    for n in range(4):
        print(f'Q{n + 3}: x{n} moved in x, y, z: E_b = ' + ', '.join(f'{e:.5f}' for e in Ep[3*n:3*n+3]) + ' J')

    # Q7: finite-difference gradient
    gradFD = (Ep - E0) / eps
    print('Q7: gradFD = [' + ' | '.join(', '.join(f'{v:.3f}' for v in gradFD[3*n:3*n+3]) for n in range(4)) + ']')

    # Q8: gradient from gradEb_hessEb_Shell
    dF, _ = gradEb_hessEb_Shell(q, None, None, None, 0.0, kb)
    print('Q8: dF     = [' + ' | '.join(', '.join(f'{v:.3f}' for v in dF[3*n:3*n+3]) for n in range(4)) + ']')
    print(f'    max |gradFD - dF| = {np.max(np.abs(gradFD - dF)):.4f}')

    # Q9-Q11: conceptual (Chapter 6 and Chapter 11 of the notes)
    print('Q9: implicit Euler adds numerical damping; energy decreases   -> (c)')
    print('Q10: beta = 0.25, gamma = 0.5 (average acceleration) conserves energy   -> (d)')
    print('Q11: corrector only when the contact set changes (penetration or adhesive reaction)   -> (a)')
