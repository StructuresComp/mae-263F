"""
Unit tests for the DER functions (moved from modular_main_discrete_elastic_rods.py).

Run:  python test_der_functions.py
"""

import numpy as np

from der_functions import signedAngle, parallel_transport, getRefTwist, assignOrthonormalAdaptedFrame
from der_functions import rotateAxisAngle, computekappa   # from Materials/appendix


def test_signedAngle():
  """
  This function tests the signedAngle function with three test cases.
  """
  # Test case 1: Orthogonal vectors
  u = np.array([1, 0, 0])
  v = np.array([0, 1, 0])
  n = np.array([0, 0, 1])
  angle = signedAngle(u, v, n)
  assert np.isclose(angle, np.pi/2), "Test case 1 failed"

  # Test case 2: Parallel vectors
  u = np.array([1, 1, 1])
  v = np.array([2, 2, 2])
  n = np.array([0, 1, 0])
  angle = signedAngle(u, v, n)
  assert np.isclose(angle, 0), "Test case 2 failed"

  # Test case 3: Anti-parallel vectors
  u = np.array([1, 1, 1])
  v = np.array([-1, -1, -1])
  n = np.array([0, 1, 0])
  angle = signedAngle(u, v, n)
  assert np.isclose(angle, np.pi), "Test case 3 failed"

  print("All test cases passed")


def test_rotateAxisAngle():
  """
  This function tests the rotateAxisAngle function by comparing the output with
  the expected output for a given set of inputs.
  """

  # Test case: Rotate a vector by 90 degrees around the z-axis
  v = np.array([1, 0, 0])
  axis = np.array([0, 0, 1])
  theta = np.pi/2
  v_rotated = rotateAxisAngle(v, axis, theta)

  # Expected output
  v_expected = np.array([0, 1, 0])

  # Check if the output is close to the expected output
  assert np.allclose(v_rotated, v_expected), "Test case failed"

  print("Test case passed")


def test_parallel_transport():
  """
  This function tests the parallel_transport function by checking if the
  transported vector is orthogonal to the new tangent vector.
  """

  # Test case 1: Orthogonal tangents
  u = np.array([1, 0, 0])
  t1 = np.array([0, 1, 0])
  t2 = np.array([0, 0, 1])
  u_transported = parallel_transport(u, t1, t2)
  assert np.allclose(np.dot(u_transported, t2), 0), "Test case 1 failed"
  # Returns True if two arrays are element-wise equal within a tolerance.

  # Test case 2: Parallel tangents
  u = np.array([1, 1, 1])
  t1 = np.array([1, 0, 0])
  t2 = np.array([2, 0, 0])
  u_transported = parallel_transport(u, t1, t2)
  assert np.allclose(u_transported, u), "Test case 2 failed"

  print("All test cases passed")


def test_getRefTwist():
  """
  This function tests the getRefTwist function by comparing the output
  with expected values for specific test cases.
  """

  # Test case 1: Straight rod
  d1 = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
  tangent = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
  refTwist = np.zeros(4)
  refTwist_calculated = getRefTwist(d1, tangent, refTwist)
  assert np.allclose(refTwist_calculated, np.zeros(4)), "Test case 1 failed"

  # Test case 2: Twisted rod (90 degrees twist)
  d1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
  tangent = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
  refTwist = np.zeros(4)
  refTwist_calculated = getRefTwist(d1, tangent, refTwist)
  assert np.allclose(refTwist_calculated,
                      np.array([0, np.pi/2, np.pi/2, 0])), "Test case 2 failed"

  print("All test cases passed")


def test_computekappa():
  """
  This function tests the computekappa function by comparing the output with
  expected values for a specific test case.
  """

  # Test case 1: Straight rod with no curvature
  node0 = np.array([0, 0, 0])
  node1 = np.array([1, 0, 0])
  node2 = np.array([2, 0, 0])
  m1e = np.array([0, 1, 0])
  m2e = np.array([0, 0, 1])
  m1f = np.array([0, 1, 0])
  m2f = np.array([0, 0, 1])

  kappa_calculated = computekappa(node0, node1, node2, m1e, m2e, m1f, m2f)

  # Expected output: zero curvature for a straight rod
  kappa_expected = np.array([0, 0])

  # Check if the calculated curvature is close to the expected curvature
  assert np.allclose(kappa_calculated, kappa_expected), "Test case failed"

  # Test case 2: A rod with inifinite curvature (180 degrees turn)
  node0 = np.array([0, 0, 0])
  node1 = np.array([1, 0, 0])
  node2 = np.array([0, 0, 0])
  m1e = np.array([0, 1, 0])
  m2e = np.array([0, 0, 1])
  m1f = np.array([0, 1, 1])
  m2f = np.array([0, 0, -1])

  kappa_calculated = computekappa(node0, node1, node2, m1e, m2e, m1f, m2f)

  # Check for NaN values
  assert np.isnan(kappa_calculated[0]), "NaN curvature check 1 failed"
  assert np.isnan(kappa_calculated[1]), "NaN curvature check 2 failed"

  print("Test case passed")


def test_assignOrthonormalAdaptedFrame():
  """
  This function tests the assignOrthonormalAdaptedFrame function by checking
  that (d1, d2, t) is an orthonormal, right-handed frame.
  """

  # Test case 1: tangent along x (expected d1 = [0, 1, 0], d2 = [0, 0, 1])
  t = np.array([1.0, 0.0, 0.0])
  d1, d2 = assignOrthonormalAdaptedFrame(t)
  assert np.allclose(d1, [0, 1, 0]), "Test case 1 failed (d1)"
  assert np.allclose(d2, [0, 0, 1]), "Test case 1 failed (d2)"

  # Test cases 2-4: general tangent, and tangents along +z and -z (fallback case)
  for t in [np.array([1.0, 2.0, 3.0]) / np.sqrt(14), np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])]:
    d1, d2 = assignOrthonormalAdaptedFrame(t)
    assert np.isclose(np.linalg.norm(d1), 1) and np.isclose(np.linalg.norm(d2), 1), "Unit length failed"
    assert np.isclose(np.dot(d1, t), 0) and np.isclose(np.dot(d2, t), 0), "Orthogonality to t failed"
    assert np.isclose(np.dot(d1, d2), 0), "Orthogonality d1-d2 failed"
    assert np.allclose(np.cross(d1, d2), t), "Right-handedness (d1 x d2 = t) failed"

  # Test case 5: randomize=True gives a different, but still orthonormal adapted, frame
  rng = np.random.default_rng(0)
  t = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
  d1a, d2a = assignOrthonormalAdaptedFrame(t, randomize=True, rng=rng)
  d1b, d2b = assignOrthonormalAdaptedFrame(t, randomize=True, rng=rng)
  assert not np.allclose(d1a, d1b), "Random frames should differ"
  for d1, d2 in [(d1a, d2a), (d1b, d2b)]:
    assert np.isclose(np.linalg.norm(d1), 1) and np.isclose(np.dot(d1, t), 0), "Random d1 failed"
    assert np.allclose(np.cross(d1, d2), t), "Random frame right-handedness failed"

  # Test case 6: same seed gives the same random frame
  d1c, _ = assignOrthonormalAdaptedFrame(t, randomize=True, rng=np.random.default_rng(0))
  assert np.allclose(d1c, d1a), "Seeded random frame is not reproducible"

  print("All test cases passed")


if __name__ == '__main__':
    test_signedAngle()
    test_rotateAxisAngle()
    test_parallel_transport()
    test_getRefTwist()
    test_computekappa()
    test_assignOrthonormalAdaptedFrame()
