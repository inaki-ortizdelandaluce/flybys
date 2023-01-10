import pytest
import math
import numpy as np
from flybys.quaternion import Quaternion


def test_init():
    q = Quaternion(np.array([1.5, 2, -7.6, 0.2]))
    assert q.__str__() == '1.500 +2.000i -7.600j +0.200k'

    q = Quaternion([1.5, 2, -7.6, 0.2])
    assert q.__str__() == '1.500 +2.000i -7.600j +0.200k'

    q = Quaternion(axis=np.array([2, 0, 0]), radians=math.pi)
    assert q.__str__() == '0.000 +1.000i +0.000j +0.000k'

    pytest.raises(TypeError, Quaternion, np.array([0, 1, 2, '3']))

    q = Quaternion(axis=np.array([2, 0, 0]), degrees=180)
    assert q.__str__() == '0.000 +1.000i +0.000j +0.000k'

    m = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    q = Quaternion(matrix=m)
    assert q.__str__() == '0.707 +0.000i +0.707j +0.000k'

    th1 = math.radians(32.0)
    m1 = np.array([[math.cos(th1), -math.sin(th1), 0], [math.sin(th1), math.cos(th1), 0], [0, 0, 1]])
    q = Quaternion(matrix=m1)
    assert q.__str__() == '0.961 +0.000i +0.000j +0.276k'

    th2 = math.radians(116.0)
    m2 = np.array([[1, 0, 0], [0, math.cos(th2), -math.sin(th2)], [0, math.sin(th2), math.cos(th2)]])
    q = Quaternion(matrix=m2)
    assert q.__str__() == '0.530 +0.848i +0.000j +0.000k'

    q = Quaternion(matrix=np.matmul(m2, m1))
    assert q.__str__() == '0.509 +0.815i -0.234j +0.146k'


def test_normalize():
    q = Quaternion([1, 4, 4, -4])
    q.normalize()
    assert q == Quaternion([1. / 7, 4. / 7, 4./7, -4. / 7])


def test_conjugate():
    q1 = Quaternion([-1, -4, 4, -4])
    q2 = q1.conjugate()
    assert q2 == Quaternion([-1, 4, -4, 4])


def test_inverse():
    q = Quaternion([1, 1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(3)])
    assert q.inverse().__str__() == '0.500 -0.289i -0.289j -0.289k'


def test_multiply():
    q1 = Quaternion([2, -2, 3, -4])
    q2 = Quaternion([1, -2, 5, -6])
    assert Quaternion.multiply(q1, q2) == Quaternion([-41, -4, 9, -20])
    assert Quaternion.multiply(q2, q1) == Quaternion([-41, -8, 17, -12])


def test_rotate_vector():
    q1 = Quaternion(axis=[1 / math.sqrt(2), 0, 1 / math.sqrt(2)], degrees=90)
    v = np.array([2, 0, 0])
    assert np.allclose(Quaternion._rotate_vector(v, q1), np.array([1.000, 1.41421, 1.000]))

    # rotating clockwise same as rotating the unit quaternion's conjugate (or any quaternion's inverse)
    q2 = Quaternion(axis=[1 / math.sqrt(2), 0, 1 / math.sqrt(2)], degrees=-90)
    assert np.allclose(Quaternion._rotate_vector(v, q2), Quaternion._rotate_vector(v, q1.conjugate()))


def test_rotate():
    q = Quaternion(axis=[1 / math.sqrt(2), 0, 1 / math.sqrt(2)], degrees=90)
    assert np.allclose(q.rotate([2, 0, 0]), [1.000, 1.4142, 1.000], rtol=1e-4)

    q = Quaternion.multiply(Quaternion(axis=np.array([0, 0, 1]), degrees=32),
                            Quaternion(axis=np.array([1, 0, 0]), degrees=116))
    assert np.allclose(q.rotate(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
                       np.array([[0.8480481, 0.52991926, 0.],
                                [0.23230132, -0.37175982, 0.89879405],
                                [0.47628828, -0.76222058, -0.43837115]]), rtol=1e-7)

    # rotate unit cube with vertex at (0,0,0) 180 degrees around ZX axis at 45 degrees
    q = Quaternion(axis=[math.sqrt(2)/2, 0, math.sqrt(2)/2], degrees=180)
    vertices = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    q.rotate(vertices)
    assert np.allclose(q.rotate(vertices),
                       np.array(
                           [[0, 0, 0], [1, 0, 0], [0, -1, 0], [1, -1, 0], [0, 0, 1], [1, 0, 1], [0, -1, 1], [1, -1, 1]]))


def test_str():
    q = Quaternion()
    assert q.__str__() == '0.000 +0.000i +0.000j +0.000k'


def test_repr():
    q = Quaternion()
    assert q.__repr__() == 'Quaternion(0.0, 0.0, 0.0, 0.0)'


def test_format():
    q = Quaternion()
    assert q.__format__('+.3f') == '+0.000 +0.000i +0.000j +0.000k'

