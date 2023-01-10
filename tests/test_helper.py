import numpy as np
from flybys.helper import *


def test_normalize():
    assert np.allclose(normalize(np.array([1, 4, 4, -4])), np.array([1. / 7, 4. / 7, 4./7, -4. / 7]))


def test_find_switch():
    condition = np.array([False, False, False, True, True, True, False, True, False, False, True])
    positive, negative = find_switch(condition)
    assert (positive == np.array([3, 7, 10])).all() and (negative == np.array([6, 8])).all()
