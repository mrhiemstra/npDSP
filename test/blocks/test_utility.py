import numpy as np

from npdsp import *


def test_copy() -> None:
    cp = Copy()

    x = np.arange(1, 10, 100)
    y = cp(x)

    np.testing.assert_array_equal(x, y)


def test_lambda() -> None:
    lmbd = Lambda(np.cos)

    y = lmbd([0, 0, 0])

    np.testing.assert_array_equal([1, 1, 1], y)
