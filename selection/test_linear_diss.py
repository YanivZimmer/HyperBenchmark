import numpy as np

from linear_diss import LinearDiss
import pytest


@pytest.fixture
def linear_diss_calc() -> LinearDiss:
    return LinearDiss()


def test_distance(linear_diss_calc):
    a = np.zeros((1, 5))
    b = np.ones((1, 5))
    assert linear_diss_calc.distance(a, b) == 2.23606797749979


def test_repr(linear_diss_calc):
    arr = np.arange(1, 6)
    res = linear_diss_calc.calc_aprrox_linear_repr(np.identity(5), np.arange(1, 6))
    assert np.array_equal(res, arr)
    space_mat = np.array([[1, 2], [3, 4]])
    res2 = linear_diss_calc.calc_aprrox_linear_repr(space_mat, np.array([1, 2]))
    mul = np.matmul(space_mat, res2)
    assert np.array_equal(mul, np.array([1, 2]))


def test_dissim(linear_diss_calc):
    space_mat = np.array([[1, 2], [0, 0]])
    candidate = np.array([1,0.5])
    error = linear_diss_calc.calc_dissimilarity(space_mat, candidate)
    assert error == 0.5
