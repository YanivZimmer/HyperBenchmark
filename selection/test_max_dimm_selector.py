import numpy as np
import pytest

from max_dissim_selector import MaxDissimSelector

VEC_SIZE = 4


@pytest.fixture
def bands_mapping():
    return {
        "1": np.array([1.0, 1.0, 0.0, 1.0]).T,
        "2": np.array([0.0, 0.0, 1.0, 0.0]).T,
        "3": np.array([0.0, 0.0, 1.0, 0.0]).T,
        "4": np.array([1.0, 0.0, 0.0, 1.0]).T,
    }


@pytest.fixture
def selector(bands_mapping):
    return MaxDissimSelector(VEC_SIZE, bands_mapping)


def test_first_selected(selector):
    assert selector.select_best_band() == "4"
    assert selector.select_best_band() == "1"
