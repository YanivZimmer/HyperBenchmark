from typing import Callable, Dict

import numpy as np
from keras import Model

from Experiments.Assesment import Assesment
from selection.max_dissim_selector import MaxDissimSelector


class MaxDissm(Assesment):
    def __init__(
        self,
        model_creator: Callable[[int], Model],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):
        super().__init__(model_creator, X_train, y_train, X_test, y_test)
        candidates = self.generate_candidates()
        vec_size = next(iter(candidates.values())).shape[0]
        self.max_dissm_selector = MaxDissimSelector(vec_size,candidates)

    def generate_candidates(self) -> Dict[int, np.ndarray]:
        candidates = {}
        for band_idx in range(self.X_train.shape[1]):
            candidates[band_idx] = self.X_train[:, band_idx].flatten()
        return candidates
