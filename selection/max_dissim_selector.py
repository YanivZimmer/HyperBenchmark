from typing import Dict

import numpy as np
from linear_diss import LinearDiss


class MaxDissimSelector:
    def __init__(self, vector_size: int, bands_vector_mapping: Dict[int, np.ndarray]):
        self.selected_bands = []
        self.space_matrix = np.ones((vector_size, 1), dtype=float)
        self.linear_diss_calc = LinearDiss()
        self.bands_vector_mapping = bands_vector_mapping

    def band_score(self, band_vector):
        return self.linear_diss_calc.calc_dissimilarity(self.space_matrix, band_vector)

    def increase_space_matrix(self, band_vec):
        self.space_matrix = np.hstack(
            (self.space_matrix, band_vec.reshape(self.space_matrix.shape[0], 1))
        )

    def select_best_band(self):
        bands_score = {}
        for band_key, band_vec in self.bands_vector_mapping.items():
            bands_score[band_key] = self.band_score(band_vec)
        max_band_idx = max(bands_score, key=bands_score.get)
        self.selected_bands.append(max_band_idx)
        self.increase_space_matrix(self.bands_vector_mapping[max_band_idx])
        del self.bands_vector_mapping[max_band_idx]

        return max_band_idx

    def select_all_best_bands(self):
        pass
