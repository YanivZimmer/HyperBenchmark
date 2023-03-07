import numpy as np


class LinearDiss:
    def calc_dissimilarity(self, space_matrix, candidate):
        approx_repr = self.calc_aprrox_linear_repr(space_matrix, candidate)
        approx_values = np.matmul(space_matrix, approx_repr)
        error = self.distance(approx_values, candidate)
        return error

    def calc_aprrox_linear_repr(self, space_matrix, candidate):
        return np.linalg.lstsq(space_matrix, candidate, rcond=None)[0]

    def distance(self, v1, v2):
        return np.linalg.norm(v1 - v2)
