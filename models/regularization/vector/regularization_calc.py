import math

import torch
from scipy import stats


class RegularizationDiag:
    def __init__(self, weights):
        self.weights = weights

    def _scale_weight_by_normal_dist_prob(self, mean=0, divg=1):
        dist = stats.norm(mean, divg)
        non_negative = (
            torch.Tensor(torch.sqrt(torch.pow(self.weights, 2))).cpu().detach().numpy()
        )
        prob = dist.cdf(non_negative)
        scale = prob - 0.5
        return torch.tensor(scale)

    def regularization_normal_dist_based(self, target: int, mean=0, divg=1):
        scale = self._scale_weight_by_normal_dist_prob(mean, divg)
        distance_from_mean = torch.norm(2 * (scale - 0.5),1)
        #return distance_from_mean
        return 1e-14 * (math.pow(distance_from_mean - target, 8))
