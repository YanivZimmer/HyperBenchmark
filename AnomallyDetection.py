import numpy as np
from sklearn.covariance import EllipticEnvelope

heatmap = np.load("data.npy")  # load
heatmap_norm = np.load("data_norm.npy")  # load
heatmap_real = np.load("datareal.npy").T  # load

print(heatmap)
print(heatmap_norm)
print(heatmap_real)
# instantiate model
ee = EllipticEnvelope(
    contamination=0.1
)  # , ensure_min_features=2)#support_fraction=0.7)
# fit model
results = ee.fit_predict(heatmap)
print(results)
ee = EllipticEnvelope(
    contamination=0.1
)  # , ensure_min_features=2)#support_fraction=0.7)
results_norm = ee.fit_predict(heatmap_norm)
print(results_norm)

ee = EllipticEnvelope(
    contamination=0.1
)  # , ensure_min_features=2)#support_fraction=0.7)
results_real = ee.fit_predict(heatmap_real)
print(results_real)


def to_indx(arr):
    indices = []
    for i in range(len(arr)):
        if arr[i] == -1:
            indices.append(i)
    return tuple(indices)


print(to_indx(results))
print(to_indx(results_norm))
print(to_indx(results_real))
