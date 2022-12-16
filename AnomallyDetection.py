import numpy as np
from sklearn.covariance import EllipticEnvelope
heatmap = np.load('data.npy') # load
print(heatmap)
# instantiate model
ee = EllipticEnvelope(contamination = 0.1)#, ensure_min_features=2)#support_fraction=0.7)
# fit model
results = ee.fit_predict(heatmap)
print(results)