from HyperDataLoader import HyperDataLoader

loader = HyperDataLoader()
X, y = loader.images_to_pixels("HSI-drive", (3, 3), True, limit=10)
print(X.shape)
print(y.shape)
X1, y1 = loader.images_to_pixels("PaviaU", (1, 1), False)
print(X1.shape)
print(y1.shape)
