from HyperDataLoader import HyperDataLoader

loader=HyperDataLoader()
X,y=loader.images_to_pixels("HSI-drive",(3,3),limit=10)
print(X.shape)
print(y.shape)
labled_data=loader.load_dataset_supervised("PaviaU",(5,5))
print()
print(labled_data[0].image.shape)
print(labled_data[0].lables.shape)
