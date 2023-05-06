from hyper_data_loader.HyperDataLoader import HyperDataLoader
import os
import mat73

if __name__ == '__main__':
    loader = HyperDataLoader()
    path = "../../datasets/bgu/bguCAMP_0514-1659.mat"
    #data = loader.load_dataset_unsupervised(path, "mat1")
    data_dict = mat73.loadmat(path)
    print(data_dict)

