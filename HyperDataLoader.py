import numpy as np
from typing import Dict, Tuple
from scipy.io import loadmat


class DatasetParams:
    def __init__(self, path, gt_path, key, gt_key):
        self.path = path
        self.gt_path = gt_path
        self.key = key
        self.gt_key = gt_key


class HyperDataLoader:
    def __init__(self):
        self.datasets_params: Dict[str, DatasetParams] = {
            "PaviaU": DatasetParams(
                "PaviaU.mat", "PaviaU_gt.mat", "paviaU", "paviaU_gt"
            )
        }

    def load_dataset_supervised(
        self, dataset: str, patch_size: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param Name:
        :param patch_size: for segmantation tasks
        :return: data array,ground truth lables array
        """
        data = loadmat(self.datasets_params[dataset].path)[
            self.datasets_params[dataset].key
        ]
        gt = loadmat(self.datasets_params[dataset].gt_path)[
            self.datasets_params[dataset].gt_key
        ]

        print(f"Data Shape: {data.shape[:-1]}\n"
              f"Number of Bands: {data.shape[-1]}")

        return data, gt
    def generate_vectors(self,dataset):
        data, lables = self.load_dataset_supervised(dataset)
        X=data.reshape(data.shape[0]*data.shape[1], -1)
        Y=lables.reshape(lables.shape[0]*lables.shape[1], -1)
        return X,Y
    def load_dataset_unsupervised(self, Name: str,
                                  patch_size: int = -1) -> np.ndarray:
        """

        :param Name:
        :param patch_size:
        :return:
        """
        raise NotImplementedError

def test():
    hdl = HyperDataLoader()
    data, lables = hdl.load_dataset_supervised("PaviaU")
    print(data[0].shape)
    print(data[1:3][1:3].shape)
    print(lables[1:3][1:3].shape)
    #print(data[1:3][1:3])
    print(lables.ravel())