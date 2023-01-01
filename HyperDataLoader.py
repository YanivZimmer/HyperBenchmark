import os

import numpy as np
from typing import Dict, Tuple, List, Union
from scipy.io import loadmat
from HyperData.png_to_mat import png_to_array
from collections import namedtuple
Labeled_Data=namedtuple('Labeled_Data',['image','lables'])

class DatasetParams:
    def __init__(
        self, data_folder: str, lables_folder: str, data_key: str, lables_key: str
    ):
        self.data_folder = data_folder
        self.lables_folder = lables_folder
        self.data_key = data_key
        self.gt_key = lables_key


class HyperDataLoader:
    def __init__(self):
        self.datasets_params: Dict[str, DatasetParams] = {
            "PaviaU": DatasetParams(
                "PaviaU.mat", "PaviaU_gt.mat", "paviaU", "paviaU_gt"
            ),
            "HSI-drive": DatasetParams(
                "./datasets/HSI-drive/cubes_float32",
                "./datasets/HSI-drive/labels",
                "cube_fl32",
                "M",
            ),
        }

    def file_to_mat(self, filename: str, key: str) -> np.ndarray:
        if filename.endswith("png"):
            return png_to_array(filename)
        return loadmat(filename)[key]

    def load_dataset_supervised(
        self, dataset_name: str, patch_size: int = 1
    ) -> List[Labeled_Data]:
        labeled_data_list = []
        print(os.getcwd())
        datafiles = os.listdir(self.datasets_params[dataset_name].data_folder)
        for lablefile in os.listdir(self.datasets_params[dataset_name].lables_folder):
            base_name = os.path.splitext(os.path.basename(lablefile))[0]
            data_files = list(filter(lambda a: a.startswith(base_name), datafiles))
            if len(data_files) == 0:
                raise AttributeError(f"no data for {base_name}")
            data,lables = self.load_one_supervised(
                os.path.join(
                    self.datasets_params[dataset_name].data_folder, datafiles[0]
                ),
                os.path.join(
                    self.datasets_params[dataset_name].lables_folder,
                    lablefile,
                ),
                self.datasets_params[dataset_name].data_key,
                self.datasets_params[dataset_name].gt_key,
            )
            labeled_data_list.append(Labeled_Data(data,lables))
        return labeled_data_list

    def load_one_supervised(
        self,
        datafile: str,
        lablefile: str,
        datakey: Union[str, None],
        labelkey: Union[str, None],
        patch_size: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param Name:
        :param patch_size: for segmantation tasks
        :return: data array,ground truth lables array
        """
        data = self.file_to_mat(datafile, datakey)
        gt = self.file_to_mat(lablefile, labelkey)
        # TODO- patch
        print(f"Data Shape: {data.shape}")#[:-1]}\n" f"Number of Bands: {data.shape[-1]}")
        return data, gt

    def generate_vectors(self, dataset):
        data, lables = self.load_dataset_supervised(dataset)
        X = data.reshape(data.shape[0] * data.shape[1], -1)
        Y = lables.reshape(lables.shape[0] * lables.shape[1], -1)
        return X, Y

    def load_dataset_unsupervised(self, Name: str, patch_size: int = -1) -> np.ndarray:
        """

        :param Name:
        :param patch_size:
        :return:
        """
        raise NotImplementedError


def test():
    hdl = HyperDataLoader()
    images = hdl.load_dataset_supervised("HSI-drive")
    #print(data[0].shape)
    #print(data[1:3][1:3].shape)
    #print(lables[1:3][1:3].shape)
    # print(data[1:3][1:3])
    # HSI-drive
    data=images[0].image
    lables=images[0].lables
    print(data[:, 0, 0])
    print(lables[0][0])
    print(lables.ravel())


test()
