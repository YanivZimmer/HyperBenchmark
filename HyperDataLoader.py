import os

import numpy as np
from typing import Dict, Tuple, List, Union
from scipy.io import loadmat
from HyperData.png_to_mat import png_to_array
from collections import namedtuple

Labeled_Data = namedtuple("Labeled_Data", ["image", "lables"])


class DatasetParams:
    def __init__(
        self,
        data_path: str,
        lables_path: str,
        data_key: str,
        lables_key: str,
        single_file=False,
    ):
        self.data_path = data_path
        self.lables_path = lables_path
        self.data_key = data_key
        self.gt_key = lables_key
        self.single_file = single_file


class HyperDataLoader:
    def __init__(self):
        self.datasets_params: Dict[str, DatasetParams] = {
            "PaviaU": DatasetParams(
                "./datasets/PaviaU/image/PaviaU.mat",
                "./datasets/PaviaU/labels/PaviaU_gt.mat",
                "paviaU",
                "paviaU_gt",
                True,
            ),
            "HSI-drive": DatasetParams(
                "./datasets/HSI-drive/cubes_float32",
                "./datasets/HSI-drive/labels",
                "cube_fl32",
                "M",
                False,
            ),
        }

    def file_to_mat(self, filename: str, key: str) -> np.ndarray:
        if filename.endswith("png"):
            return png_to_array(filename)
        return loadmat(filename)[key]

    def load_singlefile_supervised(self, dataset_param: DatasetParams) -> Labeled_Data:
        """
        Just parameter overloading wrapper for using load_one_supervised with dataParams
        :param dataset_param:
        :return: Labeled data
        """
        return self.load_one_supervised(
            dataset_param.data_path,
            dataset_param.lables_path,
            dataset_param.data_key,
            dataset_param.gt_key,
        )

    def load_dataset_supervised(
        self, dataset_name: str, patch_size: int = 1
    ) -> List[Labeled_Data]:
        if self.datasets_params[dataset_name].single_file:
            return [self.load_singlefile_supervised(self.datasets_params[dataset_name])]
        labeled_data_list = []
        datafiles = os.listdir(self.datasets_params[dataset_name].data_path)
        for lablefile in os.listdir(self.datasets_params[dataset_name].lables_path):
            base_name = os.path.splitext(os.path.basename(lablefile))[0]
            data_files = list(filter(lambda a: a.startswith(base_name), datafiles))
            if len(data_files) == 0:
                raise AttributeError(f"no data for {base_name}")
            labled_img = self.load_one_supervised(
                os.path.join(
                    self.datasets_params[dataset_name].data_path, datafiles[0]
                ),
                os.path.join(
                    self.datasets_params[dataset_name].lables_path,
                    lablefile,
                ),
                self.datasets_params[dataset_name].data_key,
                self.datasets_params[dataset_name].gt_key,
            )
            labeled_data_list.append(labled_img)
        return labeled_data_list

    def load_one_supervised(
        self,
        datafile: str,
        lablefile: str,
        datakey: Union[str, None],
        labelkey: Union[str, None],
        patch_size: int = 1,
    ) -> Labeled_Data:
        """
        :param Name:
        :param patch_size: for segmantation tasks
        :return: data array,ground truth lables array
        """
        data = self.file_to_mat(datafile, datakey)
        gt = self.file_to_mat(lablefile, labelkey)
        # TODO- patch
        print(
            f"Data Shape: {data.shape}"
        )  # [:-1]}\n" f"Number of Bands: {data.shape[-1]}")
        return Labeled_Data(data, gt)

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
    pavia = hdl.load_dataset_supervised("PaviaU")
    lables = pavia[0].lables
    data = pavia[0].image
    print(data[0].shape)
    print(data[1:3][1:3].shape)
    print(lables[1:3][1:3].shape)
    print(data[1:3][1:3])
    print(data[:, :, 55].shape, data[:, :, 55])

    # HSI-drive
    images = hdl.load_dataset_supervised("HSI-drive")
    data = images[0].image
    lables = images[0].lables
    print(data[:, 0, 0])
    print(lables[0][0])
    print(lables.ravel())


test()
