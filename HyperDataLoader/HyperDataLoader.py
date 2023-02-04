import math
import os

import numpy as np
from typing import Dict, Tuple, List, Union
from scipy.io import loadmat
from HyperDataLoader.png_to_mat import png_to_array
from collections import namedtuple
from sklearn.feature_extraction.image import extract_patches_2d
from tensorflow.keras.utils import to_categorical

Labeled_Data = namedtuple("Labeled_Data", ["image", "lables"])


class DatasetParams:
    def __init__(
        self,
        data_path: str,
        lables_path: str,
        data_key: str,
        lables_key: str,
        single_file=False,
        tranpose=False,
    ):
        self.data_path = data_path
        self.lables_path = lables_path
        self.data_key = data_key
        self.gt_key = lables_key
        self.single_file = single_file
        self.transpose = tranpose


class HyperDataLoader:
    def __init__(self):
        self.datasets_params: Dict[str, DatasetParams] = {
            "PaviaU": DatasetParams(
                "../datasets/PaviaU/image/PaviaU.mat",
                "../datasets/PaviaU/labels/PaviaU_gt.mat",
                "paviaU",
                "paviaU_gt",
                True,
                False,
            ),
            "HSI-drive": DatasetParams(
                "../datasets/HSI-drive/cubes_float32",
                "../datasets/HSI-drive/labels",
                "cube_fl32",
                "M",
                False,
                True,
            ),
        }

    def filter_unlabeled(self, X, y):
        # idx = np.argsort(y)
        idx = np.where(y != 0)
        y = y[idx[0]]
        X = X[idx[0], :]
        # Make lables 0-9
        y -= 1
        return X, y

    def images_to_pixels(
        self,
        dataset: str,
        patch_shape: Tuple[int, int],
        filter_unlabeled=False,
        limit=None,
    ):
        labeled_data = self.load_dataset_supervised(dataset, patch_shape, limit=limit)
        X, y = labeled_data[0].image, labeled_data[0].lables
        y = y.reshape(y.shape[0] * y.shape[1])
        for item in labeled_data[1:]:
            X = np.concatenate((X, item.image))
            y = np.concatenate(
                (y, item.lables.reshape(item.lables.shape[0] * item.lables.shape[1]))
            )
        if filter_unlabeled:
            X, y = self.filter_unlabeled(X, y)
        y = to_categorical(y, num_classes=10)
        return X, y

    def file_to_mat(self, filename: str, key: str) -> np.ndarray:
        if filename.endswith("png"):
            return png_to_array(filename)
        return loadmat(filename)[key]

    def patch_to_pad(self, patch_size: int):
        return (math.floor((patch_size - 1) / 2), math.ceil((patch_size - 1) / 2))

    def patches_factory(
        self, data: np.ndarray, patch_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Generate patches in patch shape surrounding each pixel. for pixels too close to image borders using 0 padding.
        Note: order is preserved.
        Shape transition: (x,y,z)->(x*y,patch_shape[0],patch_shape[1],z)
        :param data:
        :param patch_shape:
        :return:
        """
        padding_first = self.patch_to_pad(patch_shape[0])
        padding_second = self.patch_to_pad(patch_shape[1])
        data = np.pad(
            data,
            (padding_first, padding_second, (0, 0)),
            mode="constant",
            constant_values=0,
        )
        data = extract_patches_2d(data, patch_shape)
        return data

    def load_singlefile_supervised(
        self, dataset_param: DatasetParams, patch_shape: Tuple[int, int]
    ) -> Labeled_Data:
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
            dataset_param.transpose,
            patch_shape,
        )

    def load_dataset_supervised(
        self, dataset_name: str, patch_shape: Tuple[int, int], limit=float("inf")
    ) -> List[Labeled_Data]:
        if self.datasets_params[dataset_name].single_file:
            return [
                self.load_singlefile_supervised(
                    self.datasets_params[dataset_name], patch_shape
                )
            ]
        labeled_data_list = []
        datafiles = os.listdir(self.datasets_params[dataset_name].data_path)
        for count, lablefile in enumerate(
            os.listdir(self.datasets_params[dataset_name].lables_path)
        ):
            if count > limit:
                break
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
                self.datasets_params[dataset_name].transpose,
                patch_shape,
            )
            labeled_data_list.append(labled_img)
        return labeled_data_list

    def load_one_supervised(
        self,
        datafile: str,
        lablefile: str,
        datakey: Union[str, None],
        labelkey: Union[str, None],
        transpose: bool,
        patch_shape: Tuple[int, int],
    ) -> Labeled_Data:
        """
        :param Name:
        :return: data array,ground truth labels array
        """
        data = self.file_to_mat(datafile, datakey)
        gt = self.file_to_mat(lablefile, labelkey)
        if transpose:
            data = data.T
            gt = gt.T
        print(f"Data Shape: {data.shape}")
        data = self.patches_factory(data, patch_shape)
        return Labeled_Data(data, gt)

    def generate_vectors(
        self, dataset: str, patch_shape: Tuple[int, int]
    ) -> List[Labeled_Data]:
        vectors_list = []
        labled_data = self.load_dataset_supervised(dataset, patch_shape)
        for item in labled_data:
            X = item.image.reshape(item.image.shape[0] * item.image.shape[1], -1)
            Y = item.lables.reshape(item.lables.shape[0] * item.lables.shape[1], -1)
            vectors_list.append(Labeled_Data(X, Y))
        return vectors_list

    def load_dataset_unsupervised(self, Name: str) -> np.ndarray:
        """

        :param Name:
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


# test()
