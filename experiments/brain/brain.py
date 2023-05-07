from pathlib import Path

import spectral.io.envi as envi
import torch

from hyper_data_loader.HyperDataLoader import HyperDataLoader
from models.autoencoder import Autoencoder
from models.mlp.mlp import MlpModel
from models.utils.train_test import train_model,simple_test_model
from models.utils.device_utils import get_device
from models.utils.sid_loss import sid_loss, SIDLoss


def hdr_to_data(folder_path):
    datapath = Path(folder_path)
    header_file = str(datapath / "raw.hdr")
    spectral_file = str(datapath / "raw")
    numpy_ndarr = envi.open(header_file, spectral_file)
    return numpy_ndarr


if __name__ == "__main__":
    Brain_Bands = 826
    path = "../../datasets/brain/0008-1"
    loader = HyperDataLoader()
    data = loader.hdr_to_data(path)
    train_loader, test_loader = loader.data_loaders_unsupervised(data, None)
    autoencoder = Autoencoder(Brain_Bands, int(Brain_Bands / 10))
    train_model(
        model=autoencoder,
        train_loader=train_loader,
        epochs=100,
        lr=0.00001,
        device=get_device(),
        regularization=False,
        criterion=SIDLoss(),
        supervised=False
    )
