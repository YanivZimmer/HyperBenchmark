import spectral.io.envi as envi
from pathlib import Path
from hyper_data_loader.HyperDataLoader import HyperDataLoader
from models.mlp.mlp import MlpModel


def hdr_to_data(folder_path):
    datapath = Path(folder_path)
    header_file = str(datapath / 'raw.hdr')
    spectral_file = str(datapath / 'raw')
    numpy_ndarr = envi.open(header_file, spectral_file)
    return numpy_ndarr

if __name__=='__main__':
    Brain_Bands=826
    path='../../datasets/brain/0008-1'
    loader = HyperDataLoader()
    data=loader.hdr_to_data(path)
    train_loader, test_loader = loader.data_loaders_unsupervised(data,None)
    mlp = MlpModel(Brain_Bands, Brain_Bands)



