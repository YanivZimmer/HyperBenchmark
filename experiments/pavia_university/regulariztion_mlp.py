from models.regularization.regularization_mlp import RegMlpModel
from models.utils.train_test import train_model, simple_test_model
from all_bands_per_algo import data_loaders, device, NUM_CLASSES_PAVIA,INPUT_SHAPE_PAVIA


def main(bands):
    train_loader, test_loader = data_loaders(bands)
    mlp = RegMlpModel(INPUT_SHAPE_PAVIA, NUM_CLASSES_PAVIA)
    train_model(mlp, train_loader, epochs=100, lr=0.000025, device=device,regularization=True)
    simple_test_model(mlp, test_loader, device=device)

if __name__=='__main__':
    main(None)
