from models.regularization.regularization_mlp import RegMlpModel
from models.utils.sid_loss import ZeroLoss
from models.utils.train_test import train_model, simple_test_model
from pavia_university import data_loaders, device, NUM_CLASSES_PAVIA, INPUT_SHAPE_PAVIA


def main(bands):
    train_loader, test_loader = data_loaders(bands)
    mlp = RegMlpModel(INPUT_SHAPE_PAVIA, NUM_CLASSES_PAVIA, 1e-4)
    train_model(
        mlp, train_loader, epochs=75, lr=0.000025, device=device, regularization=True
    )
    simple_test_model(mlp, test_loader, device=device)


if __name__ == "__main__":
    main(None)
