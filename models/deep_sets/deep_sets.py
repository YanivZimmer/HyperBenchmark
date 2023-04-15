import torch
import torch.nn as nn
from torch import optim
from torchsummary import summary
import torch.nn.functional as F
class DeepSets(nn.Module):
    def __init__(self, in_channels, out_channels, num_hidden, classes_num):
        super(DeepSets, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.regressor = nn.Sequential(
            nn.Linear(out_channels, num_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(num_hidden, classes_num)
        )

    def forward(self, x):
        # x has shape (batch_size, num_elements, in_channels, h, w)
        # We want to apply the same feature extractor to each element
        # output will have shape (batch_size, out_channels, num_elements)

        b, e, c, h, w = x.shape
        x_reshape = x.view(b * e, c, h, w)

        features = self.feature_extractor(x_reshape)
        features = features.view(b, e, -1)
        features = features.mean(dim=1)

        output = self.regressor(features)

        return output


def train_model(model, train_loader,epochs,lr, device):
    # Set model to training mode
    model.train()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    #0.00005
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)#optim.Adam(model.parameters())

    # Move model to device
    model.to(device)
    for epoch in range(epochs):
        # Train the model for one epoch
        for i, (inputs, labels) in enumerate(train_loader):
            # Move inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Print loss every 1000 iterations
            if (i+1) % 250 == 0:
                print('Epoch {} Iteration [{}/{}], Loss: {:.4f}'.format(epoch+1,i+1, len(train_loader), loss.item()))

    # Print message when training is complete
    print('Finished training')

def simple_test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy on test set: {:.2f}%'.format(accuracy))

def test_model(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
