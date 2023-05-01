import torch
from torch import nn


def train_model(model, train_loader,epochs,lr, device,regularization=False):
    # Set model to training mode
    model.train()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    #0.00005
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)#optim.Adam(model.parameters())

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
            if regularization:
                loss += model.regularization()

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Print loss every 1000 iterations
            if (i+1) % 50 == 0:
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
    return accuracy
