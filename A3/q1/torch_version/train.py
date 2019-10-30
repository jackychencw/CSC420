import torch
import torch.optim as optim
from torch.optim import optimizer
from torch.utils.data import DataLoader
from torch.autograd import Variable
import dataset
import unet
import torch.nn as nn
import numpy as np
TRAIN_DATA_PATH = './cat_data/Train/'

def train(device, loss_type=1, n_channels = 64, kernel_size = 3, n_classes = 2, lr = 0.01, momentum = 0.9, batch_size = 10, epochs=100):
    model = unet.UNet(n_channels = n_channels, n_classes = n_classes)
    model.to(device=device)
    train_data = dataset.CatDataset(TRAIN_DATA_PATH)
    if loss_type == 1:
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum)
    trainloader = DataLoader(train_data, batch_size = batch_size, shuffle=True)
    for e in range(epochs):
        model.train()
        running_loss = 0
        total_train = 0
        correct_train = 0
        for images, labels in trainloader:
            images.to(device=device)
            labels.to(device=device)
            pred = model(images)

            loss = criterion(pred, labels)
            loss = Variable(loss, requires_grad=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            correct_train += (pred.data.eq(labels.data)).sum().item()
            total_train += pred.nelement()
        accuracy = correct_train/total_train
        print(f"Epoch Iteration: {e}\nTraining loss: {running_loss/len(trainloader)}\nAccuracy: {accuracy}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.version.cuda)
    print(f'Using device {device}')
    
    train(device)
