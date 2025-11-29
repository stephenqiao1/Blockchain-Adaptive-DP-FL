# models.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1) # MNIST is 1 channel
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_mnist_loaders(num_clients, batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # IID Split for simplicity (Use Dirichlet for Non-IID in future)
    lengths = [len(train_data) // num_clients] * num_clients
    # Handle remainder if any
    if sum(lengths) < len(train_data):
        lengths[0] += len(train_data) - sum(lengths)
        
    client_datasets = torch.utils.data.random_split(train_data, lengths)
    
    client_loaders = [torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1024, shuffle=False)
    
    return client_loaders, test_loader