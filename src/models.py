# models.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms

class SimpleCNN(nn.Module):
    """
    Lightweight CNN for CIFAR-10 with Differential Privacy support.
    
    Architecture:
    - Two convolutional layers
    - Two fully connected layers
    - GroupNorm instead of BatchNorm (required for DP independence)
    
    Privacy Engineering: GroupNorm satisfies the independence requirements
    of Differential Privacy samples, unlike BatchNorm which violates
    sample independence due to batch statistics.
    """
    def __init__(self, num_groups=8):
        super(SimpleCNN, self).__init__()
        # First convolutional layer: 3 input channels (CIFAR-10 RGB), 16 output channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # GroupNorm: num_groups=8, num_channels=16 (groups channels into 8 groups of 2)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=16)
        
        # Second convolutional layer: 16 input, 32 output channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # GroupNorm: num_groups=8, num_channels=32 (groups channels into 8 groups of 4)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=32)
        
        # Fully connected layers
        # After 2 max pooling (32x32 -> 16x16 -> 8x8), we have 32 * 8 * 8 = 2048 features
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # First conv block: conv -> GroupNorm -> ReLU -> MaxPool
        x = self.conv1(x)
        x = self.gn1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2, 2)  # 32x32 -> 16x16
        
        # Second conv block: conv -> GroupNorm -> ReLU -> MaxPool
        x = self.conv2(x)
        x = self.gn2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2, 2)  # 16x16 -> 8x8
        
        # Flatten and fully connected layers
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_cifar10_loaders(num_clients, batch_size=64):
    """
    Load CIFAR-10 dataset and split among clients.
    
    Args:
        num_clients: Number of federated learning clients
        batch_size: Batch size for training
        
    Returns:
        client_loaders: List of DataLoaders, one per client
        test_loader: DataLoader for test set
    """
    # CIFAR-10 normalization: mean and std for RGB channels
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
    
    # IID Split for simplicity (Use Dirichlet for Non-IID in future)
    lengths = [len(train_data) // num_clients] * num_clients
    # Handle remainder if any
    if sum(lengths) < len(train_data):
        lengths[0] += len(train_data) - sum(lengths)
        
    client_datasets = torch.utils.data.random_split(train_data, lengths)
    
    client_loaders = [torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1024, shuffle=False)
    
    return client_loaders, test_loader

# Backward compatibility alias
def get_mnist_loaders(num_clients, batch_size=64):
    """Legacy function name - redirects to CIFAR-10"""
    return get_cifar10_loaders(num_clients, batch_size)