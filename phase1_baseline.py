import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import matplotlib.pyplot as plt
import numpy as np

# Configuration
BATCH_SIZE = 64
MAX_GRAD_NORM = 1.2
LEARNING_RATE = 0.01
EPOCHS = 10
DELTA = 1e-5 # Standard delta is 1/N

# --- Mac M2 Device Selection ---
# We check for MPS (Metal Performance Shaders) availability
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ Success: Using Apple M4 (MPS) acceleration.")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ Warning: MPS not found. Falling back to CPU.")

# Simple CNN Model Definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 3 input channels (CIFAR-10), 32 output channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(4, 32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(4, 64)

        self.pool = nn.MaxPool2d(2, 2)

        # Flatten size calculation:
        # Image starts 32x32 -> Pool -> 16x16 -> Pool -> 8x8
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.gn1(self.conv1(x))))
        x = self.pool(self.relu(self.gn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def get_model():
    model = SimpleCNN()
    # Validate and modify model for DP compatibility
    model = ModuleValidator.fix(model)
    return model.to(DEVICE)

# Federated Data Loader
def get_cifar10_loaders(num_shards=10):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download CIFAR-10 dataset
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Split training data into shards (simulate clients)
    shard_size = len(train_data) // num_shards
    lengths = [shard_size] * num_shards
    # Handle remainder if dataset isn't perfectly divisible
    if sum(lengths) < len(train_data):
        lengths[-1] += len(train_data) - sum(lengths)

    # Ensures that every client gets a random variety of data (IID)
    train_subsets = random_split(train_data, lengths)

    # Create Loaders
    train_loaders = [DataLoader(sub, batch_size=BATCH_SIZE, shuffle=True) for sub in train_subsets]

    # Single test loader for global evaluation
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    return train_loaders, test_loader

# Centralized DP Training Routine
# This function trains the model while ensuring differential privacy
def train_centralized_dp(target_epsilon, train_loaders, test_loader):
    model = get_model()
    optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine()

    # We create a new single loader for the centralized baseline to make Opacus attachment easier
    full_dataset = torch.utils.data.ConcatDataset([loader.dataset for loader in train_loaders])
    global_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Attach Privacy Engine
    model, optimizer, global_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=global_loader,
        epochs=EPOCHS,
        target_epsilon=target_epsilon,
        target_delta=DELTA,
        max_grad_norm=MAX_GRAD_NORM,
    )

    accuracy_history = []

    print(f"\n--- Training with Target Epsilon: {target_epsilon} ---")

    for epoch in range(EPOCHS):
        model.train()
        for i, (data, target) in enumerate(global_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Evaluate
        acc = test(model, test_loader)
        accuracy_history.append(acc)

        # Check privacy budget actually spent
        epsilon_spent = privacy_engine.get_epsilon(DELTA)
        print(f"Epoch {epoch+1}/{EPOCHS}, Test Accuracy: {acc:.2f}%, Epsilon Spent: {epsilon_spent:.2f}")

    return accuracy_history

# Utility Functions
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = 100 * correct / total

    return accuracy

if __name__ == "__main__":
    # Prepare Data
    train_loaders, test_loader = get_cifar10_loaders(num_shards=10)

    # Define Experiments (Epsilon values to test)
    # High Epsilon = Low Privacy / High Utility
    # Low Epsilon = High Privacy / Low Utility
    epsilons = [2, 10, 50]
    results = {}

    # Run Training
    for eps in epsilons:
        acc_hist = train_centralized_dp(eps, train_loaders, test_loader)
        results[eps] = acc_hist

    # Plot Results
    plt.figure(figsize=(10, 6))
    for eps, accs in results.items():
        plt.plot(range(1, EPOCHS + 1), accs, label=f"Epsilon: {eps}")

    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Performance of Centralized DP Baseline (CIFAR-10)')
    plt.legend()
    plt.grid(True)
    plt.savefig('phase1_result.png')
    plt.show()
    print("Plot saved as phase1_result.png")