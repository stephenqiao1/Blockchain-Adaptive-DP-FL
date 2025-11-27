import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import os
import time
import copy
import json
import matplotlib.pyplot as plt
from web3 import Web3
from blockchain_utils import deploy_contract
from adaptive_logic import AdaptivePrivacyScaler
from tqdm import tqdm

# --- SETUP ACCOUNTS ---
AGGREGATOR_PK = "0x2dbd47064581325f7d13e063dd4380b4f9220845fff2d6d72ed0fc9b36eda044"
CLIENT_1_PK   = "0xd586349d2f99dd51755d814fc37331d98f3690485e685b34f15dd36cd026ab2c"
CLIENT_2_PK   = "0xe909edfb3310839b3ac6d25c9bf5c51268ffadf24ba908939ba3826382d2055e"

RPC_URL = "http://127.0.0.1:7545"
ROUNDS = 5 # Number of Global FL Rounds
LOCAL_EPOCHS = 1 # Epochs per client per round
BATCH_SIZE = 64
LR = 0.01

# Cost Estimation Constants
ETH_PRICE_USD = 2500.0  # Approx current price
GAS_PRICE_GWEI = 20     # Standard L2 gas price

# Mac M2 Device Setup
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ Using Apple M2 (MPS) Acceleration")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ Using CPU")

# --- 2. MODEL DEFINITION ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(4, 32) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(4, 64)
        self.pool = nn.MaxPool2d(2, 2)
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
    model = ModuleValidator.fix(model)
    return model.to(DEVICE)

# --- 3. DATA LOADING ---
def get_data_loaders():
    print("⏳ Loading CIFAR-10 Data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Train Set (Subset for speed)
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Split 50,000 images between 2 clients (25k each)
    lengths = [25000, 25000]
    client_sets = random_split(train_data, lengths)
    
    train_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) for ds in client_sets]

    # Test Set (Full)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loaders, test_loader

# --- 4. TRAIN & TEST FUNCTIONS ---
def train_client(global_model_state, train_loader, sigma):
    local_model = get_model()
    local_model.load_state_dict(global_model_state)
    local_model.train()

    optimizer = optim.SGD(local_model.parameters(), lr=LR)
    privacy_engine = PrivacyEngine()
    
    local_model, optimizer, train_loader = privacy_engine.make_private(
        module=local_model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=sigma, 
        max_grad_norm=1.2,
    )

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    
    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = local_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    epsilon = privacy_engine.get_epsilon(1e-5)
    
    # Clean keys for aggregation
    clean_state = {k.replace("_module.", ""): v for k, v in local_model.state_dict().items()}
    return clean_state, avg_loss, epsilon

def test_global(model, test_loader):
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
    return 100. * correct / total

def fed_avg(model_states):
    avg_state = copy.deepcopy(model_states[0])
    for key in avg_state.keys():
        for i in range(1, len(model_states)):
            avg_state[key] += model_states[i][key]
        avg_state[key] = torch.div(avg_state[key], len(model_states))
    return avg_state

# --- 5. MAIN SYSTEM ---
def main():
    print("\n--- 🚀 FULL SYSTEM STARTING ---")
    
    # Init Blockchain
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    aggregator = w3.eth.account.from_key(AGGREGATOR_PK)
    c1_wallet = w3.eth.account.from_key(CLIENT_1_PK)
    c2_wallet = w3.eth.account.from_key(CLIENT_2_PK)
    
    print("1. Deploying Smart Contract...")
    contract = deploy_contract(aggregator.address, AGGREGATOR_PK)
    
    print("2. Registering Clients...")
    for wallet, pk in [(c1_wallet, CLIENT_1_PK), (c2_wallet, CLIENT_2_PK)]:
        tx = contract.functions.registerClient().build_transaction({
            'from': wallet.address, 'nonce': w3.eth.get_transaction_count(wallet.address), 'gas': 500000, 'gasPrice': w3.eth.gas_price
        })
        w3.eth.send_raw_transaction(w3.eth.account.sign_transaction(tx, pk).raw_transaction)

    # Init ML
    train_loaders, test_loader = get_data_loaders()
    global_model = get_model()
    scaler = AdaptivePrivacyScaler(base_sigma=1.0)
    
    # METRICS STORAGE
    history = {
        "rounds": [],
        "accuracy": [],
        "loss": [],
        "sigma": [],
        "budget_spent": [], # Cumulative
        "gas_used": []      # Per round total
    }
    
    prev_loss = 2.5
    total_budget_cumulative = 0.0

    print("\n--- STARTING TRAINING ROUNDS ---")
    
    for round_num in range(1, ROUNDS + 1):
        print(f"\n📢 [ROUND {round_num}/{ROUNDS}]")
        history["rounds"].append(round_num)
        
        # 1. Adaptive Logic
        current_sigma = scaler.adjust_sigma(prev_loss, 0)
        history["sigma"].append(current_sigma)
        print(f"   📊 Adaptive Sigma: {current_sigma:.3f}")

        # 2. Aggregator Start Round
        try:
            tx = contract.functions.startRound(f"R{round_num}").build_transaction({
                'from': aggregator.address, 'nonce': w3.eth.get_transaction_count(aggregator.address), 'gas': 500000, 'gasPrice': w3.eth.gas_price
            })
            w3.eth.send_raw_transaction(w3.eth.account.sign_transaction(tx, AGGREGATOR_PK).raw_transaction)
        except: pass

        # 3. Client Training
        client_updates = []
        round_losses = []
        round_gas = 0
        round_privacy_cost = 0

        for i, (loader, wallet, pk) in enumerate(zip(train_loaders, [c1_wallet, c2_wallet], [CLIENT_1_PK, CLIENT_2_PK])):
            # Train
            new_weights, loss, eps_cost = train_client(global_model.state_dict(), loader, current_sigma)
            client_updates.append(new_weights)
            round_losses.append(loss)
            
            if i == 0: round_privacy_cost = eps_cost # Approx cost per client

            # Blockchain Submit
            scaled_cost = int(eps_cost * 100)
            try:
                tx = contract.functions.submitHash(f"Hash{i}", scaled_cost).build_transaction({
                    'from': wallet.address, 'nonce': w3.eth.get_transaction_count(wallet.address), 'gas': 500000, 'gasPrice': w3.eth.gas_price
                })
                signed = w3.eth.account.sign_transaction(tx, pk)
                tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
                
                # CAPTURE GAS
                receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
                gas_used = receipt['gasUsed']
                round_gas += gas_used
                print(f"      Client {i+1}: Loss {loss:.2f} | Gas Used: {gas_used}")
            except Exception as e:
                print(f"      Client {i+1} Failed: {e}")

        # 4. Aggregation & Testing
        new_global_weights = fed_avg(client_updates)
        global_model.load_state_dict(new_global_weights)
        
        test_acc = test_global(global_model, test_loader)
        prev_loss = sum(round_losses) / len(round_losses)
        total_budget_cumulative += round_privacy_cost

        # Store Metrics
        history["accuracy"].append(test_acc)
        history["loss"].append(prev_loss)
        history["budget_spent"].append(total_budget_cumulative)
        history["gas_used"].append(round_gas)
        
        print(f"   ✅ Round Complete. Test Acc: {test_acc:.2f}% | Total Budget: {total_budget_cumulative:.2f}")

    # --- 6. PLOTTING & REPORT GENERATION ---
    print("\n--- 📊 GENERATING REPORT GRAPHS ---")
    
    # Save Raw Data
    with open('experiment_data.json', 'w') as f:
        json.dump(history, f)

    # Plot 1: Utility (Accuracy vs Rounds)
    plt.figure(figsize=(10, 5))
    plt.plot(history["rounds"], history["accuracy"], marker='o', label='Adaptive DP-FL')
    plt.title('Experiment A: Utility (Test Accuracy)')
    plt.xlabel('Round')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig('ExpA_Utility.png')
    print("Saved ExpA_Utility.png")

    # Plot 2: Privacy Efficiency (Accuracy vs Budget)
    plt.figure(figsize=(10, 5))
    plt.plot(history["budget_spent"], history["accuracy"], marker='s', color='green', label='Adaptive Efficiency')
    plt.title('Experiment B: Privacy Efficiency')
    plt.xlabel('Cumulative Privacy Budget (Epsilon)')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig('ExpB_Efficiency.png')
    print("Saved ExpB_Efficiency.png")

    # Plot 3: Cost Analysis (Gas)
    # Convert Gas to USD
    gas_costs_usd = [ (g * GAS_PRICE_GWEI * 1e-9 * ETH_PRICE_USD) for g in history["gas_used"] ]
    
    plt.figure(figsize=(10, 5))
    plt.bar(history["rounds"], gas_costs_usd, color='orange', alpha=0.7)
    plt.title('Experiment C: System Cost (Estimated USD)')
    plt.xlabel('Round')
    plt.ylabel('Cost ($) per Round')
    plt.grid(axis='y')
    plt.savefig('ExpC_Cost.png')
    print("Saved ExpC_Cost.png")

    print("\n✅ DONE! All graphs and data saved.")

if __name__ == "__main__":
    main()