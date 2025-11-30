# Blockchain-Adaptive-DP-FL

A federated learning system that combines blockchain coordination, IPFS storage, and adaptive differential privacy to enable trustless, privacy-preserving collaborative machine learning.

## 🎯 Project Overview

This project implements a **Blockchain-based Federated Learning (BC-FL) system with Adaptive Differential Privacy** that eliminates the need for a trusted central aggregator. The system uses:

- **Ethereum Smart Contracts** (via Ganache) for decentralized coordination
- **IPFS** for distributed model storage
- **Adaptive Differential Privacy** to optimize the privacy-utility trade-off
- **PyTorch & Opacus** for differentially private training

### Key Innovation

The system replaces the traditional central FL server with a blockchain-based coordination protocol, while implementing an adaptive noise mechanism that dynamically adjusts privacy parameters based on training progress, optimizing both model utility and privacy budget consumption.

## 🏗️ System Architecture

### Three-Layer Architecture

1. **Compute Layer (Client-Side)**
   - Local model training with PyTorch
   - Adaptive noise injection using Opacus
   - Gradient clipping and privacy accounting
   - IPFS upload/download operations

2. **Storage Layer (Off-Chain)**
   - IPFS for decentralized model storage
   - Content-addressed storage (CID-based)
   - Tamper-proof model versioning

3. **Coordination Layer (On-Chain)**
   - `FLRegistry.sol` smart contract (Ethereum/Ganache)
   - Round management and state tracking
   - Privacy budget enforcement
   - Client reputation system

## 📋 Features

### Privacy Mechanisms

- **Centralized DP (CDP)**: Server-side noise addition (better utility, requires trusted server)
- **Local DP (LDP)**: Client-side noise addition (trustless, higher privacy cost)
- **Adaptive DP**: Dynamic noise adjustment based on validation loss
- **Adaptive Clipping**: Gradient norm-based clipping threshold adaptation

### Blockchain Features

- Client registration and reputation tracking
- Privacy budget enforcement per client
- Round-based state machine
- IPFS hash verification
- Duplicate submission prevention

### Evaluation Metrics

- **Accuracy**: Model performance on test set
- **Privacy Budget (ε)**: Cumulative epsilon consumption
- **Convergence**: Rounds to reach threshold accuracy
- **Round Latency**: System overhead (blockchain + IPFS)
- **Cost Analysis**: Gas usage and USD costs (Mainnet vs L2)

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Node.js (for Ganache)
- Solidity compiler (via Foundry)

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Install Foundry (for Solidity compilation)

```bash
curl -L https://foundry.paradigm.xyz | bash
foundryup
```

### Step 3: Install Ganache (Local Ethereum Blockchain)

```bash
npm install -g ganache
```

Or download from: https://trufflesuite.com/ganache/

### Step 4: Install IPFS

```bash
# macOS
brew install ipfs

# Or download from: https://ipfs.tech/#install
```

## 🏃 Usage

### 1. Start Ganache (Local Blockchain)

```bash
ganache --port 7545
```

Keep this running in a separate terminal.

### 2. Start IPFS Daemon

```bash
ipfs daemon
```

Keep this running in a separate terminal.

### 3. Run Experiments

```bash
python src/main_experiment.py
```

This will run all 4 experiments:
- **Centralized Static**: Traditional FL with fixed DP
- **Centralized Adaptive**: Traditional FL with adaptive DP
- **Decentralized Static**: Blockchain FL with fixed DP
- **Decentralized Adaptive**: Blockchain FL with adaptive DP

### 4. View Results

The script generates:
- `federated_learning_comparison.png`: Comparison plots (Accuracy, Epsilon, Latency)
- Console output with detailed metrics and cost analysis

## 📊 Experiments

### Experiment A: Utility Comparison
Compares test accuracy across all four scenarios to evaluate the impact of:
- Centralized vs. Decentralized architecture
- Static vs. Adaptive privacy mechanisms

### Experiment B: Privacy Efficiency
Plots "Accuracy vs. Privacy Budget Consumed" to show how adaptive DP optimizes the privacy-utility trade-off.

### Experiment C: Cost Analysis
Analyzes blockchain transaction costs:
- Gas usage per round
- Mainnet vs. Layer 2 (Optimism/Arbitrum) cost comparison
- Economic feasibility assessment

## 📁 Project Structure

```
Blockchain-Adaptive-DP-FL/
├── contracts/
│   └── FLRegistry.sol          # Smart contract for FL coordination
├── src/
│   ├── main_experiment.py      # Main experiment orchestration
│   ├── models.py               # CNN model and CIFAR-10 data loading
│   ├── adaptive_privacy.py     # Adaptive DP controller and RDP accountant
│   ├── blockchain_ganache.py  # Ganache/Web3 integration
│   └── ipfs_handler.py         # IPFS upload/download operations
├── test/
│   └── FLRegistry.t.sol        # Foundry tests for smart contract
├── test_*.py                    # Python unit tests
├── requirements.txt            # Python dependencies
├── foundry.toml                # Foundry configuration
└── README.md                   # This file
```

## 🔧 Configuration

Edit `src/main_experiment.py` to adjust experiment parameters:

```python
NUM_CLIENTS = 3              # Number of federated learning clients
ROUNDS = 5                   # Number of training rounds
LOCAL_EPOCHS = 1             # Local training epochs per round
TARGET_DELTA = 1e-5          # DP delta parameter
CONVERGENCE_THRESHOLD = 40.0 # Accuracy threshold for convergence
```

## 🧪 Testing

### Smart Contract Tests

```bash
forge test
```

### Python Component Tests

```bash
# Test adaptive privacy mechanism
python test_adaptive_privacy.py

# Test blockchain integration
python test_blockchain_ganache.py

# Test IPFS handler
python test_ipfs_handler.py

# Test RDP accountant
python test_rdp_accountant.py
```

See `TESTING.md` for detailed testing instructions.

## 📈 Key Results

The system demonstrates:

1. **Privacy-Utility Trade-off**: Adaptive DP achieves better accuracy for similar privacy budgets
2. **Decentralization Cost**: LDP (decentralized) has lower utility than CDP (centralized) but eliminates trust requirements
3. **Economic Feasibility**: Mainnet costs are prohibitive (~$17/round), but L2 solutions reduce costs by 95%
4. **System Overhead**: Blockchain/IPFS operations add measurable latency but are acceptable for FL use cases

## 🔬 Technical Details

### Model Architecture

- **Dataset**: CIFAR-10 (32×32 RGB images, 10 classes)
- **Architecture**: Lightweight CNN
  - 2 convolutional layers (3→16→32 channels)
  - 2 fully connected layers (2048→64→10)
  - GroupNorm (not BatchNorm) for DP compliance

### Privacy Accounting

- **RDP Accountant**: Custom Rényi Differential Privacy accountant
- **Composition**: Tracks privacy budget across multiple rounds with varying noise levels
- **Conversion**: RDP → (ε, δ)-DP using optimal alpha selection

### Adaptive Mechanism

- **Monitor**: Tracks validation loss after each epoch
- **Decide**: Adjusts noise multiplier based on loss trend
  - Loss improving → Reduce noise (spend budget)
  - Loss stagnating → Increase noise (conserve budget)
- **Act**: Updates PrivacyEngine parameters for next round

## 🛠️ Development

### Adding New Experiments

Modify `src/main_experiment.py` to add new experiment configurations:

```python
experiments = [
    ("Your_Experiment", "static", True),  # (name, strategy, decentralized)
    # ...
]
```

### Extending the Smart Contract

1. Edit `contracts/FLRegistry.sol`
2. Compile: `forge build`
3. Test: `forge test`
4. Update Python integration in `src/blockchain_ganache.py`

## 📚 References

- **Federated Learning**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- **Differential Privacy**: Dwork & Roth, "The Algorithmic Foundations of Differential Privacy"
- **RDP Accounting**: Mironov, "Rényi Differential Privacy"
- **Opacus**: Facebook Research, https://opacus.ai/

## 📝 License

This project is for academic research purposes (EECE 571B).

## 👥 Authors

Developed for EECE 571B: Blockchain Foundations course project.

---

**Note**: This is a research prototype. For production use, additional security audits, performance optimizations, and scalability improvements would be required.
