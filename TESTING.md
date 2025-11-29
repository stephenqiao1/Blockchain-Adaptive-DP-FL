# Testing Guide

## Testing FLRegistry.sol Contract

### Using Foundry (Recommended)

1. **Install Foundry** (if not already installed):
   ```bash
   curl -L https://foundry.paradigm.xyz | bash
   foundryup
   ```

2. **Install dependencies**:
   ```bash
   forge install foundry-rs/forge-std
   ```

3. **Run tests**:
   ```bash
   forge test
   ```

4. **Run with verbose output**:
   ```bash
   forge test -vvv
   ```

## Testing blockchain_ganache.py

### Prerequisites

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Ganache**:
   - Option 1: Ganache CLI
     ```bash
     ganache-cli --port 7545
     ```
   - Option 2: Ganache GUI
     - Open Ganache GUI
     - Create a new workspace or use default
     - Make sure it's running on port 7545

### Running the Tests

1. **Make sure Ganache is running** on `http://127.0.0.1:7545`

2. **Run the test script**:
   ```bash
   python test_blockchain_ganache.py
   ```

### What the Tests Cover

The test script (`test_blockchain_ganache.py`) tests:

1. ✅ **Initialization**: Contract deployment and connection
2. ✅ **Submit Update**: Client submitting model updates
3. ✅ **Multiple Submissions**: Multiple clients submitting updates
4. ✅ **End Round**: Aggregator ending a round
5. ✅ **Multiple Rounds**: Full round progression
6. ✅ **Cost Analysis**: Gas usage and cost calculations
7. ✅ **Access Control**: Only owner can end rounds

### Expected Output

You should see output like:
```
============================================================
BLOCKCHAIN GANACHE HANDLER TESTS
============================================================

============================================================
Test 1: Initialization
============================================================
Connected to Ganache. Block Number: 0
🔨 Compiling Solidity Contract...
Deploying Contract to Ganache...
Contract Deployed at: 0x...
✓ Initialization test passed
  - Contract address: 0x...
  - Number of accounts: 10
  - Initial gas used: 123456

...

============================================================
ALL TESTS PASSED! ✓
============================================================
```

### Troubleshooting

**Error: "Failed to connect to Ganache"**
- Make sure Ganache is running
- Check that it's running on port 7545 (default)
- If using a different port, modify the `GanacheHandler` initialization in the test script

**Error: "Contract file not found"**
- Make sure you're running the test from the project root directory
- The script should find `contracts/FLRegistry.sol` automatically

**Import errors**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Verify web3 and solcx are installed: `pip list | grep -E "web3|solcx"`

## Testing ipfs_handler.py

### Prerequisites

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install and start IPFS**:
   - Install IPFS: https://docs.ipfs.io/install/
   - Initialize IPFS (first time only):
     ```bash
     ipfs init
     ```
   - Start IPFS daemon:
     ```bash
     ipfs daemon
     ```
   - The daemon should be running on `http://127.0.0.1:5001/api/v0` (default)

### Running the Tests

1. **Make sure IPFS daemon is running**:
   ```bash
   # In a separate terminal, start IPFS
   ipfs daemon
   ```

2. **Run the test script**:
   ```bash
   python test_ipfs_handler.py
   ```

### What the Tests Cover

The test script (`test_ipfs_handler.py`) tests:

1. ✅ **Initialization**: Handler setup and temp directory creation
2. ✅ **Upload Model**: Uploading PyTorch model state_dicts to IPFS
3. ✅ **Download Model**: Downloading models from IPFS by CID
4. ✅ **Round-Trip**: Upload then download verification
5. ✅ **Multiple Uploads**: Uploading multiple models
6. ✅ **Error Handling**: Behavior when IPFS is unavailable
7. ✅ **Cleanup**: Clearing temporary files

### Expected Output

You should see output like:
```
============================================================
IPFS HANDLER TESTS
============================================================

Checking IPFS connection...
✓ IPFS is running and accessible

============================================================
Test 1: Initialization
============================================================
✓ Initialization test passed
  - API URL: http://127.0.0.1:5001/api/v0
  - Temp directory: ./temp_models

============================================================
Test 2: Upload Model
============================================================
✓ Upload model test passed
  - CID: Qm...
  - Local file: ./temp_models/test_model.pth

...

============================================================
ALL TESTS PASSED! ✓
============================================================
```

### Troubleshooting

**Error: "IPFS does not appear to be running"**
- Make sure IPFS daemon is running: `ipfs daemon`
- Check that it's running on port 5001 (default)
- Verify with: `curl http://127.0.0.1:5001/api/v0/version`

**Error: "IPFS Upload Failed"**
- Check IPFS daemon logs for errors
- Verify IPFS is initialized: `ipfs config show`
- Make sure you have write permissions

**Error: "CID should start with Qm"**
- This is normal - newer IPFS versions may use CIDv1 format (starts with "baf")
- The test accepts both formats

**Import errors**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Verify torch and requests are installed: `pip list | grep -E "torch|requests"`

## Testing adaptive_privacy.py

### Prerequisites

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **No external services required** - This module can be tested standalone

### Running the Tests

Simply run the test script:
```bash
python test_adaptive_privacy.py
```

### What the Tests Cover

The test script (`test_adaptive_privacy.py`) tests:

1. ✅ **Initialization**: Handler setup with static and adaptive strategies
2. ✅ **Static Strategy**: Verifies parameters don't change in static mode
3. ✅ **Adaptive - Decreasing Loss**: Noise reduction when loss decreases
4. ✅ **Adaptive - Increasing Loss**: Noise increase when loss increases
5. ✅ **Adaptive - Loss Spike**: Clipping adjustment on large loss increases
6. ✅ **Boundary Conditions**: Parameters stay within valid ranges
7. ✅ **Patience Counter**: Counter increments/resets correctly
8. ✅ **Privacy Spent**: Epsilon calculation
9. ✅ **Realistic Training Sequence**: Mixed loss patterns simulation

### Expected Output

You should see output like:
```
============================================================
ADAPTIVE PRIVACY CONTROLLER TESTS
============================================================

============================================================
Test 1: Initialization
============================================================
✓ Initialization test passed
  - Static strategy: static
  - Adaptive strategy: adaptive
  - Initial noise_multiplier: 1.0
  - Initial max_grad_norm: 1.0

============================================================
Test 3: Adaptive Strategy - Decreasing Loss
============================================================
  - Loss: 1.00 -> Noise: 0.9500, Clip: 1.0000
  - Loss: 0.90 -> Noise: 0.9025, Clip: 1.0000
  ...

============================================================
ALL TESTS PASSED! ✓
============================================================
```

### Test Details

**Static Strategy Test:**
- Verifies that in static mode, `noise_multiplier` and `max_grad_norm` remain constant regardless of loss values

**Adaptive Strategy Tests:**
- **Decreasing Loss**: When loss decreases, noise_multiplier should decrease by 5% (minimum 0.5)
- **Increasing Loss**: When loss increases, noise_multiplier should increase by 5% (maximum 2.0)
- **Loss Spike**: When loss increases by >10%, max_grad_norm should decrease by 10% (minimum 0.5)

**Boundary Conditions:**
- Noise multiplier: [0.5, 2.0]
- Max grad norm: >= 0.5
- Parameters should never exceed these bounds

**Patience Counter:**
- Increments when loss stagnates/increases
- Resets to 0 when loss improves

### Troubleshooting

**Import errors**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Verify torch and opacus are installed: `pip list | grep -E "torch|opacus"`

**Test failures**
- Check that opacus is properly installed: `pip install opacus`
- Verify PyTorch version compatibility with opacus

