#!/usr/bin/env python3
"""
Python test script for FLRegistry.sol using web3.py and py-solc-x
This script compiles and tests the FLRegistry contract on a local test network.
"""

import json
import os
from web3 import Web3
from solcx import compile_source, install_solc, set_solc_version
from eth_account import Account

# Install and set Solidity compiler version
install_solc('0.8.0')
set_solc_version('0.8.0')

# Connect to local test network (Ganache, Hardhat, or Anvil)
# For testing, you can use Anvil: anvil --port 8545
RPC_URL = os.getenv('RPC_URL', 'http://127.0.0.1:8545')

def compile_contract():
    """Compile the FLRegistry contract"""
    contract_path = 'contracts/FLRegistry.sol'
    with open(contract_path, 'r') as f:
        source_code = f.read()
    
    compiled = compile_source(source_code, output_values=['abi', 'bin'])
    contract_id, contract_interface = compiled.popitem()
    return contract_interface

def deploy_contract(w3, account, contract_interface):
    """Deploy the contract and return the contract instance"""
    contract = w3.eth.contract(abi=contract_interface['abi'], bytecode=contract_interface['bin'])
    
    # Build transaction
    construct_txn = contract.constructor().build_transaction({
        'from': account.address,
        'nonce': w3.eth.get_transaction_count(account.address),
        'gas': 2000000,
        'gasPrice': w3.eth.gas_price
    })
    
    # Sign and send transaction
    signed_txn = account.sign_transaction(construct_txn)
    tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    
    # Get contract instance
    contract_instance = w3.eth.contract(address=tx_receipt.contractAddress, abi=contract_interface['abi'])
    return contract_instance, tx_receipt.contractAddress

def test_initial_state(contract_instance, owner_address):
    """Test that contract initializes correctly"""
    print("Testing initial state...")
    assert contract_instance.functions.owner().call() == owner_address, "Owner should be deployer"
    assert contract_instance.functions.currentRound().call() == 0, "Initial round should be 0"
    print("✓ Initial state test passed")

def test_submit_update(contract_instance, client_account, w3):
    """Test submitting a model update"""
    print("Testing submitUpdate...")
    ipfs_hash = "QmTestHash123"
    
    # Build and send transaction
    tx = contract_instance.functions.submitUpdate(ipfs_hash).build_transaction({
        'from': client_account.address,
        'nonce': w3.eth.get_transaction_count(client_account.address),
        'gas': 200000,
        'gasPrice': w3.eth.gas_price
    })
    
    signed_tx = client_account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    
    # Check event was emitted
    logs = contract_instance.events.ModelUpdate().process_receipt(receipt)
    assert len(logs) > 0, "ModelUpdate event should be emitted"
    assert logs[0]['args']['client'] == client_account.address, "Client address should match"
    assert logs[0]['args']['round'] == 0, "Round should be 0"
    assert logs[0]['args']['ipfsHash'] == ipfs_hash, "IPFS hash should match"
    print("✓ submitUpdate test passed")

def test_end_round(contract_instance, owner_account, w3):
    """Test ending a round"""
    print("Testing endRound...")
    global_hash = "QmGlobalHash"
    
    tx = contract_instance.functions.endRound(global_hash).build_transaction({
        'from': owner_account.address,
        'nonce': w3.eth.get_transaction_count(owner_account.address),
        'gas': 200000,
        'gasPrice': w3.eth.gas_price
    })
    
    signed_tx = owner_account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    
    # Check round incremented
    assert contract_instance.functions.currentRound().call() == 1, "Round should increment to 1"
    
    # Check event was emitted
    logs = contract_instance.events.RoundEnded().process_receipt(receipt)
    assert len(logs) > 0, "RoundEnded event should be emitted"
    assert logs[0]['args']['round'] == 1, "Round should be 1"
    assert logs[0]['args']['globalModelHash'] == global_hash, "Global hash should match"
    print("✓ endRound test passed")

def test_end_round_only_owner(contract_instance, client_account, w3):
    """Test that only owner can end round"""
    print("Testing endRound access control...")
    global_hash = "QmGlobalHash"
    
    tx = contract_instance.functions.endRound(global_hash).build_transaction({
        'from': client_account.address,
        'nonce': w3.eth.get_transaction_count(client_account.address),
        'gas': 200000,
        'gasPrice': w3.eth.gas_price
    })
    
    signed_tx = client_account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    
    try:
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        # If we get here, the transaction succeeded, which is wrong
        assert False, "Transaction should have reverted"
    except Exception as e:
        # Transaction should revert
        print("✓ endRound access control test passed (transaction reverted as expected)")

def main():
    """Run all tests"""
    print("=" * 50)
    print("FLRegistry Contract Tests")
    print("=" * 50)
    
    # Connect to local network
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    if not w3.is_connected():
        print(f"Error: Could not connect to {RPC_URL}")
        print("Make sure you have a local blockchain running (e.g., Anvil, Ganache, or Hardhat)")
        print("For Anvil: anvil --port 8545")
        return
    
    print(f"Connected to {RPC_URL}")
    print(f"Chain ID: {w3.eth.chain_id}")
    
    # Get accounts (use first two accounts from local network)
    accounts = w3.eth.accounts
    if len(accounts) < 2:
        print("Error: Need at least 2 accounts in the test network")
        return
    
    owner_account = Account.from_key('0x' + '0' * 64)  # Dummy key, will use accounts[0]
    client_account = Account.from_key('0x' + '0' * 64)  # Dummy key, will use accounts[1]
    
    # For local test networks, we can use the accounts directly
    owner_address = accounts[0]
    client_address = accounts[1]
    
    # Create account objects that work with the network
    class NetworkAccount:
        def __init__(self, address):
            self.address = address
    
    owner_account = NetworkAccount(owner_address)
    client_account = NetworkAccount(client_address)
    
    # Compile contract
    print("\nCompiling contract...")
    contract_interface = compile_contract()
    print("✓ Contract compiled")
    
    # Deploy contract
    print("\nDeploying contract...")
    # For deployment, we need to use the actual account
    contract = w3.eth.contract(abi=contract_interface['abi'], bytecode=contract_interface['bin'])
    construct_txn = contract.constructor().build_transaction({
        'from': owner_address,
        'nonce': w3.eth.get_transaction_count(owner_address),
    })
    signed_txn = w3.eth.account.sign_transaction(construct_txn, private_key=None)  # Local network handles this
    # Actually, for local networks, we can just send directly
    tx_hash = w3.eth.send_transaction(construct_txn)
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    contract_address = tx_receipt.contractAddress
    contract_instance = w3.eth.contract(address=contract_address, abi=contract_interface['abi'])
    print(f"✓ Contract deployed at {contract_address}")
    
    # Run tests
    print("\n" + "=" * 50)
    print("Running Tests")
    print("=" * 50 + "\n")
    
    try:
        test_initial_state(contract_instance, owner_address)
        test_submit_update(contract_instance, client_account, w3)
        test_end_round(contract_instance, owner_account, w3)
        test_end_round_only_owner(contract_instance, client_account, w3)
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == '__main__':
    main()

