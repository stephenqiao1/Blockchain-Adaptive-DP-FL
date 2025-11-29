import json
import os
from web3 import Web3
from solcx import compile_source, install_solc

class GanacheHandler:
    def __init__(self, ganache_url="http://127.0.0.1:7545"):
        # 1. Connect to Ganache
        self.w3 = Web3(Web3.HTTPProvider(ganache_url))
        if not self.w3.is_connected():
            raise Exception(f"Failed to connect to Ganache at {ganache_url}. Is it running?")
        
        print(f"Connected to Ganache. Block Number: {self.w3.eth.block_number}")
        
        # Setup Accounts
        self.accounts = self.w3.eth.accounts
        self.aggregator = self.accounts[0] # The server
        self.w3.eth.default_account = self.aggregator
        
        # Metrics
        self.total_gas_used = 0
        self.round_gas_used = []  # Track gas per round
        self.current_round_gas = 0  # Gas used in current round
        
        # 2. Compile & Deploy Contract
        self.contract = self._deploy_contract()

    def _deploy_contract(self):
        print("🔨 Compiling Solidity Contract...")
        # Ensure solc is installed
        install_solc("0.8.0")
        
        # Find contract file relative to project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        contract_path = os.path.join(project_root, "contracts", "FLRegistry.sol")
        
        with open(contract_path, "r") as f:
            source_code = f.read()

        compiled_sol = compile_source(
            source_code,
            output_values=["abi", "bin"],
            solc_version="0.8.0"
        )
        contract_id, contract_interface = next(iter(compiled_sol.items()))  # Get first contract

        # Deploy
        print("Deploying Contract to Ganache...")
        Contract = self.w3.eth.contract(
            abi=contract_interface["abi"], 
            bytecode=contract_interface["bin"]
        )
        tx_hash = Contract.constructor().transact({'from': self.aggregator})
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        self.total_gas_used += tx_receipt.gasUsed
        print(f"Contract Deployed at: {tx_receipt.contractAddress}")
        
        return self.w3.eth.contract(
            address=tx_receipt.contractAddress, 
            abi=contract_interface["abi"]
        )

    def submit_update(self, client_index, ipfs_hash_sim):
        """
        Client submits a hash. We use different accounts for different clients
        to simulate a real network.
        """
        # Map client index to a Ganache account (wrapping around if needed)
        client_account = self.accounts[(client_index + 1) % len(self.accounts)]
        
        tx_hash = self.contract.functions.submitUpdate(ipfs_hash_sim).transact({
            'from': client_account
        })
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        self.total_gas_used += receipt.gasUsed
        self.current_round_gas += receipt.gasUsed
        return receipt.gasUsed

    def end_round(self, global_hash_sim):
        """Aggregator publishes new global model hash"""
        tx_hash = self.contract.functions.endRound(global_hash_sim).transact({
            'from': self.aggregator
        })
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        self.total_gas_used += receipt.gasUsed
        self.current_round_gas += receipt.gasUsed
        
        # Store gas for this round and reset
        self.round_gas_used.append(self.current_round_gas)
        round_gas = self.current_round_gas
        self.current_round_gas = 0
        
        return round_gas

    def get_cost_analysis(self):
        # Ganache usually defaults to 20 Gwei, but we can query it
        gas_price_wei = self.w3.eth.gas_price
        
        # Real Math
        eth_cost = self.total_gas_used * gas_price_wei * 1e-18 # Convert Wei to ETH
        usd_price = 3200.00 # Assume $3200/ETH
        
        cost_mainnet_usd = eth_cost * usd_price
        cost_l2_usd = cost_mainnet_usd * 0.05 # L2 approx 5% of Mainnet
        
        # Calculate per-round costs
        per_round_costs = []
        if self.round_gas_used:
            for round_gas in self.round_gas_used:
                round_eth = round_gas * gas_price_wei * 1e-18
                round_mainnet = round_eth * usd_price
                round_l2 = round_mainnet * 0.05
                per_round_costs.append({
                    'gas': round_gas,
                    'usd_mainnet': round_mainnet,
                    'usd_l2': round_l2
                })

        return {
            'gas_used': self.total_gas_used,
            'eth_cost': eth_cost,
            'usd_mainnet': cost_mainnet_usd,
            'usd_l2': cost_l2_usd,
            'per_round': per_round_costs  # New: per-round breakdown
        }