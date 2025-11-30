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
        
        # 3. Register aggregator as owner (already registered by default)
        # Register all client accounts
        self._register_clients()

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
    
    def _register_clients(self):
        """Register all client accounts with the contract"""
        print("Registering clients...")
        registered_count = 0
        for i in range(1, min(len(self.accounts), 10)):  # Register up to 9 clients
            try:
                client_account = self.accounts[i]
                # Check if already registered
                is_registered = self.contract.functions.clients(client_account).call()[0]
                if not is_registered:
                    # Use new registerClient() function
                    try:
                        tx_hash = self.contract.functions.registerClient().transact({
                            'from': client_account
                        })
                    except:
                        # Fallback to legacy register()
                        tx_hash = self.contract.functions.register().transact({
                            'from': client_account
                        })
                    self.w3.eth.wait_for_transaction_receipt(tx_hash)
                    registered_count += 1
            except Exception as e:
                # If registration fails, continue (might already be registered)
                pass
        if registered_count > 0:
            print(f"   Registered {registered_count} clients")

    def submit_update(self, client_index, ipfs_hash_sim, epsilon_cost=None):
        """
        Client submits a hash with epsilon cost tracking.
        We use different accounts for different clients to simulate a real network.
        
        Args:
            client_index: Index of the client
            ipfs_hash_sim: IPFS hash (CID) of the model update
            epsilon_cost: Privacy budget consumed (scaled by 1e18). If None, uses default 0.1
        """
        # Map client index to a Ganache account (wrapping around if needed)
        client_account = self.accounts[(client_index + 1) % len(self.accounts)]
        
        # Convert string hash to bytes32 if needed
        # For IPFS hashes (Qm...), we'll use keccak256 hash of the string
        # In production, you'd want to properly decode the base58 IPFS hash
        if isinstance(ipfs_hash_sim, str):
            # Use web3 to hash the string to bytes32
            from web3 import Web3
            ipfs_hash_bytes32 = Web3.keccak(text=ipfs_hash_sim)
        else:
            ipfs_hash_bytes32 = ipfs_hash_sim
        
        # Default epsilon cost: 0.1 (scaled by 1e18)
        if epsilon_cost is None:
            epsilon_cost = int(0.1 * 1e18)  # 0.1 epsilon
        else:
            # Scale epsilon_cost to 1e18 if it's a float
            if isinstance(epsilon_cost, float):
                epsilon_cost = int(epsilon_cost * 1e18)
        
        # Use new submitHash function (matches specification)
        try:
            # Convert bytes32 back to string for submitHash (it accepts string)
            # Or use registerUpdate with bytes32
            # Try submitHash first (matches spec)
            tx_hash = self.contract.functions.submitHash(
                ipfs_hash_sim,  # String format
                epsilon_cost
            ).transact({
                'from': client_account
            })
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            self.total_gas_used += receipt.gasUsed
            self.current_round_gas += receipt.gasUsed
            return receipt.gasUsed
        except Exception as e1:
            # Fallback to registerUpdate (bytes32 version)
            try:
                tx_hash = self.contract.functions.registerUpdate(
                    ipfs_hash_bytes32,
                    epsilon_cost
                ).transact({
                    'from': client_account
                })
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                self.total_gas_used += receipt.gasUsed
                self.current_round_gas += receipt.gasUsed
                return receipt.gasUsed
            except Exception as e2:
                # Fallback to legacy function if new ones fail
                print(f"   Warning: Using legacy submitUpdate")
                tx_hash = self.contract.functions.submitUpdate(ipfs_hash_sim).transact({
                    'from': client_account
                })
                receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                self.total_gas_used += receipt.gasUsed
                self.current_round_gas += receipt.gasUsed
                return receipt.gasUsed

    def start_round(self, global_hash_sim):
        """
        Start a new round and emit event signaling availability of new global model CID.
        This is called by the aggregator to signal clients that a new global model is available.
        
        Args:
            global_hash_sim: IPFS hash (CID) of the new global model (string)
        """
        if isinstance(global_hash_sim, bytes):
            # Convert bytes to string if needed
            global_hash_sim = global_hash_sim.decode('utf-8') if isinstance(global_hash_sim, bytes) else str(global_hash_sim)
        
        try:
            tx_hash = self.contract.functions.startRound(global_hash_sim).transact({
                'from': self.aggregator
            })
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            self.total_gas_used += receipt.gasUsed
            self.current_round_gas += receipt.gasUsed
            return receipt.gasUsed
        except Exception as e:
            print(f"   Warning: startRound failed, using verifyAndAggregate: {str(e)[:50]}")
            return self.end_round(global_hash_sim)
    
    def end_round(self, global_hash_sim):
        """
        Aggregator verifies and publishes new global model hash.
        Uses verifyAndAggregate() for new contract, falls back to endRound() for legacy.
        """
        # Convert string hash to bytes32 if needed
        if isinstance(global_hash_sim, str):
            from web3 import Web3
            hash_bytes32 = Web3.keccak(text=global_hash_sim)
        else:
            hash_bytes32 = global_hash_sim
        
        # Use new verifyAndAggregate function
        try:
            tx_hash = self.contract.functions.verifyAndAggregate(hash_bytes32).transact({
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
        except Exception as e:
            # Fallback to legacy function
            print(f"   Warning: Using legacy endRound")
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
    
    def get_client_status(self, client_index):
        """
        Get client status including reputation and consumed epsilon.
        
        Args:
            client_index: Index of the client
            
        Returns:
            dict with registered, reputation, consumedEpsilon, lastRoundParticipated
        """
        client_account = self.accounts[(client_index + 1) % len(self.accounts)]
        try:
            status = self.contract.functions.getClientStatus(client_account).call()
            return {
                'registered': status[0],
                'reputation': status[1],
                'consumedEpsilon': status[2] / 1e18,  # Convert back from scaled
                'lastRoundParticipated': status[3]
            }
        except Exception as e:
            return None
    
    def get_global_model_hash(self):
        """Get the current global model hash from the contract"""
        try:
            return self.contract.functions.globalModelHash().call()
        except Exception as e:
            return None

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