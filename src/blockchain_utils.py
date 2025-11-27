import json
import os
import time
from web3 import Web3
from solcx import compile_source, install_solc

# Force install exactly 0.8.0
install_solc('0.8.0')

RPC_URL = "http://127.0.0.1:7545"
w3 = Web3(Web3.HTTPProvider(RPC_URL))

if not w3.is_connected():
    raise Exception("❌ Failed to connect to Blockchain. Is Ganache running?")

class IPFSSimulator:
    def __init__(self, storage_folder="./ipfs_sim"):
        self.folder = storage_folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def add(self, file_path):
        filename = os.path.basename(file_path)
        dest = os.path.join(self.folder, filename)
        with open(file_path, 'rb') as f_src:
            with open(dest, 'wb') as f_dst:
                f_dst.write(f_src.read())
        return filename 

    def get(self, ipfs_hash, save_path):
        src = os.path.join(self.folder, ipfs_hash)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Hash {ipfs_hash} not found in IPFS Sim.")
        
        with open(src, 'rb') as f_src:
            with open(save_path, 'wb') as f_dst:
                f_dst.write(f_src.read())
        print(f"⬇️  Downloaded model {ipfs_hash} from IPFS.")

def deploy_contract(account_addr, private_key):
    contract_path = './contracts/FLRegistry.sol'
    if not os.path.exists(contract_path):
        raise FileNotFoundError(f"Cannot find contract at {contract_path}")

    with open(contract_path, 'r') as f:
        contract_source = f.read()

    # Compile with explicit version
    compiled_sol = compile_source(
        contract_source,
        output_values=['abi', 'bin'],
        solc_version='0.8.0'
    )
    
    contract_id, contract_interface = compiled_sol.popitem()
    bytecode = contract_interface['bin']
    abi = contract_interface['abi']

    FLRegistry = w3.eth.contract(abi=abi, bytecode=bytecode)
    
    print(f"Deploying from: {account_addr}...")
    nonce = w3.eth.get_transaction_count(account_addr)
    
    tx = FLRegistry.constructor(2).build_transaction({
        'chainId': w3.eth.chain_id,
        'gas': 2000000,
        'from': account_addr,
        'nonce': nonce,
        'gasPrice': w3.eth.gas_price
    })
    
    signed_tx = w3.eth.account.sign_transaction(tx, private_key)
    
    # --- FIX 1: Change .rawTransaction to .raw_transaction ---
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    
    print("Waiting for transaction receipt...")
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    print(f"✅ Contract Deployed at: {tx_receipt.contractAddress}")
    return w3.eth.contract(address=tx_receipt.contractAddress, abi=abi)