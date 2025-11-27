import time
import torch
import os
from blockchain_utils import w3, deploy_contract, IPFSSimulator

# --- SETUP ACCOUNTS ---
AGGREGATOR_PK = "0x2dbd47064581325f7d13e063dd4380b4f9220845fff2d6d72ed0fc9b36eda044"
CLIENT_1_PK   = "0xd586349d2f99dd51755d814fc37331d98f3690485e685b34f15dd36cd026ab2c"
CLIENT_2_PK   = "0xe909edfb3310839b3ac6d25c9bf5c51268ffadf24ba908939ba3826382d2055e"

aggregator = w3.eth.account.from_key(AGGREGATOR_PK)
client1 = w3.eth.account.from_key(CLIENT_1_PK)
client2 = w3.eth.account.from_key(CLIENT_2_PK)

ipfs = IPFSSimulator()

def main():
    print("--- 🚀 Phase 2: Blockchain Simulation Started ---")
    
    # 1. Deploy Contract
    contract = deploy_contract(aggregator.address, AGGREGATOR_PK)
    
    # 2. Clients Register
    print("\n--- Registering Clients ---")
    for client, pk in [(client1, CLIENT_1_PK), (client2, CLIENT_2_PK)]:
        tx = contract.functions.registerClient().build_transaction({
            'from': client.address,
            'nonce': w3.eth.get_transaction_count(client.address),
            'gas': 500000,
            'gasPrice': w3.eth.gas_price
        })
        signed = w3.eth.account.sign_transaction(tx, pk)
        
        # --- FIX 2: .raw_transaction ---
        w3.eth.send_raw_transaction(signed.raw_transaction)
        print(f"Client {client.address[:6]}... registered.")

    # 3. Aggregator Starts Round 1
    print("\n--- Starting Round 1 ---")
    dummy_model_path = "global_model_r0.pt"
    torch.save({"state": "dummy"}, dummy_model_path) 
    global_hash = ipfs.add(dummy_model_path)
    
    tx = contract.functions.startRound(global_hash).build_transaction({
        'from': aggregator.address,
        'nonce': w3.eth.get_transaction_count(aggregator.address),
        'gas': 500000,
        'gasPrice': w3.eth.gas_price
    })
    
    signed_tx = w3.eth.account.sign_transaction(tx, AGGREGATOR_PK)
    w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    print(f"Aggregator posted Global Model: {global_hash}")

    # 4. Clients Poll and React
    print("\n--- Clients Polling ---")
    round_num = contract.functions.currentRound().call()
    
    if round_num == 1:
        print("🔔 Clients detected Round 1!")
        
        ipfs.get(global_hash, "client1_received_model.pt")
        
        print("Client 1 Training...")
        time.sleep(1)
        torch.save({"state": "trained_c1"}, "client1_update.pt")
        
        update_hash = ipfs.add("client1_update.pt")
        
        tx = contract.functions.submitHash(update_hash).build_transaction({
            'from': client1.address,
            'nonce': w3.eth.get_transaction_count(client1.address),
            'gas': 500000,
            'gasPrice': w3.eth.gas_price
        })
        
        # --- FIX 4: .raw_transaction ---
        signed_tx = w3.eth.account.sign_transaction(tx, CLIENT_1_PK)
        w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        print(f"Client 1 submitted update: {update_hash}")

    updates = contract.functions.getRoundUpdates(1).call()
    print(f"\n--- Contract State ---")
    print(f"Updates received on-chain: {len(updates)}")
    if len(updates) > 0:
        print(f"Latest update from: {updates[0][0]}")

    if os.path.exists(dummy_model_path): os.remove(dummy_model_path)

if __name__ == "__main__":
    main()