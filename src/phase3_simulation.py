import time
import numpy as np
import matplotlib.pyplot as plt
from web3 import Web3
from blockchain_utils import w3, deploy_contract
from adaptive_logic import AdaptivePrivacyScaler

## --- SETUP ---
AGGREGATOR_PK = "0x2dbd47064581325f7d13e063dd4380b4f9220845fff2d6d72ed0fc9b36eda044"
CLIENT_1_PK   = "0xd586349d2f99dd51755d814fc37331d98f3690485e685b34f15dd36cd026ab2c"
CLIENT_2_PK   = "0xe909edfb3310839b3ac6d25c9bf5c51268ffadf24ba908939ba3826382d2055e"

aggregator = w3.eth.account.from_key(AGGREGATOR_PK)
client1 = w3.eth.account.from_key(CLIENT_1_PK)
client2 = w3.eth.account.from_key(CLIENT_2_PK)

def main():
    print("--- ⚖️ Phase 3: Adaptive Privacy & Budgeting ---")

    # 1. Deploy UPDATED Contract
    print("Deploying Smart Contract...")
    contract = deploy_contract(aggregator.address, AGGREGATOR_PK)

    # 2. Register Clients (WE NEED BOTH CLIENTS NOW)
    # --- Register Client 1 ---
    print("Registering Client 1...")
    tx = contract.functions.registerClient().build_transaction({
        'from': client1.address,
        'nonce': w3.eth.get_transaction_count(client1.address),
        'gas': 500000,
        'gasPrice': w3.eth.gas_price
    })
    w3.eth.send_raw_transaction(w3.eth.account.sign_transaction(tx, CLIENT_1_PK).raw_transaction)

    # --- Register Client 2 (THIS WAS MISSING) ---
    print("Registering Client 2...")
    tx = contract.functions.registerClient().build_transaction({
        'from': client2.address,
        'nonce': w3.eth.get_transaction_count(client2.address),
        'gas': 500000,
        'gasPrice': w3.eth.gas_price
    })
    w3.eth.send_raw_transaction(w3.eth.account.sign_transaction(tx, CLIENT_2_PK).raw_transaction)


    # 3. Simulation Loop
    simulated_losses = [2.0, 1.8, 1.6, 1.4, 1.0, 0.8, 0.5, 0.4, 0.3, 0.2]
    
    scaler = AdaptivePrivacyScaler(base_sigma=1.0)
    total_budget_spent = 0
    budget_history = []
    sigma_history = []

    print("\n--- Starting Training Rounds ---")
    
    for round_num, loss in enumerate(simulated_losses, 1):
        print(f"\n[Round {round_num}] Current Val Loss: {loss}")
        
        # A. Start Round (Aggregator)
        # Note: We rely on the contract allowing the Aggregator to force-start new rounds
        try:
            tx = contract.functions.startRound("hash_placeholder").build_transaction({
                'from': aggregator.address,
                'nonce': w3.eth.get_transaction_count(aggregator.address),
                'gas': 500000,
                'gasPrice': w3.eth.gas_price
            })
            w3.eth.send_raw_transaction(w3.eth.account.sign_transaction(tx, AGGREGATOR_PK).raw_transaction)
        except Exception as e:
            # If round is already active or other issues, print but don't crash
            print(f"   ⚠️ Blockchain Warning: {e}")

        # B. Adaptive Logic (Calculate Sigma & Cost)
        current_sigma = scaler.adjust_sigma(loss, 0) 
        round_cost_float = scaler.calculate_epsilon_cost(current_sigma)
        
        # Convert to Integer for Blockchain (Scale x100)
        round_cost_int = int(round_cost_float * 100)
        
        print(f"   -> Adaptive Sigma: {current_sigma:.3f}")
        print(f"   -> Budget Cost: {round_cost_float:.2f} (Int: {round_cost_int})")

        # C. Submit Update to Blockchain
        try:
            tx = contract.functions.submitHash("update_hash", round_cost_int).build_transaction({
                'from': client1.address,
                'nonce': w3.eth.get_transaction_count(client1.address),
                'gas': 500000,
                'gasPrice': w3.eth.gas_price
            })
            w3.eth.send_raw_transaction(w3.eth.account.sign_transaction(tx, CLIENT_1_PK).raw_transaction)
            
            total_budget_spent += round_cost_float
            print(f"   ✅ Update Accepted. Total Budget Used: {total_budget_spent:.2f}")
            
        except Exception as e:
            # We catch the specific "revert" message here
            if "Privacy Budget Exceeded" in str(e) or "revert" in str(e):
                print(f"   ❌ BLOCKED BY CONTRACT: Budget Exceeded!")
                print("   (Client has been locked out to preserve privacy)")
                break
            else:
                print(f"   ❌ Transaction Error: {e}")

        budget_history.append(total_budget_spent)
        sigma_history.append(current_sigma)
        time.sleep(1) 

    # 4. Generate the Phase 3 Deliverable Plot
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(budget_history)+1), budget_history, marker='o', color='red')
    plt.title('Privacy Budget Consumption')
    plt.xlabel('Round')
    plt.ylabel('Cumulative Epsilon')
    plt.axhline(y=20.0, color='black', linestyle='--', label='Max Budget')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(sigma_history)+1), sigma_history, marker='o', color='blue')
    plt.title('Adaptive Noise (Sigma)')
    plt.xlabel('Round')
    plt.ylabel('Sigma (Noise Level)')
    
    plt.tight_layout()
    plt.savefig('phase3_adaptive_results.png')
    print("\nGraph saved as phase3_adaptive_results.png")
    
if __name__ == "__main__":
    main()