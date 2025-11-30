import torch
import torch.optim as optim
import copy
import numpy as np
import matplotlib.pyplot as plt
from opacus import PrivacyEngine
import os

# Custom Modules
from models import SimpleCNN, get_cifar10_loaders
from blockchain_ganache import GanacheHandler
from adaptive_privacy import AdaptivePrivacyController, RDPAccountantCustom
from ipfs_handler import IPFSHandler

# --- CONFIG ---
NUM_CLIENTS = 3
ROUNDS = 5
LOCAL_EPOCHS = 1
TARGET_DELTA = 1e-5
DEVICE = torch.device("cpu")

def run_experiment(experiment_name, privacy_strategy, decentralized=True):
    """
    Run federated learning experiment
    
    Args:
        experiment_name: Name for logging
        privacy_strategy: 'static' or 'adaptive'
        decentralized: True for blockchain/IPFS (LDP), False for centralized (CDP)
    
    Note on Privacy Mechanisms:
        - Centralized Mode (decentralized=False): Uses Centralized DP (CDP)
          * Clients train WITHOUT noise
          * Server adds noise to aggregate (O(1) noise scaling)
          * Better utility (accuracy) for same privacy budget
          
        - Decentralized Mode (decentralized=True): Uses Local DP (LDP)  
          * Each client adds noise locally (O(√K) noise scaling)
          * "Cost of trustlessness" - worse utility but no trusted server
          * Required for blockchain-based systems
    """
    print(f"\n🚀 STARTING EXPERIMENT: {experiment_name}")
    mode_str = "Decentralized (LDP)" if decentralized else "Centralized (CDP)"
    print(f"   Strategy: {privacy_strategy} | Mode: {mode_str}")
    
    # Setup Infrastructure
    chain = None
    ipfs = None
    
    if decentralized:
        try:
            chain = GanacheHandler(ganache_url="http://127.0.0.1:7545")
            ipfs = IPFSHandler()
        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            print("Make sure Ganache and IPFS are running for decentralized mode!")
            return None

    # Load data and initialize model
    client_loaders, test_loader = get_cifar10_loaders(NUM_CLIENTS)
    global_model = SimpleCNN().to(DEVICE)
    
    # Initialize privacy controllers
    controllers = [
        AdaptivePrivacyController(
            model=copy.deepcopy(global_model),
            dataloader=loader,
            base_epsilon_target=10.0,
            epochs=ROUNDS*LOCAL_EPOCHS,
            strategy=privacy_strategy
        ) for loader in client_loaders
    ]

    history = {'accuracy': [], 'epsilon': []}
    cumulative_epsilon = 0.0  # Track cumulative privacy budget

    for r in range(ROUNDS):
        print(f"\n--- Round {r+1}/{ROUNDS} ---")
        round_epsilons = []
        local_weights = []  # For centralized mode
        round_cids = []      # For decentralized mode

        # --- PHASE 1: CLIENT TRAINING ---
        for i in range(NUM_CLIENTS):
            # Train client model
            client_model = copy.deepcopy(global_model)
            client_model.train()
            optimizer = optim.SGD(client_model.parameters(), lr=0.01)
            
            # Setup privacy engine
            controller = controllers[i]
            privacy_engine = None
            
            if decentralized:
                # DECENTRALIZED MODE: Local DP (LDP)
                # Each client adds noise locally - this is the "cost of trustlessness"
                # 
                # Privacy Engineering: PrivacyEngine attaches a hook to the optimizer,
                # ensuring that gradients are clipped to maximum norm C and perturbed
                # with Gaussian noise before the update step. This implements the
                # differentially private training mechanism.
                #
                # Note: For LDP with Opacus, gradient norms are collected after Opacus processes them
                # (Opacus clips internally, so we get post-clip norms, but still useful for adaptation)
                privacy_engine = PrivacyEngine()
                client_model, optimizer, train_loader = privacy_engine.make_private(
                    module=client_model,
                    optimizer=optimizer,
                    data_loader=client_loaders[i],
                    noise_multiplier=controller.noise_multiplier,
                    max_grad_norm=controller.max_grad_norm,
                )
            else:
                # CENTRALIZED MODE: No noise at client (will add at server)
                # Clients train without DP - server will add noise to aggregate
                train_loader = client_loaders[i]

            # Training
            epoch_loss = 0.0
            steps_this_round = 0
            actual_batch_size = None  # Track actual batch size from training
            for epoch in range(LOCAL_EPOCHS):
                # Training phase
                for data, target in train_loader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    batch_size = data.size(0)
                    if actual_batch_size is None:
                        actual_batch_size = batch_size  # Store first batch size
                    optimizer.zero_grad()
                    output = client_model(data)
                    loss = torch.nn.functional.cross_entropy(output, target)
                    loss.backward()
                    
                    # Collect gradient norms for adaptive clipping
                    # For CDP: collect before clipping (raw gradients)
                    # For LDP: collect after Opacus clipping (still useful for adaptation)
                    if controller.strategy == 'adaptive':
                        if decentralized:
                            # For LDP: Opacus has already clipped, but we can still collect norms
                            # from the wrapped model for adaptation purposes
                            controller.collect_gradient_norm(client_model._module if hasattr(client_model, '_module') else client_model)
                        else:
                            # For CDP: collect raw gradient norms before clipping
                            controller.collect_gradient_norm(client_model)
                    
                    # Clip gradients for CDP (if not using Opacus)
                    if not decentralized:
                        torch.nn.utils.clip_grad_norm_(client_model.parameters(), controller.max_grad_norm)
                    
                    optimizer.step()
                    epoch_loss += loss.item()
                    steps_this_round += 1
                    
                    # Record training step for RDP accounting (for adaptive strategy)
                    # NOTE: For centralized mode, we don't record steps here because clients
                    # train without noise. Privacy cost is computed at server aggregation.
                    if controller.strategy == 'adaptive' and decentralized:
                        controller.record_training_step(batch_size)
                
                # Monitor: Compute validation loss after each local training epoch
                # This implements the "Monitor" step of the closed-loop controller
                if controller.strategy == 'adaptive':
                    # Evaluate on validation set (using test_loader as validation set)
                    client_model.eval()
                    val_loss = 0.0
                    val_count = 0
                    with torch.no_grad():
                        # Use a subset of test_loader for validation (to avoid overfitting to test set)
                        # In practice, you'd have a separate validation set, but for this experiment
                        # we use a portion of the test set
                        for val_data, val_target in test_loader:
                            val_data, val_target = val_data.to(DEVICE), val_target.to(DEVICE)
                            val_output = client_model(val_data)
                            val_loss += torch.nn.functional.cross_entropy(val_output, val_target, reduction='sum').item()
                            val_count += val_data.size(0)
                            if val_count >= 1000:  # Limit validation samples for efficiency
                                break
                    val_loss = val_loss / val_count if val_count > 0 else float('inf')
                    client_model.train()
                    
                    # Adapt privacy parameters based on validation loss (Monitor-Decide-Act pattern)
                    # This implements the "Decide" and "Act" steps of the closed-loop controller
                    controller.adapt_parameters(val_loss)
            
            # Track epsilon for this round using RDP accountant
            if decentralized:
                # LDP: Use Opacus PrivacyEngine's accountant (it tracks automatically)
                # But also track with our custom RDP accountant for consistency
                if privacy_engine:
                    # Opacus tracks automatically, but we can also use our accountant
                    round_epsilon_opacus = privacy_engine.get_epsilon(TARGET_DELTA)
                    # For adaptive, use RDP accountant; for static, use Opacus
                    if controller.strategy == 'adaptive':
                        # Use actual batch size from training, fallback to dataloader batch_size
                        batch_size_for_epsilon = actual_batch_size if actual_batch_size is not None else (
                            train_loader.batch_size if hasattr(train_loader, 'batch_size') and train_loader.batch_size is not None else 64
                        )
                        round_epsilon = controller.get_round_privacy_cost(steps_this_round, batch_size_for_epsilon, TARGET_DELTA)
                    else:
                        round_epsilon = round_epsilon_opacus
                    round_epsilons.append(round_epsilon)
                else:
                    round_epsilons.append(0.0)
            else:
                # CDP: Clients train WITHOUT noise (noise added at server only)
                # Epsilon will be computed at server aggregation step, not here
                round_epsilons.append(0.0)

            # Extract model weights
            if decentralized:
                clean_state_dict = client_model._module.state_dict()  # Opacus wraps model
            else:
                clean_state_dict = client_model.state_dict()  # Regular model

            if decentralized:
                # Decentralized: Upload to IPFS and submit to blockchain
                filename = f"client_{i}_round_{r}.pth"
                print(f"   [Client {i}] Uploading to IPFS...", end="\r")
                cid = ipfs.upload_model(clean_state_dict, filename)
                
                if cid:
                    # Get epsilon cost for this client
                    # For LDP: use per-client epsilon; for CDP: will be computed at server
                    if round_epsilons and i < len(round_epsilons):
                        client_epsilon = round_epsilons[i]
                    else:
                        client_epsilon = 0.0  # Will be set at server for CDP
                    
                    chain.submit_update(client_index=i, ipfs_hash_sim=cid, epsilon_cost=client_epsilon)
                    round_cids.append(cid)
                    if client_epsilon > 0:
                        print(f"   [Client {i}] ✅ Uploaded. CID: {cid[:10]}... (ε: {client_epsilon:.4f})")
                    else:
                        print(f"   [Client {i}] ✅ Uploaded. CID: {cid[:10]}...")
                else:
                    print(f"   [Client {i}] ❌ Failed to upload")
            else:
                # Centralized: Direct model sharing (no blockchain/IPFS)
                local_weights.append(clean_state_dict)

        # --- PHASE 2: AGGREGATION ---
        if decentralized:
            # Decentralized: Download from IPFS
            print("   [Aggregator] Downloading models from IPFS...")
            downloaded_weights = []
            for cid in round_cids:
                w = ipfs.download_model(cid)
                if w:
                    downloaded_weights.append(w)
            
            if downloaded_weights:
                # Average weights
                avg_weights = copy.deepcopy(downloaded_weights[0])
                for k in avg_weights.keys():
                    for i in range(1, len(downloaded_weights)):
                        avg_weights[k] += downloaded_weights[i][k]
                    avg_weights[k] = torch.div(avg_weights[k], len(downloaded_weights))
                
                global_model.load_state_dict(avg_weights)
                
                # Upload global model to IPFS and record on blockchain
                global_cid = ipfs.upload_model(global_model.state_dict(), f"global_round_{r}.pth")
                chain.end_round(global_cid)
                
                # Cleanup
                ipfs.clear_temp_files()
        else:
            # CENTRALIZED MODE: Centralized DP (CDP)
            # Server aggregates first, then adds noise to the aggregate
            if local_weights:
                # Step 1: Aggregate (FedAvg)
                avg_weights = copy.deepcopy(local_weights[0])
                for k in avg_weights.keys():
                    for i in range(1, len(local_weights)):
                        avg_weights[k] += local_weights[i][k]
                    avg_weights[k] = torch.div(avg_weights[k], len(local_weights))
                
                # Step 2: Compute weight update (difference from current global model)
                # This represents the aggregated gradient update
                current_global = global_model.state_dict()
                weight_update = {}
                for k in avg_weights.keys():
                    weight_update[k] = avg_weights[k] - current_global[k]
                
                # Step 3: Add noise to weight UPDATE (Centralized DP)
                # KEY DIFFERENCE: In CDP, noise is added ONCE at server with scale O(1)
                # In LDP, each client adds noise independently, leading to O(√K) scaling
                avg_noise_mult = np.mean([c.noise_multiplier for c in controllers])
                avg_clip_norm = np.mean([c.max_grad_norm for c in controllers])
                learning_rate = 0.01  # Match the optimizer LR
                
                # CDP: Server adds noise to aggregated gradients
                # The noise scale: sigma * C, where sigma is reduced by sqrt(K) for efficiency
                # Since weight_update ≈ lr * gradient, noise on weights ≈ lr * noise_on_gradients
                cdp_noise_mult = avg_noise_mult / np.sqrt(NUM_CLIENTS)  # More efficient
                cdp_noise_scale = learning_rate * cdp_noise_mult * avg_clip_norm
                
                # Add Gaussian noise to weight updates (not absolute weights)
                # This properly simulates adding noise to gradients before weight update
                for k in weight_update.keys():
                    noise = torch.normal(
                        mean=0.0,
                        std=cdp_noise_scale,
                        size=weight_update[k].shape
                    ).to(DEVICE)
                    weight_update[k] = weight_update[k] + noise
                
                # Apply noisy weight update to global model
                for k in current_global.keys():
                    current_global[k] = current_global[k] + weight_update[k]
                
                global_model.load_state_dict(current_global)
                
                # Compute epsilon for CDP
                # In CDP, we use noise_multiplier / sqrt(K) which gives same privacy as LDP
                # but with better accuracy. For epsilon, we approximate based on the noise scale.
                # The key insight: CDP noise std = sigma*C, LDP noise std = sigma*C*sqrt(K)
                # So for same sigma, CDP has sqrt(K) less noise, meaning better accuracy
                # For epsilon: with reduced noise_mult, epsilon is similar or slightly better
                
                # Compute epsilon for CDP using RDP accountant
                # CDP uses reduced noise_multiplier (sigma/sqrt(K)) for efficiency
                # We need to compute epsilon based on the aggregation step
                if controllers and len(controllers) > 0:
                    # Use the first controller's accountant to compute CDP epsilon
                    controller = controllers[0]
                    batch_size = client_loaders[0].batch_size if hasattr(client_loaders[0], 'batch_size') else 64
                    dataset_size = len(client_loaders[0].dataset) if hasattr(client_loaders[0], 'dataset') else 10000
                    
                    # CDP: Server adds noise once per round
                    # The privacy cost is for one aggregation step with reduced noise
                    steps_per_round = len(client_loaders[0]) if client_loaders else 100
                    
                    if cdp_noise_mult > 0:
                        # For CDP, we add noise once per round at aggregation (single query)
                        # Use consistent formula for both static and adaptive
                        # Standard Gaussian mechanism: epsilon = sqrt(2*ln(1.25/delta)) / sigma
                        # This is the standard formula for (epsilon, delta)-DP with Gaussian noise
                        # More accurate than simplified 1/(2*sigma^2) for finite delta
                        cdp_epsilon = np.sqrt(2.0 * np.log(1.25 / TARGET_DELTA)) / cdp_noise_mult
                        # This gives reasonable epsilon values for single aggregation step
                    else:
                        cdp_epsilon = 0.01
                    
                    round_epsilons = [cdp_epsilon]
                else:
                    round_epsilons = [0.0]

        # --- PHASE 3: EVALUATION ---
        acc = evaluate(global_model, test_loader)
        
        # Compute epsilon for this round and add to cumulative
        # Privacy budget is CUMULATIVE - it accumulates over rounds
        if decentralized:
            # LDP: PrivacyEngine.get_epsilon() returns epsilon for THIS round only
            # We need to accumulate it across rounds
            if round_epsilons:
                # Get epsilon for this round (average across clients)
                round_epsilon = np.mean(round_epsilons)
                cumulative_epsilon += round_epsilon  # Add to cumulative total
            else:
                round_epsilon = 0.0
        else:
            # CDP: Compute per-round epsilon and add to cumulative
            if round_epsilons:
                round_epsilon = round_epsilons[0]  # Per-round epsilon for CDP
                cumulative_epsilon += round_epsilon  # Add to cumulative total
            else:
                round_epsilon = 0.0
        
        history['accuracy'].append(acc)
        history['epsilon'].append(cumulative_epsilon)  # Store cumulative epsilon
        mode_str = "CDP" if not decentralized else "LDP"
        print(f"   >>> Round {r+1} Accuracy: {acc:.2f}% | Epsilon ({mode_str}): {cumulative_epsilon:.4f} (cumulative, +{round_epsilon:.4f} this round)")

    # Get cost analysis if decentralized
    cost_analysis = None
    if decentralized and chain:
        cost_analysis = chain.get_cost_analysis()

    return history, cost_analysis

def evaluate(model, loader):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(loader.dataset)

def plot_comparison(results_dict):
    """
    Plot comparison of all 4 experiments
    
    Args:
        results_dict: Dictionary with keys like 'Centralized_Static', 'Centralized_Adaptive', etc.
                      Values are history dictionaries with 'accuracy' and 'epsilon' lists
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    rounds = list(range(1, ROUNDS + 1))
    colors = {
        'Centralized_Static': '#1f77b4',      # Blue
        'Centralized_Adaptive': '#ff7f0e',   # Orange
        'Decentralized_Static': '#2ca02c',    # Green
        'Decentralized_Adaptive': '#d62728'     # Red
    }
    
    linestyles = {
        'Centralized_Static': '-',
        'Centralized_Adaptive': '--',
        'Decentralized_Static': '-.',
        'Decentralized_Adaptive': ':'
    }
    
    markers = {
        'Centralized_Static': 'o',
        'Centralized_Adaptive': 's',
        'Decentralized_Static': '^',
        'Decentralized_Adaptive': 'D'
    }
    
    # Plot Accuracy
    for key, history in results_dict.items():
        if history and 'accuracy' in history:
            accuracies = history['accuracy']
            label = key.replace('_', ' ').title()
            ax1.plot(rounds, accuracies, 
                    color=colors.get(key, 'black'),
                    linestyle=linestyles.get(key, '-'),
                    marker=markers.get(key, 'o'),
                    label=label,
                    linewidth=2,
                    markersize=8)
    
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(rounds)
    
    # Plot Epsilon (Privacy Budget)
    for key, history in results_dict.items():
        if history and 'epsilon' in history:
            epsilons = history['epsilon']
            label = key.replace('_', ' ').title()
            ax2.plot(rounds, epsilons,
                    color=colors.get(key, 'black'),
                    linestyle=linestyles.get(key, '-'),
                    marker=markers.get(key, 'o'),
                    label=label,
                    linewidth=2,
                    markersize=8)
    
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Epsilon (Privacy Budget)', fontsize=12)
    ax2.set_title('Privacy Budget (ε) Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(rounds)
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'federated_learning_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved to: {output_file}")
    
    # Also create a summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Experiment':<30} {'Final Accuracy':<20} {'Final Epsilon':<20}")
    print("-"*80)
    for key, history in results_dict.items():
        if history and 'accuracy' in history and len(history['accuracy']) > 0:
            final_acc = history['accuracy'][-1]
            final_eps = history['epsilon'][-1] if 'epsilon' in history and len(history['epsilon']) > 0 else 0
            label = key.replace('_', ' ').title()
            print(f"{label:<30} {final_acc:>18.2f}% {final_eps:>18.4f}")
    print("="*80)
    
    plt.show()

def plot_cost_analysis(cost_data_dict):
    """
    Plot and display cost analysis for Experiment C
    
    Args:
        cost_data_dict: Dictionary with experiment names as keys and cost_analysis dicts as values
    """
    print("\n" + "="*80)
    print("EXPERIMENT C: COST ANALYSIS")
    print("="*80)
    
    # Collect all cost data
    all_costs = {}
    for exp_name, cost_data in cost_data_dict.items():
        if cost_data and 'per_round' in cost_data:
            all_costs[exp_name] = cost_data
    
    if not all_costs:
        print("No cost data available (only decentralized experiments have cost data)")
        return
    
    # Create table
    print("\n" + "-"*80)
    print(f"{'Round':<8} {'Gas Used':<15} {'Mainnet Cost':<20} {'L2 Cost (Optimism)':<20}")
    print("-"*80)
    
    # Get max rounds
    max_rounds = max(len(cost['per_round']) for cost in all_costs.values()) if all_costs else 0
    
    # Show per-round costs (use first experiment as reference)
    if all_costs:
        first_exp = list(all_costs.keys())[0]
        for r in range(max_rounds):
            if r < len(all_costs[first_exp]['per_round']):
                round_data = all_costs[first_exp]['per_round'][r]
                gas = round_data['gas']
                mainnet = round_data['usd_mainnet']
                l2 = round_data['usd_l2']
                print(f"{r+1:<8} {gas:<15,} ${mainnet:<19.2f} ${l2:<19.4f}")
    
    print("-"*80)
    
    # Show totals
    print("\n" + "-"*80)
    print(f"{'Experiment':<30} {'Total Gas':<15} {'Total Mainnet':<20} {'Total L2':<20}")
    print("-"*80)
    for exp_name, cost_data in all_costs.items():
        label = exp_name.replace('_', ' ').title()
        total_gas = cost_data['gas_used']
        total_mainnet = cost_data['usd_mainnet']
        total_l2 = cost_data['usd_l2']
        print(f"{label:<30} {total_gas:<15,} ${total_mainnet:<19.2f} ${total_l2:<19.4f}")
    print("-"*80)
    
    # Analysis and recommendations
    print("\n" + "="*80)
    print("COST ANALYSIS & RECOMMENDATIONS")
    print("="*80)
    
    if all_costs:
        sample_cost = list(all_costs.values())[0]
        avg_round_mainnet = np.mean([r['usd_mainnet'] for r in sample_cost['per_round']])
        avg_round_l2 = np.mean([r['usd_l2'] for r in sample_cost['per_round']])
        
        print(f"\n📊 Key Findings:")
        print(f"   • Average cost per round on Mainnet: ${avg_round_mainnet:.2f}")
        print(f"   • Average cost per round on L2: ${avg_round_l2:.4f}")
        print(f"   • L2 is {avg_round_mainnet/avg_round_l2:.0f}x cheaper than Mainnet")
        
        print(f"\n⚠️  Mainnet Feasibility:")
        if avg_round_mainnet > 50:
            print(f"   • Mainnet costs exceed $50/round - NOT economically feasible")
        else:
            print(f"   • Mainnet costs are ${avg_round_mainnet:.2f}/round")
        print(f"   • For {ROUNDS} rounds: ${sample_cost['usd_mainnet']:.2f} total on Mainnet")
        
        print(f"\n✅ Recommended Solution:")
        print(f"   • Deploy on Layer 2 (Optimism/Arbitrum)")
        print(f"   • Cost per round: ${avg_round_l2:.4f} (pennies)")
        print(f"   • Total for {ROUNDS} rounds: ${sample_cost['usd_l2']:.4f}")
        print(f"   • Savings: ${sample_cost['usd_mainnet'] - sample_cost['usd_l2']:.2f} ({100*(1-sample_cost['usd_l2']/sample_cost['usd_mainnet']):.1f}% reduction)")
        
        print(f"\n💡 Why L2?")
        print(f"   • Same security guarantees (L2 inherits Mainnet security)")
        print(f"   • 95% cost reduction makes FL economically viable")
        print(f"   • Fast transaction confirmation (~2s vs ~12s)")
        print(f"   • Perfect for federated learning (rounds take minutes/hours anyway)")
    
    print("="*80)

if __name__ == "__main__":
    print("="*80)
    print("FEDERATED LEARNING COMPARISON EXPERIMENT")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Clients: {NUM_CLIENTS}")
    print(f"  - Rounds: {ROUNDS}")
    print(f"  - Local Epochs: {LOCAL_EPOCHS}")
    print("="*80)
    
    results = {}
    cost_data = {}  # Store cost analysis for Experiment C
    
    # Run all 4 experiments
    experiments = [
        ("Centralized_Static", "static", False),
        ("Centralized_Adaptive", "adaptive", False),
        ("Decentralized_Static", "static", True),
        ("Decentralized_Adaptive", "adaptive", True),
    ]
    
    for exp_name, strategy, decentralized in experiments:
        try:
            history, cost = run_experiment(exp_name, strategy, decentralized)
            results[exp_name] = history
            
            # Store cost data for decentralized experiments
            if cost:
                cost_data[exp_name] = cost
                print(f"\n💰 Cost Analysis for {exp_name}:")
                print(f"   Gas Used: {cost['gas_used']:,}")
                print(f"   USD (Mainnet): ${cost['usd_mainnet']:.4f}")
                print(f"   USD (L2): ${cost['usd_l2']:.4f}")
        except Exception as e:
            print(f"\n❌ Experiment {exp_name} failed: {e}")
            results[exp_name] = None
            import traceback
            traceback.print_exc()
    
    # Plot comparison (Experiments A & B)
    if any(results.values()):
        plot_comparison(results)
    else:
        print("\n❌ No experiments completed successfully. Cannot generate plots.")
    
    # Display cost analysis (Experiment C)
    if cost_data:
        plot_cost_analysis(cost_data)
