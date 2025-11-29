#!/usr/bin/env python3
"""
Test script for adaptive_privacy.py
Tests the AdaptivePrivacyController class functionality including:
- Initialization with static and adaptive strategies
- Static strategy behavior (no parameter changes)
- Adaptive strategy with decreasing loss (noise reduction)
- Adaptive strategy with increasing loss (noise increase)
- Adaptive strategy with loss spike (clipping adjustment)
- Boundary conditions (parameter limits)
- Privacy spent calculation
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from adaptive_privacy import AdaptivePrivacyController

def create_mock_model():
    """Create a simple mock model for testing"""
    return nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )

def create_mock_dataloader(num_samples=100, batch_size=32):
    """Create a mock dataloader for testing"""
    # Create dummy data
    X = torch.randn(num_samples, 10)
    y = torch.randint(0, 2, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def test_initialization():
    """Test that AdaptivePrivacyController initializes correctly"""
    print("=" * 60)
    print("Test 1: Initialization")
    print("=" * 60)
    try:
        model = create_mock_model()
        dataloader = create_mock_dataloader()
        
        # Test static strategy
        controller_static = AdaptivePrivacyController(
            model=model,
            dataloader=dataloader,
            base_epsilon_target=10.0,
            epochs=5,
            strategy='static'
        )
        
        assert controller_static.strategy == 'static', "Strategy should be 'static'"
        assert controller_static.noise_multiplier == 1.0, "Initial noise_multiplier should be 1.0"
        assert controller_static.max_grad_norm == 1.0, "Initial max_grad_norm should be 1.0"
        assert controller_static.prev_loss == float('inf'), "Initial prev_loss should be inf"
        assert controller_static.patience_counter == 0, "Initial patience_counter should be 0"
        
        # Test adaptive strategy
        controller_adaptive = AdaptivePrivacyController(
            model=model,
            dataloader=dataloader,
            base_epsilon_target=10.0,
            epochs=5,
            strategy='adaptive'
        )
        
        assert controller_adaptive.strategy == 'adaptive', "Strategy should be 'adaptive'"
        assert controller_adaptive.noise_multiplier == 1.0, "Initial noise_multiplier should be 1.0"
        assert controller_adaptive.max_grad_norm == 1.0, "Initial max_grad_norm should be 1.0"
        
        print("✓ Initialization test passed")
        print(f"  - Static strategy: {controller_static.strategy}")
        print(f"  - Adaptive strategy: {controller_adaptive.strategy}")
        print(f"  - Initial noise_multiplier: {controller_static.noise_multiplier}")
        print(f"  - Initial max_grad_norm: {controller_static.max_grad_norm}")
        
        return controller_static, controller_adaptive
    except Exception as e:
        print(f"✗ Initialization test failed: {e}")
        raise

def test_static_strategy(controller):
    """Test that static strategy doesn't change parameters"""
    print("\n" + "=" * 60)
    print("Test 2: Static Strategy")
    print("=" * 60)
    try:
        initial_noise = controller.noise_multiplier
        initial_clip = controller.max_grad_norm
        
        # Call adapt_parameters multiple times with different losses
        losses = [1.0, 0.8, 0.6, 0.9, 1.2]
        for loss in losses:
            noise, clip = controller.adapt_parameters(loss)
            assert noise == initial_noise, "Noise should not change in static strategy"
            assert clip == initial_clip, "Clip should not change in static strategy"
            assert controller.noise_multiplier == initial_noise, "Noise multiplier should remain unchanged"
            assert controller.max_grad_norm == initial_clip, "Max grad norm should remain unchanged"
        
        print("✓ Static strategy test passed")
        print(f"  - Noise multiplier: {initial_noise} (unchanged)")
        print(f"  - Max grad norm: {initial_clip} (unchanged)")
        print(f"  - Tested with {len(losses)} different loss values")
    except Exception as e:
        print(f"✗ Static strategy test failed: {e}")
        raise

def test_adaptive_decreasing_loss(controller):
    """Test adaptive strategy with decreasing loss (should reduce noise)"""
    print("\n" + "=" * 60)
    print("Test 3: Adaptive Strategy - Decreasing Loss")
    print("=" * 60)
    try:
        # Reset controller state
        controller.prev_loss = float('inf')
        controller.noise_multiplier = 1.0
        controller.max_grad_norm = 1.0
        
        # Simulate decreasing loss
        losses = [1.0, 0.9, 0.8, 0.7, 0.6]
        noise_values = []
        
        for loss in losses:
            noise, clip = controller.adapt_parameters(loss)
            noise_values.append(noise)
            print(f"  - Loss: {loss:.2f} -> Noise: {noise:.4f}, Clip: {clip:.4f}")
        
        # Verify noise is decreasing
        for i in range(1, len(noise_values)):
            assert noise_values[i] <= noise_values[i-1], f"Noise should decrease when loss decreases (step {i})"
        
        # Verify noise stays above minimum (0.5)
        assert all(n >= 0.5 for n in noise_values), "Noise should not go below 0.5"
        
        print("✓ Decreasing loss test passed")
        print(f"  - Noise reduced from {noise_values[0]:.4f} to {noise_values[-1]:.4f}")
    except Exception as e:
        print(f"✗ Decreasing loss test failed: {e}")
        raise

def test_adaptive_increasing_loss(controller):
    """Test adaptive strategy with increasing loss (should increase noise)"""
    print("\n" + "=" * 60)
    print("Test 4: Adaptive Strategy - Increasing Loss")
    print("=" * 60)
    try:
        # Reset controller state
        controller.prev_loss = float('inf')
        controller.noise_multiplier = 1.0
        controller.max_grad_norm = 1.0
        
        # Simulate increasing loss
        losses = [0.5, 0.6, 0.7, 0.8, 0.9]
        noise_values = []
        
        for loss in losses:
            noise, clip = controller.adapt_parameters(loss)
            noise_values.append(noise)
            print(f"  - Loss: {loss:.2f} -> Noise: {noise:.4f}, Clip: {clip:.4f}")
        
        # Verify noise is increasing (after first step)
        for i in range(2, len(noise_values)):
            assert noise_values[i] >= noise_values[i-1], f"Noise should increase when loss increases (step {i})"
        
        # Verify noise stays below maximum (2.0)
        assert all(n <= 2.0 for n in noise_values), "Noise should not go above 2.0"
        
        print("✓ Increasing loss test passed")
        print(f"  - Noise increased from {noise_values[0]:.4f} to {noise_values[-1]:.4f}")
    except Exception as e:
        print(f"✗ Increasing loss test failed: {e}")
        raise

def test_adaptive_loss_spike(controller):
    """Test adaptive strategy with loss spike (should tighten clipping)"""
    print("\n" + "=" * 60)
    print("Test 5: Adaptive Strategy - Loss Spike")
    print("=" * 60)
    try:
        # Reset controller state
        controller.prev_loss = 1.0
        controller.noise_multiplier = 1.0
        controller.max_grad_norm = 1.0
        
        # Simulate loss spike (>10% increase)
        spike_loss = 1.0 * 1.1 + 0.01  # 11% increase
        noise_before = controller.noise_multiplier
        clip_before = controller.max_grad_norm
        
        noise_after, clip_after = controller.adapt_parameters(spike_loss)
        
        # Verify clipping is tightened
        assert clip_after < clip_before, "Clip should decrease when loss spikes"
        assert clip_after >= 0.5, "Clip should not go below 0.5"
        
        print("✓ Loss spike test passed")
        print(f"  - Loss spike: {spike_loss:.4f} (11% increase)")
        print(f"  - Clip before: {clip_before:.4f}, after: {clip_after:.4f}")
    except Exception as e:
        print(f"✗ Loss spike test failed: {e}")
        raise

def test_boundary_conditions(controller):
    """Test that parameters stay within boundaries"""
    print("\n" + "=" * 60)
    print("Test 6: Boundary Conditions")
    print("=" * 60)
    try:
        # Test noise_multiplier lower bound
        controller.prev_loss = float('inf')
        controller.noise_multiplier = 0.51  # Just above minimum
        controller.max_grad_norm = 1.0
        
        # Apply many decreasing losses
        for _ in range(10):
            noise, clip = controller.adapt_parameters(0.1)
            assert noise >= 0.5, f"Noise should not go below 0.5 (got {noise})"
        
        # Test noise_multiplier upper bound
        controller.prev_loss = float('inf')
        controller.noise_multiplier = 1.95  # Just below maximum
        controller.max_grad_norm = 1.0
        
        # Apply many increasing losses
        for _ in range(10):
            noise, clip = controller.adapt_parameters(10.0)
            assert noise <= 2.0, f"Noise should not go above 2.0 (got {noise})"
        
        # Test max_grad_norm lower bound
        controller.prev_loss = 0.5
        controller.noise_multiplier = 1.0
        controller.max_grad_norm = 0.51  # Just above minimum
        
        # Apply loss spike
        for _ in range(10):
            noise, clip = controller.adapt_parameters(0.6)  # >10% increase
            assert clip >= 0.5, f"Clip should not go below 0.5 (got {clip})"
        
        print("✓ Boundary conditions test passed")
        print("  - Noise multiplier stays within [0.5, 2.0]")
        print("  - Max grad norm stays >= 0.5")
    except Exception as e:
        print(f"✗ Boundary conditions test failed: {e}")
        raise

def test_patience_counter(controller):
    """Test that patience counter increments correctly"""
    print("\n" + "=" * 60)
    print("Test 7: Patience Counter")
    print("=" * 60)
    try:
        # Reset controller
        controller.prev_loss = float('inf')
        controller.noise_multiplier = 1.0
        controller.max_grad_norm = 1.0
        controller.patience_counter = 0
        
        # First loss (should reset counter)
        controller.adapt_parameters(1.0)
        assert controller.patience_counter == 0, "Counter should be 0 after improvement"
        
        # Increasing loss (should increment counter)
        controller.adapt_parameters(1.1)
        assert controller.patience_counter == 1, "Counter should increment to 1"
        
        controller.adapt_parameters(1.2)
        assert controller.patience_counter == 2, "Counter should increment to 2"
        
        # Improvement (should reset counter)
        controller.adapt_parameters(1.0)
        assert controller.patience_counter == 0, "Counter should reset to 0 after improvement"
        
        print("✓ Patience counter test passed")
        print(f"  - Counter increments on stagnation: ✓")
        print(f"  - Counter resets on improvement: ✓")
    except Exception as e:
        print(f"✗ Patience counter test failed: {e}")
        raise

def test_privacy_spent(controller):
    """Test privacy spent calculation"""
    print("\n" + "=" * 60)
    print("Test 8: Privacy Spent")
    print("=" * 60)
    try:
        # Get privacy spent (should work even if accountant is empty)
        epsilon = controller.get_privacy_spent(delta=1e-5)
        
        # The accountant might return 0.0 if no steps have been tracked
        # This is expected behavior for a fresh accountant
        assert isinstance(epsilon, (int, float)), "Epsilon should be a number"
        assert epsilon >= 0, "Epsilon should be non-negative"
        
        print("✓ Privacy spent test passed")
        print(f"  - Epsilon (delta=1e-5): {epsilon}")
        print(f"  - Note: Epsilon may be 0 if no training steps tracked")
    except Exception as e:
        print(f"✗ Privacy spent test failed: {e}")
        raise

def test_realistic_training_sequence(controller):
    """Test a realistic training sequence with mixed loss patterns"""
    print("\n" + "=" * 60)
    print("Test 9: Realistic Training Sequence")
    print("=" * 60)
    try:
        # Reset controller
        controller.prev_loss = float('inf')
        controller.noise_multiplier = 1.0
        controller.max_grad_norm = 1.0
        controller.patience_counter = 0
        
        # Simulate realistic training: initial high loss, then decreasing, then plateau
        losses = [2.0, 1.5, 1.2, 1.0, 0.9, 0.85, 0.9, 0.88, 0.87, 0.86]
        
        print("  Simulating training sequence:")
        for i, loss in enumerate(losses):
            noise, clip = controller.adapt_parameters(loss)
            trend = "↓" if i > 0 and loss < losses[i-1] else "↑" if i > 0 and loss > losses[i-1] else "="
            print(f"    Step {i+1}: Loss={loss:.2f} {trend} -> Noise={noise:.4f}, Clip={clip:.4f}")
        
        # Verify final parameters are reasonable
        assert 0.5 <= controller.noise_multiplier <= 2.0, "Final noise should be in valid range"
        assert controller.max_grad_norm >= 0.5, "Final clip should be >= 0.5"
        
        print("✓ Realistic training sequence test passed")
        print(f"  - Final noise_multiplier: {controller.noise_multiplier:.4f}")
        print(f"  - Final max_grad_norm: {controller.max_grad_norm:.4f}")
        print(f"  - Final patience_counter: {controller.patience_counter}")
    except Exception as e:
        print(f"✗ Realistic training sequence test failed: {e}")
        raise

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ADAPTIVE PRIVACY CONTROLLER TESTS")
    print("=" * 60)
    
    try:
        # Test 1: Initialization
        controller_static, controller_adaptive = test_initialization()
        
        # Test 2: Static strategy
        test_static_strategy(controller_static)
        
        # Test 3: Adaptive - decreasing loss
        test_adaptive_decreasing_loss(controller_adaptive)
        
        # Test 4: Adaptive - increasing loss
        test_adaptive_increasing_loss(controller_adaptive)
        
        # Test 5: Adaptive - loss spike
        test_adaptive_loss_spike(controller_adaptive)
        
        # Test 6: Boundary conditions
        test_boundary_conditions(controller_adaptive)
        
        # Test 7: Patience counter
        test_patience_counter(controller_adaptive)
        
        # Test 8: Privacy spent
        test_privacy_spent(controller_adaptive)
        
        # Test 9: Realistic training sequence
        test_realistic_training_sequence(controller_adaptive)
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("TESTS FAILED")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

