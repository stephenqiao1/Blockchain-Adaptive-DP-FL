#!/usr/bin/env python3
"""
Test script for RDP Accountant implementation
Tests the RDPAccountantCustom class functionality
"""

import sys
import os
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from adaptive_privacy import RDPAccountantCustom

def test_rdp_accountant_basic():
    """Test basic RDP accountant functionality"""
    print("=" * 60)
    print("Test 1: Basic RDP Accountant")
    print("=" * 60)
    try:
        accountant = RDPAccountantCustom()
        
        # Record some steps
        accountant.step(noise_multiplier=1.0, sample_rate=0.01, steps=100)
        
        epsilon = accountant.get_epsilon(delta=1e-5)
        
        assert epsilon > 0, "Epsilon should be positive"
        assert epsilon < 100, "Epsilon should be reasonable"
        
        print("✓ Basic RDP accountant test passed")
        print(f"  - Epsilon (delta=1e-5): {epsilon:.4f}")
    except Exception as e:
        print(f"✗ Basic RDP accountant test failed: {e}")
        raise

def test_rdp_accountant_accumulation():
    """Test that RDP accountant accumulates privacy budget"""
    print("\n" + "=" * 60)
    print("Test 2: RDP Accountant Accumulation")
    print("=" * 60)
    try:
        accountant = RDPAccountantCustom()
        
        # Record steps in multiple rounds
        for _ in range(3):
            accountant.step(noise_multiplier=1.0, sample_rate=0.01, steps=10)
        
        epsilon_after_30 = accountant.get_epsilon(delta=1e-5)
        
        # Add more steps
        accountant.step(noise_multiplier=1.0, sample_rate=0.01, steps=10)
        epsilon_after_40 = accountant.get_epsilon(delta=1e-5)
        
        # Epsilon should increase with more steps
        assert epsilon_after_40 > epsilon_after_30, "Epsilon should increase with more steps"
        
        print("✓ RDP accountant accumulation test passed")
        print(f"  - After 30 steps: {epsilon_after_30:.4f}")
        print(f"  - After 40 steps: {epsilon_after_40:.4f}")
    except Exception as e:
        print(f"✗ RDP accountant accumulation test failed: {e}")
        raise

def test_rdp_accountant_adaptive_noise():
    """Test RDP accountant with adaptive noise multipliers"""
    print("\n" + "=" * 60)
    print("Test 3: RDP Accountant with Adaptive Noise")
    print("=" * 60)
    try:
        accountant = RDPAccountantCustom()
        
        # Simulate adaptive training: noise decreases over time
        noise_levels = [1.0, 0.95, 0.90, 0.85, 0.80]
        epsilons = []
        
        for noise in noise_levels:
            accountant.step(noise_multiplier=noise, sample_rate=0.01, steps=10)
            epsilon = accountant.get_epsilon(delta=1e-5)
            epsilons.append(epsilon)
            print(f"  - Noise: {noise:.2f} -> Epsilon: {epsilon:.4f}")
        
        # Epsilon should increase (more privacy spent)
        assert epsilons[-1] > epsilons[0], "Epsilon should increase with more steps"
        
        # Lower noise should lead to higher epsilon per step
        # (less noise = less privacy, so higher epsilon cost)
        print("✓ Adaptive noise RDP accountant test passed")
        print(f"  - Final epsilon: {epsilons[-1]:.4f}")
    except Exception as e:
        print(f"✗ Adaptive noise RDP accountant test failed: {e}")
        raise

def test_rdp_accountant_sample_rate():
    """Test that sample rate affects privacy budget"""
    print("\n" + "=" * 60)
    print("Test 4: RDP Accountant Sample Rate Effect")
    print("=" * 60)
    try:
        # Test with different sample rates
        sample_rates = [0.001, 0.01, 0.1, 1.0]
        epsilons = []
        
        for q in sample_rates:
            accountant = RDPAccountantCustom()
            accountant.step(noise_multiplier=1.0, sample_rate=q, steps=100)
            epsilon = accountant.get_epsilon(delta=1e-5)
            epsilons.append(epsilon)
            print(f"  - Sample rate: {q:.3f} -> Epsilon: {epsilon:.4f}")
        
        # Lower sample rate should lead to lower epsilon (privacy amplification)
        # Higher sample rate should lead to higher epsilon
        assert epsilons[-1] > epsilons[0], "Higher sample rate should increase epsilon"
        
        print("✓ Sample rate effect test passed")
        print("  - Lower sample rate = lower epsilon (privacy amplification)")
    except Exception as e:
        print(f"✗ Sample rate effect test failed: {e}")
        raise

def test_rdp_accountant_reset():
    """Test reset functionality"""
    print("\n" + "=" * 60)
    print("Test 5: RDP Accountant Reset")
    print("=" * 60)
    try:
        accountant = RDPAccountantCustom()
        
        # Add some steps
        accountant.step(noise_multiplier=1.0, sample_rate=0.01, steps=100)
        epsilon_before = accountant.get_epsilon(delta=1e-5)
        assert epsilon_before > 0, "Should have epsilon before reset"
        
        # Reset
        accountant.reset()
        epsilon_after = accountant.get_epsilon(delta=1e-5)
        assert epsilon_after == 0.0, "Epsilon should be 0 after reset"
        
        print("✓ Reset test passed")
        print(f"  - Before reset: {epsilon_before:.4f}")
        print(f"  - After reset: {epsilon_after:.4f}")
    except Exception as e:
        print(f"✗ Reset test failed: {e}")
        raise

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("RDP ACCOUNTANT TESTS")
    print("=" * 60)
    
    try:
        test_rdp_accountant_basic()
        test_rdp_accountant_accumulation()
        test_rdp_accountant_adaptive_noise()
        test_rdp_accountant_sample_rate()
        test_rdp_accountant_reset()
        
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

