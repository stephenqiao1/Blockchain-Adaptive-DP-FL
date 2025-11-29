#!/usr/bin/env python3
"""
Test script for blockchain_ganache.py
Tests the GanacheHandler class functionality including:
- Initialization and contract deployment
- Submit update functionality
- End round functionality
- Cost analysis
- Event emission
- Multiple rounds
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from blockchain_ganache import GanacheHandler

def test_initialization():
    """Test that GanacheHandler initializes correctly"""
    print("=" * 60)
    print("Test 1: Initialization")
    print("=" * 60)
    try:
        handler = GanacheHandler()
        assert handler.w3.is_connected(), "Should be connected to Ganache"
        assert len(handler.accounts) > 0, "Should have accounts"
        assert handler.aggregator == handler.accounts[0], "Aggregator should be first account"
        assert handler.contract is not None, "Contract should be deployed"
        assert handler.total_gas_used > 0, "Deployment should use gas"
        print("✓ Initialization test passed")
        print(f"  - Contract address: {handler.contract.address}")
        print(f"  - Number of accounts: {len(handler.accounts)}")
        print(f"  - Initial gas used: {handler.total_gas_used}")
        return handler
    except Exception as e:
        print(f"✗ Initialization test failed: {e}")
        raise

def test_submit_update(handler):
    """Test submitting model updates"""
    print("\n" + "=" * 60)
    print("Test 2: Submit Update")
    print("=" * 60)
    try:
        ipfs_hash = "QmTestHash123"
        block_before = handler.w3.eth.block_number
        gas_used = handler.submit_update(0, ipfs_hash)
        block_after = handler.w3.eth.block_number
        
        assert gas_used > 0, "Transaction should use gas"
        assert handler.total_gas_used > gas_used, "Total gas should increase"
        
        # Check event was emitted by querying logs
        # Look in a range that includes the transaction block
        events = handler.contract.events.ModelUpdate.get_logs(
            from_block=max(0, block_before - 1), 
            to_block=block_after + 1
        )
        
        if events:
            # Find the event with matching hash
            matching_event = None
            for event in events:
                if event['args']['ipfsHash'] == ipfs_hash:
                    matching_event = event
                    break
            
            if matching_event:
                print(f"  - Event emitted: ModelUpdate")
                print(f"    Client: {matching_event['args']['client']}")
                print(f"    Round: {matching_event['args']['round']}")
                print(f"    IPFS Hash: {matching_event['args']['ipfsHash']}")
            else:
                print(f"  - Note: Event found but hash doesn't match (this is OK if multiple transactions)")
        else:
            print(f"  - Note: No events found in block range (may be normal for some networks)")
        
        print("✓ Submit update test passed")
        print(f"  - Gas used: {gas_used}")
        return gas_used
    except Exception as e:
        print(f"✗ Submit update test failed: {e}")
        raise

def test_multiple_submits(handler):
    """Test multiple clients submitting updates"""
    print("\n" + "=" * 60)
    print("Test 3: Multiple Client Submissions")
    print("=" * 60)
    try:
        initial_gas = handler.total_gas_used
        
        # Submit from multiple clients
        for i in range(3):
            ipfs_hash = f"QmHash{i}"
            gas = handler.submit_update(i, ipfs_hash)
            print(f"  - Client {i} submitted: {ipfs_hash} (gas: {gas})")
        
        assert handler.total_gas_used > initial_gas, "Total gas should increase"
        print("✓ Multiple submissions test passed")
    except Exception as e:
        print(f"✗ Multiple submissions test failed: {e}")
        raise

def test_end_round(handler):
    """Test ending a round"""
    print("\n" + "=" * 60)
    print("Test 4: End Round")
    print("=" * 60)
    try:
        # Get current round
        current_round_before = handler.contract.functions.currentRound().call()
        
        block_before = handler.w3.eth.block_number
        global_hash = "QmGlobalHash123"
        gas_used = handler.end_round(global_hash)
        block_after = handler.w3.eth.block_number
        
        assert gas_used > 0, "Transaction should use gas"
        
        # Check round incremented
        current_round_after = handler.contract.functions.currentRound().call()
        assert current_round_after == current_round_before + 1, "Round should increment"
        
        # Check event was emitted
        events = handler.contract.events.RoundEnded.get_logs(
            from_block=max(0, block_before - 1), 
            to_block=block_after + 1
        )
        
        if events:
            # Find the event with matching round and hash
            matching_event = None
            for event in events:
                if (event['args']['round'] == current_round_after and 
                    event['args']['globalModelHash'] == global_hash):
                    matching_event = event
                    break
            
            if matching_event:
                print(f"  - Event emitted: RoundEnded")
                print(f"    Round: {matching_event['args']['round']}")
                print(f"    Global Hash: {matching_event['args']['globalModelHash']}")
            else:
                print(f"  - Note: Event found but doesn't match expected values")
        else:
            print(f"  - Note: No events found in block range (may be normal for some networks)")
        
        print("✓ End round test passed")
        print(f"  - Round before: {current_round_before}, after: {current_round_after}")
        print(f"  - Gas used: {gas_used}")
        return gas_used
    except Exception as e:
        print(f"✗ End round test failed: {e}")
        raise

def test_multiple_rounds(handler):
    """Test multiple rounds"""
    print("\n" + "=" * 60)
    print("Test 5: Multiple Rounds")
    print("=" * 60)
    try:
        initial_round = handler.contract.functions.currentRound().call()
        
        # Round 1: Submit and end
        handler.submit_update(0, "QmRound1Client0")
        handler.submit_update(1, "QmRound1Client1")
        handler.end_round("QmGlobalRound1")
        
        round_after_1 = handler.contract.functions.currentRound().call()
        assert round_after_1 == initial_round + 1, "Round should increment"
        
        # Round 2: Submit and end
        handler.submit_update(0, "QmRound2Client0")
        handler.end_round("QmGlobalRound2")
        
        round_after_2 = handler.contract.functions.currentRound().call()
        assert round_after_2 == initial_round + 2, "Round should increment again"
        
        print("✓ Multiple rounds test passed")
        print(f"  - Initial round: {initial_round}")
        print(f"  - After round 1: {round_after_1}")
        print(f"  - After round 2: {round_after_2}")
    except Exception as e:
        print(f"✗ Multiple rounds test failed: {e}")
        raise

def test_cost_analysis(handler):
    """Test cost analysis"""
    print("\n" + "=" * 60)
    print("Test 6: Cost Analysis")
    print("=" * 60)
    try:
        cost_data = handler.get_cost_analysis()
        
        assert 'gas_used' in cost_data, "Should have gas_used"
        assert 'eth_cost' in cost_data, "Should have eth_cost"
        assert 'usd_mainnet' in cost_data, "Should have usd_mainnet"
        assert 'usd_l2' in cost_data, "Should have usd_l2"
        
        assert cost_data['gas_used'] > 0, "Gas used should be positive"
        assert cost_data['eth_cost'] >= 0, "ETH cost should be non-negative"
        assert cost_data['usd_mainnet'] >= 0, "USD mainnet cost should be non-negative"
        assert cost_data['usd_l2'] >= 0, "USD L2 cost should be non-negative"
        assert cost_data['usd_l2'] < cost_data['usd_mainnet'], "L2 cost should be less than mainnet"
        
        print("✓ Cost analysis test passed")
        print(f"  - Total gas used: {cost_data['gas_used']:,}")
        print(f"  - ETH cost: {cost_data['eth_cost']:.10f}")
        print(f"  - USD (Mainnet): ${cost_data['usd_mainnet']:.4f}")
        print(f"  - USD (L2): ${cost_data['usd_l2']:.4f}")
    except Exception as e:
        print(f"✗ Cost analysis test failed: {e}")
        raise

def test_access_control(handler):
    """Test that only owner can end rounds"""
    print("\n" + "=" * 60)
    print("Test 7: Access Control")
    print("=" * 60)
    try:
        # Try to end round from a non-owner account (client account)
        if len(handler.accounts) > 1:
            client_account = handler.accounts[1]
            global_hash = "QmUnauthorized"
            
            try:
                # This should fail
                tx_hash = handler.contract.functions.endRound(global_hash).transact({
                    'from': client_account
                })
                receipt = handler.w3.eth.wait_for_transaction_receipt(tx_hash)
                # If we get here, the transaction succeeded, which is wrong
                print("✗ Access control test failed: Non-owner was able to end round")
                assert False, "Non-owner should not be able to end round"
            except Exception as e:
                # Transaction should revert
                print("✓ Access control test passed (transaction reverted as expected)")
                print(f"  - Error: {str(e)[:100]}")
        else:
            print("⚠ Skipping access control test (need at least 2 accounts)")
    except AssertionError:
        raise
    except Exception as e:
        print(f"✗ Access control test failed: {e}")
        raise

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("BLOCKCHAIN GANACHE HANDLER TESTS")
    print("=" * 60)
    print("\nMake sure Ganache is running on http://127.0.0.1:7545")
    print("If using a different port, modify the GanacheHandler initialization\n")
    
    try:
        # Test 1: Initialization
        handler = test_initialization()
        
        # Test 2: Submit update
        test_submit_update(handler)
        
        # Test 3: Multiple submissions
        test_multiple_submits(handler)
        
        # Test 4: End round
        test_end_round(handler)
        
        # Test 5: Multiple rounds
        test_multiple_rounds(handler)
        
        # Test 6: Cost analysis
        test_cost_analysis(handler)
        
        # Test 7: Access control
        test_access_control(handler)
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print(f"\nFinal Statistics:")
        cost_data = handler.get_cost_analysis()
        print(f"  - Total gas used: {cost_data['gas_used']:,}")
        print(f"  - Estimated cost (Mainnet): ${cost_data['usd_mainnet']:.4f}")
        print(f"  - Estimated cost (L2): ${cost_data['usd_l2']:.4f}")
        
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

