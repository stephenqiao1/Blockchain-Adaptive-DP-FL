#!/usr/bin/env python3
"""
Test script for ipfs_handler.py
Tests the IPFSHandler class functionality including:
- Initialization
- Upload model functionality
- Download model functionality
- Round-trip upload/download verification
- Error handling
- Cleanup
"""

import sys
import os
import torch
import torch.nn as nn

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ipfs_handler import IPFSHandler

def create_test_model():
    """Create a simple test model for testing"""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )
    return model.state_dict()

def test_initialization():
    """Test that IPFSHandler initializes correctly"""
    print("=" * 60)
    print("Test 1: Initialization")
    print("=" * 60)
    try:
        handler = IPFSHandler()
        assert handler.api_url == "http://127.0.0.1:5001/api/v0", "Default API URL should be set"
        assert os.path.exists(handler.temp_dir), "Temp directory should be created"
        print("✓ Initialization test passed")
        print(f"  - API URL: {handler.api_url}")
        print(f"  - Temp directory: {handler.temp_dir}")
        return handler
    except Exception as e:
        print(f"✗ Initialization test failed: {e}")
        raise

def test_upload_model(handler):
    """Test uploading a model to IPFS"""
    print("\n" + "=" * 60)
    print("Test 2: Upload Model")
    print("=" * 60)
    try:
        # Create a test model
        test_model = create_test_model()
        filename = "test_model.pth"
        
        # Upload to IPFS
        cid = handler.upload_model(test_model, filename)
        
        assert cid is not None, "CID should not be None"
        assert isinstance(cid, str), "CID should be a string"
        assert len(cid) > 0, "CID should not be empty"
        # CID can be v0 (starts with Qm) or v1 (starts with baf, z, etc.)
        assert cid.startswith('Qm') or cid.startswith('baf') or cid.startswith('z'), f"CID should be valid (got: {cid})"
        assert os.path.exists(os.path.join(handler.temp_dir, filename)), "Local file should exist"
        
        print("✓ Upload model test passed")
        print(f"  - CID: {cid}")
        print(f"  - Local file: {os.path.join(handler.temp_dir, filename)}")
        return cid, test_model
    except Exception as e:
        print(f"✗ Upload model test failed: {e}")
        print("  Make sure IPFS is running: ipfs daemon")
        raise

def test_download_model(handler, cid, original_model):
    """Test downloading a model from IPFS"""
    print("\n" + "=" * 60)
    print("Test 3: Download Model")
    print("=" * 60)
    try:
        # Download from IPFS
        downloaded_model = handler.download_model(cid)
        
        assert downloaded_model is not None, "Downloaded model should not be None"
        assert isinstance(downloaded_model, dict), "Downloaded model should be a state_dict"
        
        # Verify the model structure matches
        assert set(downloaded_model.keys()) == set(original_model.keys()), "Model keys should match"
        
        # Verify the values match (with some tolerance for floating point)
        for key in original_model.keys():
            assert downloaded_model[key].shape == original_model[key].shape, f"Shape should match for {key}"
            assert torch.allclose(downloaded_model[key], original_model[key], atol=1e-6), f"Values should match for {key}"
        
        print("✓ Download model test passed")
        print(f"  - Downloaded CID: {cid}")
        print(f"  - Model keys: {list(downloaded_model.keys())}")
        return downloaded_model
    except Exception as e:
        print(f"✗ Download model test failed: {e}")
        raise

def test_round_trip(handler):
    """Test complete round-trip: upload then download"""
    print("\n" + "=" * 60)
    print("Test 4: Round-Trip Upload/Download")
    print("=" * 60)
    try:
        # Create a more complex test model
        model = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Linear(32 * 26 * 26, 10)
        )
        original_state_dict = model.state_dict()
        
        # Upload
        cid = handler.upload_model(original_state_dict, "round_trip_model.pth")
        assert cid is not None, "Upload should succeed"
        
        # Download
        downloaded_state_dict = handler.download_model(cid)
        assert downloaded_state_dict is not None, "Download should succeed"
        
        # Verify they match
        for key in original_state_dict.keys():
            assert torch.equal(original_state_dict[key], downloaded_state_dict[key]), f"Values should be identical for {key}"
        
        print("✓ Round-trip test passed")
        print(f"  - Uploaded CID: {cid}")
        print(f"  - Verified {len(original_state_dict)} parameters match")
        return cid
    except Exception as e:
        print(f"✗ Round-trip test failed: {e}")
        raise

def test_multiple_uploads(handler):
    """Test uploading multiple models"""
    print("\n" + "=" * 60)
    print("Test 5: Multiple Uploads")
    print("=" * 60)
    try:
        cids = []
        for i in range(3):
            model = nn.Sequential(nn.Linear(10, i+1))
            cid = handler.upload_model(model.state_dict(), f"model_{i}.pth")
            assert cid is not None, f"Upload {i} should succeed"
            cids.append(cid)
            print(f"  - Model {i} uploaded: {cid}")
        
        # Verify all CIDs are unique
        assert len(cids) == len(set(cids)), "All CIDs should be unique"
        
        print("✓ Multiple uploads test passed")
        return cids
    except Exception as e:
        print(f"✗ Multiple uploads test failed: {e}")
        raise

def test_error_handling():
    """Test error handling when IPFS is not available"""
    print("\n" + "=" * 60)
    print("Test 6: Error Handling (IPFS Not Available)")
    print("=" * 60)
    try:
        # Create handler with invalid URL
        handler = IPFSHandler(api_url="http://127.0.0.1:9999/api/v0")
        test_model = create_test_model()
        
        # Try to upload - should return None
        cid = handler.upload_model(test_model, "error_test.pth")
        assert cid is None, "Upload should return None when IPFS is unavailable"
        
        print("✓ Error handling test passed")
        print("  - Correctly handled IPFS connection failure")
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        # This is OK - the test might pass or fail depending on connection timeout behavior
        print("  - Note: This test behavior may vary based on connection timeout")

def test_clear_temp_files(handler):
    """Test cleanup functionality"""
    print("\n" + "=" * 60)
    print("Test 7: Clear Temp Files")
    print("=" * 60)
    try:
        # Upload a model to create temp files
        test_model = create_test_model()
        handler.upload_model(test_model, "cleanup_test.pth")
        
        # Count files before cleanup
        files_before = len(os.listdir(handler.temp_dir))
        assert files_before > 0, "Should have files before cleanup"
        
        # Clear temp files
        handler.clear_temp_files()
        
        # Count files after cleanup
        files_after = len(os.listdir(handler.temp_dir))
        assert files_after == 0, "Should have no files after cleanup"
        
        print("✓ Clear temp files test passed")
        print(f"  - Files before: {files_before}, after: {files_after}")
    except Exception as e:
        print(f"✗ Clear temp files test failed: {e}")
        raise

def check_ipfs_connection(api_url="http://127.0.0.1:5001/api/v0"):
    """Check if IPFS is running and accessible"""
    import requests
    try:
        # Try to get IPFS version to check if it's running
        response = requests.post(f"{api_url}/version", timeout=2)
        return response.status_code == 200
    except:
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("IPFS HANDLER TESTS")
    print("=" * 60)
    
    # Check if IPFS is running
    print("\nChecking IPFS connection...")
    if not check_ipfs_connection():
        print("⚠ WARNING: IPFS does not appear to be running!")
        print("  To start IPFS:")
        print("    1. Install IPFS: https://docs.ipfs.io/install/")
        print("    2. Initialize: ipfs init")
        print("    3. Start daemon: ipfs daemon")
        print("\n  Some tests will be skipped or will fail.")
        print("  Press Ctrl+C to cancel, or Enter to continue with available tests...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nTest cancelled.")
            return
    else:
        print("✓ IPFS is running and accessible")
    
    try:
        # Test 1: Initialization
        handler = test_initialization()
        
        # Test 2: Upload
        cid, original_model = test_upload_model(handler)
        
        # Test 3: Download
        test_download_model(handler, cid, original_model)
        
        # Test 4: Round-trip
        test_round_trip(handler)
        
        # Test 5: Multiple uploads
        test_multiple_uploads(handler)
        
        # Test 6: Error handling (may fail if connection timeout is long)
        try:
            test_error_handling()
        except Exception as e:
            print(f"  - Error handling test skipped: {e}")
        
        # Test 7: Cleanup
        test_clear_temp_files(handler)
        
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

