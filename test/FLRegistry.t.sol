// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "forge-std/Test.sol";
import "../contracts/FLRegistry.sol";

contract FLRegistryTest is Test {
    FLRegistry public registry;
    address public owner;
    address public client1;
    address public client2;

    event ModelUpdate(address indexed client, uint round, string ipfsHash);
    event RoundEnded(uint round, string globalModelHash);

    function setUp() public {
        owner = address(this);
        client1 = address(0x1);
        client2 = address(0x2);
        registry = new FLRegistry();
    }

    function testInitialState() public view {
        assertEq(registry.owner(), owner);
        assertEq(registry.currentRound(), 0);
    }

    function testSubmitUpdate() public {
        string memory ipfsHash = "QmTestHash123";
        
        vm.prank(client1);
        vm.expectEmit(true, false, false, true);
        emit ModelUpdate(client1, 0, ipfsHash);
        registry.submitUpdate(ipfsHash);
    }

    function testMultipleClientsSubmitUpdates() public {
        string memory hash1 = "QmHash1";
        string memory hash2 = "QmHash2";
        
        vm.prank(client1);
        registry.submitUpdate(hash1);
        
        vm.prank(client2);
        registry.submitUpdate(hash2);
        
        // Both should be in round 0
        assertEq(registry.currentRound(), 0);
    }

    function testEndRound() public {
        string memory globalHash = "QmGlobalHash";
        
        vm.expectEmit(true, false, false, true);
        emit RoundEnded(1, globalHash);
        registry.endRound(globalHash);
        
        assertEq(registry.currentRound(), 1);
    }

    function testEndRoundOnlyOwner() public {
        string memory globalHash = "QmGlobalHash";
        
        vm.prank(client1);
        vm.expectRevert("Only owner can end round");
        registry.endRound(globalHash);
    }

    function testMultipleRounds() public {
        // Round 0
        vm.prank(client1);
        registry.submitUpdate("QmHash0");
        
        // End round 0
        registry.endRound("QmGlobal0");
        assertEq(registry.currentRound(), 1);
        
        // Round 1
        vm.prank(client1);
        registry.submitUpdate("QmHash1");
        
        // End round 1
        registry.endRound("QmGlobal1");
        assertEq(registry.currentRound(), 2);
    }

    function testSubmitUpdateAfterRoundEnd() public {
        // Submit in round 0
        vm.prank(client1);
        registry.submitUpdate("QmHash0");
        
        // End round
        registry.endRound("QmGlobal0");
        
        // Submit in round 1
        vm.prank(client1);
        vm.expectEmit(true, false, false, true);
        emit ModelUpdate(client1, 1, "QmHash1");
        registry.submitUpdate("QmHash1");
    }

    function testFuzzSubmitUpdate(string memory ipfsHash) public {
        vm.assume(bytes(ipfsHash).length > 0);
        vm.prank(client1);
        registry.submitUpdate(ipfsHash);
        assertEq(registry.currentRound(), 0);
    }
}

