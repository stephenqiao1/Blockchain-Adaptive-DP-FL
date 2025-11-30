// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "forge-std/Test.sol";
import "../contracts/FLRegistry.sol";

contract FLRegistryTest is Test {
    FLRegistry public registry;
    address public owner;
    address public client1;
    address public client2;
    address public client3;

    event UpdateSubmitted(address indexed client, uint round, bytes32 ipfsHash, uint epsilonCost);
    event RoundEnded(uint round, bytes32 newGlobalHash);
    event RoundStarted(uint round, string globalHash);
    event ClientRegistered(address indexed client);
    event ReputationUpdated(address indexed client, uint newReputation);

    function setUp() public {
        owner = address(this);
        client1 = address(0x1);
        client2 = address(0x2);
        client3 = address(0x3);
        registry = new FLRegistry();
    }

    function testInitialState() public view {
        assertEq(registry.owner(), owner);
        assertEq(registry.currentRound(), 0);
        assertEq(registry.globalModelHash(), bytes32(0));
        assertEq(registry.PRIVACY_BUDGET_LIMIT(), 10 * 1e18);
    }

    function testRegisterClient() public {
        vm.prank(client1);
        registry.registerClient();
        
        (bool registered, uint reputation, , ) = registry.getClientStatus(client1);
        assertTrue(registered, "Client should be registered");
        assertEq(reputation, 100, "Client should start with 100 reputation");
    }
    
    function testRegisterClientAlias() public {
        // Test that register() still works (backward compatibility)
        vm.prank(client1);
        registry.register();
        
        (bool registered, , , ) = registry.getClientStatus(client1);
        assertTrue(registered, "Client should be registered via register()");
    }

    function testSubmitUpdateRequiresRegistration() public {
        bytes32 hash = keccak256("test");
        uint epsilon = 1e17; // 0.1
        
        vm.prank(client1);
        vm.expectRevert("Client not registered");
        registry.registerUpdate(hash, epsilon);
    }

    function testRegisterUpdate() public {
        // Register client first
        vm.prank(client1);
        registry.register();
        
        bytes32 hash = keccak256("test");
        uint epsilon = 1e17; // 0.1
        
        vm.prank(client1);
        vm.expectEmit(true, false, false, true);
        emit UpdateSubmitted(client1, 0, hash, epsilon);
        registry.registerUpdate(hash, epsilon);
        
        (, , uint consumedEpsilon, uint lastRound) = registry.getClientStatus(client1);
        assertEq(consumedEpsilon, epsilon, "Epsilon should be consumed");
        assertEq(lastRound, 0, "Last round should be 0");
    }

    function testPrivacyBudgetLimit() public {
        vm.prank(client1);
        registry.register();
        
        bytes32 hash1 = keccak256("test1");
        bytes32 hash2 = keccak256("test2");
        uint maxEpsilon = 10 * 1e18; // Max allowed
        
        vm.prank(client1);
        registry.registerUpdate(hash1, maxEpsilon);
        
        // End round so client can submit again
        registry.verifyAndAggregate(keccak256("global"));
        
        // Try to exceed limit in next round
        vm.prank(client1);
        vm.expectRevert("Privacy budget exceeded");
        registry.registerUpdate(hash2, 1);
    }

    function testReputationIncreases() public {
        vm.prank(client1);
        registry.register();
        
        bytes32 hash = keccak256("test");
        uint epsilon = 1e17;
        
        uint reputationBefore;
        (, reputationBefore, , ) = registry.getClientStatus(client1);
        
        vm.prank(client1);
        registry.registerUpdate(hash, epsilon);
        
        uint reputationAfter;
        (, reputationAfter, , ) = registry.getClientStatus(client1);
        assertGt(reputationAfter, reputationBefore, "Reputation should increase");
    }

    function testVerifyAndAggregate() public {
        bytes32 newHash = keccak256("global_model");
        
        vm.expectEmit(true, false, false, true);
        emit RoundEnded(1, newHash);
        registry.verifyAndAggregate(newHash);
        
        assertEq(registry.globalModelHash(), newHash, "Global hash should be updated");
        assertEq(registry.currentRound(), 1, "Round should increment");
    }

    function testVerifyAndAggregateOnlyOwner() public {
        bytes32 newHash = keccak256("global_model");
        
        vm.prank(client1);
        vm.expectRevert("Only aggregator can verify and aggregate");
        registry.verifyAndAggregate(newHash);
    }

    function testPenalizeClient() public {
        vm.prank(client1);
        registry.register();
        
        uint reputationBefore;
        (, reputationBefore, , ) = registry.getClientStatus(client1);
        
        registry.penalizeClient(client1, 10);
        
        uint reputationAfter;
        (, reputationAfter, , ) = registry.getClientStatus(client1);
        assertEq(reputationAfter, reputationBefore - 10, "Reputation should decrease");
    }

    function testMultipleRounds() public {
        // Register clients
        vm.prank(client1);
        registry.register();
        vm.prank(client2);
        registry.register();
        
        bytes32 hash1 = keccak256("round1");
        bytes32 hash2 = keccak256("round2");
        uint epsilon = 1e17;
        
        // Round 0
        vm.prank(client1);
        registry.registerUpdate(hash1, epsilon);
        vm.prank(client2);
        registry.registerUpdate(hash1, epsilon);
        
        bytes32 global1 = keccak256("global1");
        registry.verifyAndAggregate(global1);
        
        // Round 1
        vm.prank(client1);
        registry.registerUpdate(hash2, epsilon);
        
        bytes32 global2 = keccak256("global2");
        registry.verifyAndAggregate(global2);
        
        assertEq(registry.currentRound(), 2, "Should be in round 2");
        assertEq(registry.globalModelHash(), global2, "Global hash should be updated");
    }

    function testPreventDuplicateSubmissions() public {
        vm.prank(client1);
        registry.register();
        
        bytes32 hash = keccak256("test");
        uint epsilon = 1e17;
        
        vm.prank(client1);
        registry.registerUpdate(hash, epsilon);
        
        // Try to submit again in same round
        vm.prank(client1);
        vm.expectRevert("Already submitted this round");
        registry.registerUpdate(hash, epsilon);
    }

    function testLegacyFunctions() public {
        // Test backward compatibility
        string memory hashStr = "QmTestHash123";
        
        vm.prank(client1);
        registry.register();
        
        // Legacy submitUpdate should work
        vm.prank(client1);
        registry.submitUpdate(hashStr);
        
        // Legacy endRound should work
        string memory globalStr = "QmGlobalHash";
        registry.endRound(globalStr);
        
        assertEq(registry.currentRound(), 1, "Round should increment");
    }

    function testFuzzRegisterUpdate(bytes32 hash, uint epsilon) public {
        // Bound epsilon to reasonable range
        epsilon = bound(epsilon, 1, 9 * 1e18);
        
        vm.prank(client1);
        registry.register();
        
        vm.prank(client1);
        registry.registerUpdate(hash, epsilon);
        
        (, , uint consumed, ) = registry.getClientStatus(client1);
        assertEq(consumed, epsilon, "Epsilon should match");
    }
    
    function testStartRound() public {
        string memory globalHash = "QmNewGlobalModelHash";
        uint roundBefore = registry.currentRound();
        
        vm.expectEmit(true, false, false, true);
        emit RoundStarted(roundBefore + 1, globalHash);
        registry.startRound(globalHash);
        
        assertEq(registry.currentRound(), roundBefore + 1, "Round should increment");
        assertTrue(registry.globalModelHash() != bytes32(0), "Global hash should be set");
    }
    
    function testStartRoundOnlyOwner() public {
        string memory globalHash = "QmTestHash";
        
        vm.prank(client1);
        vm.expectRevert("Only aggregator can start round");
        registry.startRound(globalHash);
    }
    
    function testSubmitHash() public {
        vm.prank(client1);
        registry.registerClient();
        
        string memory ipfsHash = "QmClientUpdateHash";
        uint budgetSpent = 1e17; // 0.1 epsilon
        
        vm.prank(client1);
        registry.submitHash(ipfsHash, budgetSpent);
        
        (, , uint consumed, ) = registry.getClientStatus(client1);
        assertEq(consumed, budgetSpent, "Budget should be consumed");
    }
    
    function testSubmitHashBudgetExceeded() public {
        vm.prank(client1);
        registry.registerClient();
        
        string memory ipfsHash1 = "QmTestHash1";
        string memory ipfsHash2 = "QmTestHash2";
        uint maxBudget = 10 * 1e18; // Max allowed
        
        // Use all budget in round 0
        vm.prank(client1);
        registry.submitHash(ipfsHash1, maxBudget);
        
        // Start new round so client can submit again
        registry.startRound("QmNewGlobal");
        
        // Try to exceed budget in new round
        vm.prank(client1);
        vm.expectRevert("Budget Exceeded");
        registry.submitHash(ipfsHash2, 1);
    }
    
    function testSubmitHashRequiresRegistration() public {
        string memory ipfsHash = "QmTestHash";
        uint budgetSpent = 1e17;
        
        vm.prank(client1);
        vm.expectRevert("Client not registered");
        registry.submitHash(ipfsHash, budgetSpent);
    }
    
    function testCompleteWorkflow() public {
        // 1. Register clients
        vm.prank(client1);
        registry.registerClient();
        vm.prank(client2);
        registry.registerClient();
        
        // 2. Start round 1
        string memory globalHash1 = "QmGlobalModelRound1";
        registry.startRound(globalHash1);
        assertEq(registry.currentRound(), 1, "Should be in round 1");
        
        // 3. Clients submit updates
        vm.prank(client1);
        registry.submitHash("QmClient1Update", 1e17);
        vm.prank(client2);
        registry.submitHash("QmClient2Update", 1e17);
        
        // 4. Start round 2
        string memory globalHash2 = "QmGlobalModelRound2";
        registry.startRound(globalHash2);
        assertEq(registry.currentRound(), 2, "Should be in round 2");
        
        // 5. Clients can submit in new round
        vm.prank(client1);
        registry.submitHash("QmClient1Update2", 1e17);
        
        // Verify budget accumulation
        (, , uint consumed, ) = registry.getClientStatus(client1);
        assertEq(consumed, 2e17, "Budget should accumulate across rounds");
    }
}
