// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title FLRegistry
 * @notice Smart contract serving as the immutable "Virtual Aggregator" for Federated Learning
 * @dev Acts as a state machine and pointer registry. Does NOT perform matrix operations
 *      (aggregation happens off-chain due to Gas limits and lack of native floating-point in Solidity)
 */
contract FLRegistry {
    // Events
    event UpdateSubmitted(address indexed client, uint round, bytes32 ipfsHash, uint epsilonCost);
    event RoundEnded(uint round, bytes32 newGlobalHash);
    event ClientRegistered(address indexed client);
    event ReputationUpdated(address indexed client, uint newReputation);
    event RoundStarted(uint round, string globalHash);  // New event for round start
    
    // State Variables
    address public owner;
    uint public currentRound;
    bytes32 public globalModelHash;  // IPFS CID of current global model
    
    // Privacy budget limit (scaled by 1e18 for precision)
    uint public constant PRIVACY_BUDGET_LIMIT = 10 * 1e18;  // Max epsilon = 10.0
    
    // Client tracking
    struct ClientStatus {
        bool registered;
        uint reputation;           // Reliability score
        uint consumedEpsilon;       // Cumulative privacy budget consumed (scaled by 1e18)
        uint lastRoundParticipated; // Last round this client submitted an update
    }
    
    mapping(address => ClientStatus) public clients;
    
    // Track which clients submitted in current round
    mapping(uint => mapping(address => bool)) public roundSubmissions;
    
    constructor() {
        owner = msg.sender;
        currentRound = 0;
        globalModelHash = bytes32(0);  // Initial state: no model
    }
    
    /**
     * @notice Register a new client (allows nodes to stake their identity on-chain)
     * @dev Clients must register before submitting updates
     */
    function registerClient() public {
        require(!clients[msg.sender].registered, "Client already registered");
        clients[msg.sender] = ClientStatus({
            registered: true,
            reputation: 100,  // Start with base reputation
            consumedEpsilon: 0,
            lastRoundParticipated: 0
        });
        emit ClientRegistered(msg.sender);
    }
    
    /**
     * @notice Legacy function for backward compatibility
     * @dev Kept for compatibility with existing Python code
     */
    function register() public {
        registerClient();
    }
    
    /**
     * @notice Client submits their local model update (IPFS hash) with epsilon cost
     * @param _ipfsHash The IPFS Content Identifier (CID) of the client's model update
     * @param _epsilonCost The privacy budget consumed for this update (scaled by 1e18)
     * @dev Records the hash, increments consumed privacy budget, and emits UpdateSubmitted event
     */
    function registerUpdate(bytes32 _ipfsHash, uint _epsilonCost) public {
        ClientStatus storage client = clients[msg.sender];
        require(client.registered, "Client not registered");
        require(!roundSubmissions[currentRound][msg.sender], "Already submitted this round");
        require(
            client.consumedEpsilon + _epsilonCost <= PRIVACY_BUDGET_LIMIT,
            "Privacy budget exceeded"
        );
        
        // Update client status
        client.consumedEpsilon += _epsilonCost;
        client.lastRoundParticipated = currentRound;
        roundSubmissions[currentRound][msg.sender] = true;
        
        // Increment reputation for valid submission
        client.reputation += 1;
        
        emit UpdateSubmitted(msg.sender, currentRound, _ipfsHash, _epsilonCost);
        emit ReputationUpdated(msg.sender, client.reputation);
    }
    
    /**
     * @notice Accepts model updates and enforces privacy budget check
     * @param _ipfsHash The IPFS Content Identifier (CID) of the client's model update (as string)
     * @param _budgetSpent The privacy budget consumed for this update (scaled by 1e18)
     * @dev Enforces the privacy budget check with exact error message:
     *      require(usedBudget[msg.sender] + cost <= MAX_BUDGET, "Budget Exceeded")
     */
    function submitHash(string memory _ipfsHash, uint _budgetSpent) public {
        ClientStatus storage client = clients[msg.sender];
        require(client.registered, "Client not registered");
        require(!roundSubmissions[currentRound][msg.sender], "Already submitted this round");
        
        // Enforce privacy budget check with exact error message from specification
        require(
            client.consumedEpsilon + _budgetSpent <= PRIVACY_BUDGET_LIMIT,
            "Budget Exceeded"
        );
        
        // Convert string to bytes32
        bytes32 hashBytes32;
        bytes memory hashBytes = bytes(_ipfsHash);
        require(hashBytes.length <= 32, "Hash too long");
        assembly {
            hashBytes32 := mload(add(hashBytes, 32))
        }
        
        // Update client status
        client.consumedEpsilon += _budgetSpent;
        client.lastRoundParticipated = currentRound;
        roundSubmissions[currentRound][msg.sender] = true;
        
        // Increment reputation for valid submission
        client.reputation += 1;
        
        emit UpdateSubmitted(msg.sender, currentRound, hashBytes32, _budgetSpent);
        emit ReputationUpdated(msg.sender, client.reputation);
    }
    
    /**
     * @notice Start a new round and emit event signaling availability of new global model CID
     * @param _globalHash The IPFS CID of the new global model (as string)
     * @dev Emits an event signaling the availability of a new global model CID.
     *      This allows clients to know when to start training on the new model.
     */
    function startRound(string memory _globalHash) public {
        require(msg.sender == owner, "Only aggregator can start round");
        require(bytes(_globalHash).length > 0, "Invalid hash");
        
        // Increment round and update global model hash
        currentRound++;
        
        // Convert string to bytes32 for storage
        bytes32 hashBytes32;
        bytes memory hashBytes = bytes(_globalHash);
        require(hashBytes.length <= 32, "Hash too long");
        assembly {
            hashBytes32 := mload(add(hashBytes, 32))
        }
        globalModelHash = hashBytes32;
        
        emit RoundStarted(currentRound, _globalHash);
    }
    
    /**
     * @notice Aggregator verifies and publishes new global model hash
     * @param _newGlobalHash The IPFS CID of the aggregated global model
     * @dev In production, this would be called by a consensus committee.
     *      For prototype, a rotating aggregator performs off-chain aggregation,
     *      uploads result to IPFS, and submits the new hash.
     */
    function verifyAndAggregate(bytes32 _newGlobalHash) public {
        require(msg.sender == owner, "Only aggregator can verify and aggregate");
        require(_newGlobalHash != bytes32(0), "Invalid hash");
        
        // Update global model hash and increment round
        globalModelHash = _newGlobalHash;
        currentRound++;
        
        emit RoundEnded(currentRound, _newGlobalHash);
    }
    
    /**
     * @notice Penalize a client for malicious behavior
     * @param _client Address of the client to penalize
     * @param _penalty Amount to reduce reputation by
     * @dev Only owner (aggregator) can penalize clients
     */
    function penalizeClient(address _client, uint _penalty) public {
        require(msg.sender == owner, "Only aggregator can penalize");
        require(clients[_client].registered, "Client not registered");
        
        if (clients[_client].reputation > _penalty) {
            clients[_client].reputation -= _penalty;
        } else {
            clients[_client].reputation = 0;
        }
        
        emit ReputationUpdated(_client, clients[_client].reputation);
    }
    
    /**
     * @notice Get client status
     * @param _client Address of the client
     * @return registered Whether client is registered
     * @return reputation Current reputation score
     * @return consumedEpsilon Total privacy budget consumed
     * @return lastRoundParticipated Last round client submitted update
     */
    function getClientStatus(address _client) public view returns (
        bool registered,
        uint reputation,
        uint consumedEpsilon,
        uint lastRoundParticipated
    ) {
        ClientStatus memory client = clients[_client];
        return (
            client.registered,
            client.reputation,
            client.consumedEpsilon,
            client.lastRoundParticipated
        );
    }
    
    /**
     * @notice Legacy function for backward compatibility
     * @dev Kept for compatibility with existing Python code
     */
    function submitUpdate(string memory _ipfsHash) public {
        // Convert string to bytes32 (truncate if needed)
        bytes32 hash;
        bytes memory hashBytes = bytes(_ipfsHash);
        require(hashBytes.length <= 32, "Hash too long");
        assembly {
            hash := mload(add(hashBytes, 32))
        }
        
        // Call registerUpdate with default epsilon cost (0.1 scaled by 1e18)
        registerUpdate(hash, 1e17);  // 0.1 * 1e18
    }
    
    /**
     * @notice Legacy function for backward compatibility
     * @dev Kept for compatibility with existing Python code
     */
    function endRound(string memory _globalModelHash) public {
        // Convert string to bytes32
        bytes32 hash;
        bytes memory hashBytes = bytes(_globalModelHash);
        require(hashBytes.length <= 32, "Hash too long");
        assembly {
            hash := mload(add(hashBytes, 32))
        }
        
        verifyAndAggregate(hash);
    }
}
