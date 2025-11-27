// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FLRegistry {
    address public owner;
    uint public currentRound;
    uint public requiredClients;

    // Privacy Budget Config (Scaled by 100 to handle 2 decimals)
    uint public constant MAX_BUDGET = 2000;

    // Tracks how much budget each client has consumed
    mapping(address => uint) public usedBudget;

    // State: "0" = Registration, "1" = Training, "2" = Aggregation
    enum State { Registration, Training, Aggregation }
    State public currentState;

    struct ModelUpdate {
        address client;
        string ipfsHash;
        uint budgetConsumed;
    }

    // Mapping: Round Number => List of Updates
    mapping(uint => ModelUpdate[]) public roundUpdates;

    // Registered Clients
    mapping(address => bool) public registeredClients;
    uint public clientCount;

    event RoundStarted(uint round, string globalModelHash);
    event UpdateSubmitted(address client, uint round, uint budgetConsumed);
    event BudgetExceeded(address client, uint totalUsed);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this.");
        _;
    }

    constructor(uint _requiredClients) {
        owner = msg.sender;
        requiredClients = _requiredClients;
        currentState = State.Registration;
        currentRound = 0;
    }

    // 1. Clients register themselves
    function registerClient() external {
        require(!registeredClients[msg.sender], "Already registered.");
        require(currentState == State.Registration, "Registration closed.");
        registeredClients[msg.sender] = true;
        clientCount++;
    }

    // 2. Aggregator starts a new round
    function startRound(string memory globalModelHash) external onlyOwner {
        // Automatically close registration if we have enough clients
        if (currentState == State.Registration) {
            require(clientCount >= requiredClients, "Not enough clients.");
        }

        currentRound++;
        currentState = State.Training;
        emit RoundStarted(currentRound, globalModelHash);
    }

    // 3. Clients submit their training results
    function submitHash(string memory _ipfsHash, uint _roundBudgetSpent) external {
        require(currentState == State.Training, "Not in training phase.");
        require(registeredClients[msg.sender], "Not a registered client.");

        // 1. Update cumulative budget
        usedBudget[msg.sender] += _roundBudgetSpent;

        // 2. Check strict limit
        require(usedBudget[msg.sender] <= MAX_BUDGET, "Privacy budget exceeded. You are locked out.");

        roundUpdates[currentRound].push(ModelUpdate(msg.sender, _ipfsHash, _roundBudgetSpent));
        emit UpdateSubmitted(msg.sender, currentRound, _roundBudgetSpent);

        // If all clients have submitted, move to Aggregation
        if (roundUpdates[currentRound].length >= requiredClients) {
            currentState = State.Aggregation;
        }
    }

    // Helper to get all updates for the current round (for Aggregator)
    function getRoundUpdates(uint _round) external view returns (ModelUpdate[] memory) {
        return roundUpdates[_round];
    }
}