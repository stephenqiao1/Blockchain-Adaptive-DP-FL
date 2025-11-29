// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FLRegistry {
    // Events to log data on the blockchain
    event ModelUpdate(address indexed client, uint round, string ipfsHash);
    event RoundEnded(uint round, string globalModelHash);

    address public owner;
    uint public currentRound;
    
    constructor() {
        owner = msg.sender;
        currentRound = 0;
    }

    // Client submits their local model update (IPFS hash)
    function submitUpdate(string memory _ipfsHash) public {
        emit ModelUpdate(msg.sender, currentRound, _ipfsHash);
    }

    // Aggregator (Owner) ends the round and publishes new global model
    function endRound(string memory _globalModelHash) public {
        require(msg.sender == owner, "Only owner can end round");
        currentRound++;
        emit RoundEnded(currentRound, _globalModelHash);
    }
}