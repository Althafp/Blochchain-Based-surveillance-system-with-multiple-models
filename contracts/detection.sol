// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DetectionLog {
    struct Log {
        uint timestamp;
        string framePath;
        string detections;
    }

    Log[] public logs;

    event LogAdded(uint id, uint timestamp, string framePath, string detections);

    function addLog(string memory framePath, string memory detections) public {
        logs.push(Log(block.timestamp, framePath, detections));
        emit LogAdded(logs.length - 1, block.timestamp, framePath, detections);
    }

    function getLog(uint id) public view returns (uint, string memory, string memory) {
        require(id < logs.length, "Invalid log ID");
        Log memory log = logs[id];
        return (log.timestamp, log.framePath, log.detections);
    }

    function getLogCount() public view returns (uint) {
        return logs.length;
    }
}
