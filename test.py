from web3 import Web3
import json

# Ethereum Node Connection (Local or Hosted)
eth_url = "http://127.0.0.1:8545"  # Replace with your RPC URL (e.g., Infura, Hardhat, Anvil)
web3 = Web3(Web3.HTTPProvider(eth_url))

if not web3.is_connected():
    print("Failed to connect to Ethereum network. Check your node.")
    exit()
else:
    print("Connected to Ethereum network.")

# Load Smart Contract ABI and Address
with open("detetion.abi", "r") as abi_file:
    contract_abi = json.load(abi_file)
with open("detection.address", "r") as address_file:
    contract_address = address_file.read().strip()

# Connect to the Smart Contract
contract = web3.eth.contract(address=contract_address, abi=contract_abi)

# Retrieve and Display Logs
try:
    log_count = contract.functions.getLogCount().call()  # Total logs stored
    print(f"Total logs stored: {log_count}")

    for i in range(log_count):
        log = contract.functions.getLog(i).call()  # Retrieve each log
        print(f"Log {i}:")
        print(f"  Timestamp: {log[0]}")  # Timestamp in UNIX format
        print(f"  Frame Path: {log[1]}")  # Path to the frame image
        print(f"  Detections: {log[2]}")  # Detection data in JSON format
except Exception as e:
    print(f"Error retrieving logs: {e}")


