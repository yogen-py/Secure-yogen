from train import train_local
from grpc_client import send_model
from fedavg import fed_avg
import torch
import time
import yaml
import socket

def load_peers(config_file='host_config.yaml'):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    peers = config['peers']
    return [f"{peer['ip']}:{peer['port']}" for peer in peers]

def get_own_ip():
    # Method to get your local IP address (not 127.0.0.1)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def run_round(peer_addresses, own_address, local=True):
    local_weights = train_local()
    
    # Send to all other peers, skip self
    for addr in peer_addresses:
        if addr != own_address:
            send_model(local_weights, addr)
    return local_weights

def simulate_federation(peer_models):
    return fed_avg(peer_models)

if __name__ == "__main__":
    peer_addresses = load_peers()  # Load all peers
    own_ip = get_own_ip()
    own_port = '50051'  # Make sure this matches your server port
    own_address = f"{own_ip}:{own_port}"
    
    print(f"Own address detected as: {own_address}")
    print(f"All peers: {peer_addresses}")

    local_model = run_round(peer_addresses, own_address)
    
    # Simulate gathering remote models
    all_models = [local_model]
    # Normally, you'd collect remote weights from your gRPC server here
    
    global_model = simulate_federation(all_models)
    print("FedAvg complete. Global model ready.")

