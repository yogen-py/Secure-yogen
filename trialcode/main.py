import logging
from train import train_local
from grpc_client import send_model
from fedavg import fed_avg
import torch
import time
import yaml
import socket
from grpc_server import received_models, serve
from tqdm import tqdm
import argparse
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_peers(config_file='host_config.yaml'):
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        peers = config['peers']
        return [f"{peer['ip']}:{peer['port']}" for peer in peers]
    except Exception as e:
        logger.error(f"Error loading peer configuration: {str(e)}")
        raise

def get_own_ip():
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

def wait_for_peer_models(required_peers, timeout=300):
    """Wait for models from all peers, with option to wake up/retry if timeout occurs."""
    start_time = time.time()
    with tqdm(total=required_peers, desc="Waiting for peer models", ncols=80) as pbar:
        last_count = 0
        while len(received_models) < required_peers:
            if time.time() - start_time > timeout:
                tqdm.write("[TIMEOUT] Timeout waiting for peer models. Press Enter to retry waiting, or type 'exit' to abort.")
                user_input = input()
                if user_input.strip().lower() == 'exit':
                    raise TimeoutError(f"Timeout waiting for peer models after {timeout}s (user aborted)")
                # Reset timer and continue waiting
                start_time = time.time()
                continue
            current_count = len(received_models)
            if current_count > last_count:
                pbar.update(current_count - last_count)
                last_count = current_count
            time.sleep(1)
        pbar.update(required_peers - last_count)
    tqdm.write(f"[DIAG] Received {len(received_models)} peer models.")
    # Print summary stats for each received model
    for i, state_dict in enumerate(received_models):
        stats = summarize_weights(state_dict)
        tqdm.write(f"[DIAG] Model {i+1} stats: " + ", ".join([f"{k}: mean={v['mean']:.4f}, std={v['std']:.4f}" for k,v in stats.items() if 'weight' in k]))

def summarize_weights(state_dict):
    stats = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            stats[k] = {
                'shape': tuple(v.shape),
                'mean': float(v.mean()),
                'std': float(v.std())
            }
    return stats

def run_round(peer_addresses, own_address, max_retries=3, retry_delay=5):
    """
    Run a training round and send model to peers
    Args:
        peer_addresses: List of peer addresses
        own_address: This node's address
        max_retries: Maximum number of retry attempts per peer
        retry_delay: Delay between retries in seconds
    Returns:
        tuple: (local_weights, successful_peers)
    """
    try:
        tqdm.write("[ROUND] Starting local training round...")
        local_weights = train_local()
        # Log summary stats of local weights
        stats = summarize_weights(local_weights)
        tqdm.write(f"[ROUND] Local model stats: " + ", ".join([f"{k}: mean={v['mean']:.4f}, std={v['std']:.4f}" for k,v in stats.items() if 'weight' in k]))
        successful_peers = []
        failed_peers = []
        
        # Send to all other peers, skip self
        for addr in peer_addresses:
            if addr == own_address:
                tqdm.write(f"[SEND] Skipping send to self ({addr})")
                continue
            success = False
            retries = 0
            while not success and retries < max_retries:
                if retries > 0:
                    tqdm.write(f"[SEND] Retrying send to {addr} (attempt {retries + 1}/{max_retries})")
                    time.sleep(retry_delay)
                tqdm.write(f"[SEND] Sending model to {addr}...")
                success = send_model(
                    local_weights, 
                    addr,
                    timeout=30,  # 30 second timeout
                    use_ssl=False  # Enable if SSL certificates are set up
                )
                retries += 1
            if success:
                tqdm.write(f"[SEND] Model sent to {addr} successfully.")
                successful_peers.append(addr)
            else:
                tqdm.write(f"[SEND] Failed to send model to {addr} after {max_retries} attempts.")
                failed_peers.append(addr)
        if failed_peers:
            tqdm.write(f"[WARN] Failed to send model to peers: {failed_peers}")
        return local_weights, len(successful_peers)
    except Exception as e:
        tqdm.write(f"[ERROR] Error in training round: {str(e)}")
        raise

def simulate_federation(peer_models):
    try:
        tqdm.write(f"[FEDAVG] Aggregating {len(peer_models)} models...")
        global_model = fed_avg(peer_models)
        stats = summarize_weights(global_model)
        tqdm.write(f"[FEDAVG] Global model stats: " + ", ".join([f"{k}: mean={v['mean']:.4f}, std={v['std']:.4f}" for k,v in stats.items() if 'weight' in k]))
        return global_model
    except Exception as e:
        tqdm.write(f"[ERROR] Error in federation: {str(e)}")
        raise

def start_grpc_server_in_thread(port=50051):
    server_thread = threading.Thread(target=serve, args=(port,), daemon=True)
    server_thread.start()
    return server_thread

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--rounds", type=int, default=10, help="Number of federated learning rounds")
        args = parser.parse_args()
        num_rounds = args.rounds

        # Start gRPC server in a background thread (so received_models is shared)
        start_grpc_server_in_thread(port=50051)
        tqdm.write("[INFO] gRPC server started in background thread.")
        time.sleep(2)  # Give the server a moment to start

        peer_addresses = load_peers()
        own_ip = get_own_ip()
        own_port = '50051'
        own_address = f"{own_ip}:{own_port}"
        tqdm.write(f"[INFO] Own address: {own_address}")
        tqdm.write(f"[INFO] All peers: {peer_addresses}")
        for round_num in range(1, num_rounds + 1):
            tqdm.write(f"\n=== Federated Learning Round {round_num} ===")
            received_models.clear()
            # Run local training and send to peers
            local_model, successful_sends = run_round(peer_addresses, own_address)
            # Calculate required peers (excluding self)
            total_peers = len(peer_addresses) - 1
            min_required_peers = max(1, total_peers // 2)  # At least 50% of peers
            if successful_sends < min_required_peers:
                tqdm.write(f"[ERROR] Failed to reach minimum required peers ({successful_sends}/{min_required_peers})")
                raise RuntimeError("Insufficient peer connectivity")
            # Wait for other peers
            if total_peers > 0:
                tqdm.write(f"[INFO] Waiting for peer models (minimum {min_required_peers} required)")
                try:
                    prev_count = 0
                    while len(received_models) < min_required_peers:
                        if len(received_models) > prev_count:
                            tqdm.write(f"[RECEIVE] New model received! Total received: {len(received_models)}")
                            prev_count = len(received_models)
                        time.sleep(1)
                except TimeoutError as e:
                    tqdm.write(f"[ERROR] Timeout waiting for peer models: {str(e)}")
                    raise
            # Combine local and received models
            all_models = [local_model] + received_models
            # Perform federation
            global_model = simulate_federation(all_models)
            tqdm.write(f"[ROUND] FedAvg complete with {len(all_models)} models")
            # Show updated model stats
            stats = summarize_weights(global_model)
            tqdm.write(f"[UPDATE] Model updated after FedAvg: " + ", ".join([f"{k}: mean={v['mean']:.4f}, std={v['std']:.4f}" for k,v in stats.items() if 'weight' in k]))
            tqdm.write(f"=== End of Round {round_num} ===\n")
            time.sleep(5)  # Optional: wait before next round
    except Exception as e:
        tqdm.write(f"[FATAL] Exception in main federated loop: {str(e)}")

