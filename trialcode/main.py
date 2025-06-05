import logging
from train import train_local
from grpc_client import send_model
from fedavg import fed_avg
import torch
import time
import yaml
import socket
from grpc_server import received_models, serve, set_current_round
from tqdm import tqdm
import argparse
import threading
import hashlib
from datetime import datetime
from train import evaluate
from start_fl_node import test_connections

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SAVE_MODEL_DEBUG = True  # Toggle to save sent/received models for inspection
NODE_ID = None  # Set this to a unique identifier for each node (e.g., from host_config.yaml)

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
        stats = summarize_weights_full(state_dict)
        tqdm.write(f"[DIAG] Model {i+1} stats: " + ", ".join([f"{k}: mean={v['mean']:.4f}, std={v['std']:.4f}" for k,v in stats.items() if 'weight' in k]))

def summarize_weights_full(state_dict):
    stats = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            stats[k] = {
                'shape': tuple(v.shape),
                'mean': float(v.mean()),
                'std': float(v.std()),
                'min': float(v.min()),
                'max': float(v.max())
            }
    return stats

def state_dict_checksum(state_dict):
    m = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        if torch.is_tensor(v):
            m.update(v.cpu().numpy().tobytes())
    return m.hexdigest()[:12]

def run_round(peer_addresses, own_address, max_retries=10, retry_delay=10, global_model=None, round_num=1):
    try:
        tqdm.write("[ROUND] Starting local training round...")
        # Detailed per-epoch logging
        def train_with_logging(*args, **kwargs):
            set_seed = kwargs.get('set_seed', 42)
            epochs = kwargs.get('epochs', 3)
            batch_size = kwargs.get('batch_size', 64)
            save_dir = kwargs.get('save_dir', "models")
            machine_id = kwargs.get('machine_id', 0)
            total_machines = kwargs.get('total_machines', 4)
            from data import get_data_loaders
            from model import SimpleBinaryClassifier, get_loss, get_optimizer
            from train import evaluate
            import os
            os.makedirs(save_dir, exist_ok=True)
            train_loader, test_loader = get_data_loaders(machine_id=machine_id, total_machines=total_machines, batch_size=batch_size)
            input_dim = next(iter(train_loader))[0].shape[1]
            model = SimpleBinaryClassifier(input_dim)
            # Load global model weights if provided
            if global_model is not None:
                model.load_state_dict(global_model)
            criterion = get_loss()
            optimizer = get_optimizer(model)
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    outputs = model(xb)
                    loss = criterion(outputs, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * xb.size(0)
                avg_loss = epoch_loss / len(train_loader.dataset)
                acc, prec, rec, f1 = evaluate(model, test_loader)
                tqdm.write(f"[TRAIN][Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
            return model.state_dict()
        # Use the above for detailed per-epoch logging
        local_weights = train_with_logging()
        # Log summary stats of local weights
        stats = summarize_weights_full(local_weights)
        tqdm.write("[ROUND] Local model weights summary:")
        for k, v in stats.items():
            tqdm.write(f"  {k}: mean={v['mean']:.4f}, std={v['std']:.4f}, min={v['min']:.4f}, max={v['max']:.4f}")
        tqdm.write(f"  Checksum: {state_dict_checksum(local_weights)}")
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
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                model_size = sum(v.numel() for v in local_weights.values() if torch.is_tensor(v)) * 4 / 1024
                checksum = state_dict_checksum(local_weights)
                tqdm.write(f"[SEND][{now}] Sending model to {addr} | Size: {model_size:.2f} KB | Checksum: {checksum} | Node: {NODE_ID}")
                if SAVE_MODEL_DEBUG:
                    fname = f"sent_model_{NODE_ID}_to_{addr.replace(':', '_')}_round{round_num}.pt"
                    torch.save(local_weights, fname)
                    tqdm.write(f"[SEND][DEBUG] Saved sent model to {fname}")
                success = send_model(
                    local_weights, 
                    addr,
                    round_num=round_num,
                    timeout=30,  # 30 second timeout
                    use_ssl=False  # Enable if SSL certificates are set up
                )
                retries += 1
            if success:
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                tqdm.write(f"[SEND][{now}] Model sent to {addr} successfully.")
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

def summarize_received_models():
    for i, state_dict in enumerate(received_models):
        stats = summarize_weights_full(state_dict)
        checksum = state_dict_checksum(state_dict)
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        tqdm.write(f"[RECEIVE][{now}] Model {i+1} weights summary | Checksum: {checksum} | Node: {NODE_ID}")
        for k, v in stats.items():
            tqdm.write(f"  {k}: mean={v['mean']:.4f}, std={v['std']:.4f}, min={v['min']:.4f}, max={v['max']:.4f}")
        if SAVE_MODEL_DEBUG:
            fname = f"received_model_{NODE_ID}_idx{i+1}_round{current_round}.pt"
            torch.save(state_dict, fname)
            tqdm.write(f"[RECEIVE][DEBUG] Saved received model to {fname}")

def simulate_federation(peer_models):
    try:
        tqdm.write(f"[FEDAVG] Aggregating {len(peer_models)} models...")
        global_model = fed_avg(peer_models)
        stats = summarize_weights_full(global_model)
        tqdm.write(f"[FEDAVG] Global model weights summary:")
        for k, v in stats.items():
            tqdm.write(f"  {k}: mean={v['mean']:.4f}, std={v['std']:.4f}, min={v['min']:.4f}, max={v['max']:.4f}")
        checksum = state_dict_checksum(global_model)
        tqdm.write(f"[FEDAVG] Global model checksum: {checksum}")
        torch.save(global_model, f'global_model_round_{round_num}.pt')
        tqdm.write(f"[INFO] Saved global model for round {round_num} to disk.")
        return global_model
    except Exception as e:
        tqdm.write(f"[ERROR] Error in federation: {str(e)}")
        raise

def start_grpc_server_in_thread(port=50051):
    server_thread = threading.Thread(target=serve, args=(port,), daemon=True)
    server_thread.start()
    return server_thread

def evaluate_global_model(global_model_state_dict, machine_id=0, total_machines=4, batch_size=64):
    from data import get_data_loaders
    from model import SimpleBinaryClassifier
    _, test_loader = get_data_loaders(machine_id=machine_id, total_machines=total_machines, batch_size=batch_size)
    input_dim = next(iter(test_loader))[0].shape[1]
    model = SimpleBinaryClassifier(input_dim)
    model.load_state_dict(global_model_state_dict)
    acc, prec, rec, f1 = evaluate(model, test_loader)
    tqdm.write(f"[EVAL][Global Model] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

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
        global_model = None
        for round_num in range(1, num_rounds + 1):
            set_current_round(round_num)
            tqdm.write(f"\n=== Federated Learning Round {round_num} ===")
            received_models.clear()
            # Evaluate the latest global model before starting the next round (after round 1)
            if round_num > 1 and global_model is not None:
                tqdm.write(f"[EVAL] Evaluating global model before round {round_num}...")
                evaluate_global_model(global_model)
            # Run local training and send to peers, passing global_model and round_num
            local_model, successful_sends = run_round(peer_addresses, own_address, global_model=global_model, round_num=round_num)
            # Calculate required peers (excluding self)
            total_peers = len(peer_addresses) - 1
            min_required_peers = max(1, total_peers // 2)  # At least 50% of peers
            if successful_sends < min_required_peers:
                tqdm.write(f"[ERROR] Failed to reach minimum required peers ({successful_sends}/{min_required_peers})")
                tqdm.write(f"[DEBUG] Peer addresses: {peer_addresses}")
                tqdm.write(f"[DEBUG] Own address: {own_address}")
                tqdm.write(f"[DEBUG] Successful sends: {successful_sends}")
                tqdm.write(f"[DEBUG] This node will sleep and retry next round.")
                # Optionally, re-test connections here
                test_connections()  # Uncomment if you want to run your connection test script
                time.sleep(60)  # Wait 60 seconds before next round
                continue  # Skip this round, don't raise
            # Wait for other peers
            if total_peers > 0:
                tqdm.write(f"[INFO] Waiting for peer models (minimum {min_required_peers} required)")
                try:
                    prev_count = 0
                    while len(received_models) < min_required_peers:
                        if len(received_models) > prev_count:
                            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            tqdm.write(f"[RECEIVE][{now}] New model received! Total received: {len(received_models)}")
                            summarize_received_models()
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
            stats = summarize_weights_full(global_model)
            tqdm.write(f"[UPDATE] Model updated after FedAvg: " + ", ".join([f"{k}: mean={v['mean']:.4f}, std={v['std']:.4f}, min={v['min']:.4f}, max={v['max']:.4f}" for k,v in stats.items() if 'weight' in k]))
            tqdm.write(f"[UPDATE] Global model checksum: {state_dict_checksum(global_model)}")
            tqdm.write(f"=== End of Round {round_num} ===\n")
            time.sleep(5)  # Optional: wait before next round
    except Exception as e:
        tqdm.write(f"[FATAL] Exception in main federated loop: {str(e)}")

