import logging
from train import train_local
from grpc_client import send_model
from fedavg import fed_avg
import torch
import time
import yaml
import socket
from grpc_server import received_models

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
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
        logger.warning("Could not determine IP address, using localhost")
    finally:
        s.close()
    return IP

def wait_for_peer_models(required_peers, timeout=300):
    """Wait for models from all peers"""
    start_time = time.time()
    while len(received_models) < required_peers:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for peer models after {timeout}s")
        time.sleep(1)
    logger.info(f"Received {len(received_models)} peer models")

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
        logger.info("Starting local training round")
        local_weights = train_local()
        successful_peers = []
        failed_peers = []
        
        # Send to all other peers, skip self
        for addr in peer_addresses:
            if addr == own_address:
                continue
                
            success = False
            retries = 0
            
            while not success and retries < max_retries:
                if retries > 0:
                    logger.info(f"Retrying send to {addr} (attempt {retries + 1}/{max_retries})")
                    time.sleep(retry_delay)
                
                success = send_model(
                    local_weights, 
                    addr,
                    timeout=30,  # 30 second timeout
                    use_ssl=False  # Enable if SSL certificates are set up
                )
                retries += 1
            
            if success:
                successful_peers.append(addr)
            else:
                failed_peers.append(addr)
        
        if failed_peers:
            logger.warning(f"Failed to send model to peers: {failed_peers}")
        
        return local_weights, len(successful_peers)
        
    except Exception as e:
        logger.error(f"Error in training round: {str(e)}")
        raise

def simulate_federation(peer_models):
    try:
        return fed_avg(peer_models)
    except Exception as e:
        logger.error(f"Error in federation: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        peer_addresses = load_peers()
        own_ip = get_own_ip()
        own_port = '50051'
        own_address = f"{own_ip}:{own_port}"
        
        logger.info(f"Own address: {own_address}")
        logger.info(f"All peers: {peer_addresses}")

        # Clear any previous models
        received_models.clear()
        
        # Run local training and send to peers
        local_model, successful_sends = run_round(peer_addresses, own_address)
        
        # Calculate required peers (excluding self)
        total_peers = len(peer_addresses) - 1
        min_required_peers = max(1, total_peers // 2)  # At least 50% of peers
        
        if successful_sends < min_required_peers:
            logger.error(f"Failed to reach minimum required peers ({successful_sends}/{min_required_peers})")
            raise RuntimeError("Insufficient peer connectivity")
            
        # Wait for other peers
        if total_peers > 0:
            logger.info(f"Waiting for peer models (minimum {min_required_peers} required)")
            try:
                wait_for_peer_models(min_required_peers)
            except TimeoutError as e:
                logger.error(f"Timeout waiting for peer models: {str(e)}")
                raise
        
        # Combine local and received models
        all_models = [local_model] + received_models
        
        # Perform federation
        global_model = simulate_federation(all_models)
        logger.info(f"FedAvg complete with {len(all_models)} models")
        
    except Exception as e:
        logger.error(f"Fatal error in main process: {str(e)}")
        raise

