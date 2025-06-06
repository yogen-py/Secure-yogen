import yaml
import logging
from grpc_client import test_health_check

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load peer addresses from host_config.yaml
def load_peers(config_file='host_config.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return [(peer['ip'], peer['port'], peer['name']) for peer in config['peers']]

if __name__ == "__main__":
    peers = load_peers()
    for ip, port, name in peers:
        address = f"{ip}:{port}"
        logger.info(f"Testing health check for peer: {name} ({address})")
        success = test_health_check(address=address, node_id=name)
        if success:
            logger.info(f"Peer {name} ({address}) is reachable.")
        else:
            logger.warning(f"Peer {name} ({address}) is NOT reachable.")
