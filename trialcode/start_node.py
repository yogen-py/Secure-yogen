import subprocess
import threading
import time
import sys
from peer_monitor import PeerStatusMonitor
import netifaces
import yaml
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_own_address():
    """Detect and return the peer address from config that matches any of the local IPs."""
    local_ips = []
    for iface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(iface)
        if netifaces.AF_INET in addrs:
            for addr in addrs[netifaces.AF_INET]:
                local_ips.append(addr['addr'])
    
    logger.info(f"Detected local IPs: {local_ips}")

    with open('host_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        for peer in config['peers']:
            if peer['ip'] in local_ips:
                logger.info(f"Matched local IP {peer['ip']} to peer configuration")
                return f"{peer['ip']}:{peer['port']}", peer['name']
    return None, None

def start_server(port):
    """Start the gRPC server"""
    subprocess.Popen([sys.executable, 'grpc_server.py'])
    logger.info(f"Started gRPC server on port {port}")

def start_monitor(address):
    """Start the peer monitor"""
    # Using 5-second interval for more frequent updates
    monitor = PeerStatusMonitor(address, check_interval=5)
    monitor.start_monitoring()
    return monitor

def start_main_process():
    """Start the main FL process"""
    subprocess.Popen([sys.executable, 'main.py'])
    logger.info("Started main FL process")

def main():
    # Get own address
    address, peer_name = get_own_address()
    if not address:
        logger.error("Could not find local machine in host_config.yaml")
        return

    logger.info(f"Starting node {peer_name} at {address}")
    
    try:
        # Start gRPC server
        port = int(address.split(':')[1])
        start_server(port)
        time.sleep(2)  # Wait for server to start
        
        # Start peer monitor with 5-second interval
        monitor = start_monitor(address)
        time.sleep(2)  # Reduced wait time since we're checking more frequently
        
        # Start main FL process
        start_main_process()
        
        logger.info(f"Node {peer_name} is fully operational")
        
        # Keep the script running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down node...")
            monitor.stop_monitoring()
            
    except Exception as e:
        logger.error(f"Error starting node: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
