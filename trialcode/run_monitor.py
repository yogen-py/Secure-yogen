from peer_monitor import PeerStatusMonitor
import socket
import yaml

def get_own_address():
    """Get the local machine's address from config"""
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    with open('host_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        for peer in config['peers']:
            if peer['ip'] == local_ip:
                return f"{peer['ip']}:{peer['port']}"
    return None

def main():
    own_address = get_own_address()
    if not own_address:
        print("Error: Could not find local machine in host_config.yaml")
        return

    monitor = PeerStatusMonitor(own_address, check_interval=30)
    
    try:
        print(f"Starting peer monitor for {own_address}")
        monitor.start_monitoring()
        
        # Keep the main thread alive
        while True:
            input()
    except KeyboardInterrupt:
        print("\nStopping peer monitor...")
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()
