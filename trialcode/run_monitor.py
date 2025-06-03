from peer_monitor import PeerStatusMonitor
import netifaces
import yaml

def get_own_address():
    """Detect and return the peer address from config that matches any of the local IPs."""
    # Get list of all IPs on the machine (IPv4 only)
    local_ips = []
    for iface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(iface)
        if netifaces.AF_INET in addrs:
            for addr in addrs[netifaces.AF_INET]:
                local_ips.append(addr['addr'])
    
    print("Detected local IPs:", local_ips)  # Debug output

    with open('host_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        for peer in config['peers']:
            if peer['ip'] in local_ips:
                print(f"Matched local IP {peer['ip']} to peer configuration")  # Debug output
                return f"{peer['ip']}:{peer['port']}"
    return None

def main():
    own_address = get_own_address()
    if not own_address:
        print("Error: Could not find local machine in host_config.yaml")
        print("Available local IPs:", [addr for iface in netifaces.interfaces() 
                                     for addr in netifaces.ifaddresses(iface).get(netifaces.AF_INET, [])])
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
