import grpc
import model_pb2
import model_pb2_grpc
import yaml
import time
import logging
import socket
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import netifaces

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_local_ips():
    """Get all local IP addresses"""
    local_ips = []
    for iface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(iface)
        if netifaces.AF_INET in addrs:
            for addr in addrs[netifaces.AF_INET]:
                local_ips.append(addr['addr'])
    return local_ips

def check_port_open(host, port, timeout=2):
    """Check if a port is open"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        result = sock.connect_ex((host, port))
        return result == 0
    except socket.error:
        return False
    finally:
        sock.close()

def test_peer_connection(peer_info):
    """Test connection to a single peer with detailed diagnostics"""
    name = peer_info['name']
    ip = peer_info['ip']
    port = peer_info['port']
    address = f"{ip}:{port}"
    
    results = {
        'name': name,
        'address': address,
        'port_open': False,
        'grpc_available': False,
        'health_check': False,
        'latency': None,
        'error': None
    }
    
    # Check if this is a local address
    local_ips = get_local_ips()
    if ip in local_ips:
        results['is_local'] = True
    
    # Test port
    if not check_port_open(ip, port):
        results['error'] = f"Port {port} is not open"
        return results
    
    results['port_open'] = True
    
    try:
        start_time = time.time()
        channel = grpc.insecure_channel(address)
        
        # Try to create stub
        try:
            stub = model_pb2_grpc.FLPeerStub(channel)
            results['grpc_available'] = True
        except Exception as e:
            results['error'] = f"gRPC stub creation failed: {str(e)}"
            return results
        
        # Try health check
        try:
            request = model_pb2.HealthCheckRequest(
                peer_id="tester",
                timestamp=str(time.time())
            )
            response = stub.HealthCheck(request, timeout=5)
            results['health_check'] = True
            results['latency'] = (time.time() - start_time) * 1000  # ms
            
        except grpc.RpcError as e:
            results['error'] = f"RPC error: {e.code()}: {e.details()}"
        except Exception as e:
            results['error'] = f"Health check failed: {str(e)}"
            
    except Exception as e:
        results['error'] = f"Connection failed: {str(e)}"
    finally:
        if 'channel' in locals():
            channel.close()
    
    return results

def test_all_peers():
    """Test connections to all peers with parallel execution"""
    try:
        with open('host_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load host_config.yaml: {e}")
        return
    
    print("\nTesting peer connections...")
    print("=" * 60)
    
    # Test connections in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_peer = {
            executor.submit(test_peer_connection, peer): peer
            for peer in config['peers']
        }
        
        results = []
        for future in as_completed(future_to_peer):
            results.append(future.result())
    
    # Display results
    print("\nConnection Test Results:")
    print("=" * 60)
    
    success_count = 0
    for result in results:
        status_parts = []
        
        # Basic connection status
        if result['health_check']:
            status = "✓ Connected"
            success_count += 1
            if result.get('is_local'):
                status += " (Local)"
            if result['latency']:
                status += f" ({result['latency']:.0f}ms)"
        else:
            status = "✗ Failed"
        
        # Add diagnostic details if failed
        if not result['health_check']:
            if result['port_open']:
                status_parts.append("Port: Open")
            else:
                status_parts.append("Port: Closed")
            
            if result['grpc_available']:
                status_parts.append("gRPC: Available")
            else:
                status_parts.append("gRPC: Unavailable")
                
            if result['error']:
                status_parts.append(f"Error: {result['error']}")
        
        # Print result
        print(f"{result['name']} ({result['address']}): {status}")
        if status_parts:
            print("  " + " | ".join(status_parts))
    
    print("=" * 60)
    print(f"Summary: {success_count}/{len(results)} peers accessible")
    
    if success_count == 0:
        print("\nTroubleshooting steps:")
        print("1. Run 'python regenerate_grpc.py' to rebuild gRPC files")
        print("2. Check if the gRPC server is running on each peer")
        print("3. Verify firewall settings allow port 50051")
        print("4. Ensure all peers are on the same network or have network connectivity")
        sys.exit(1)

if __name__ == "__main__":
    test_all_peers()
