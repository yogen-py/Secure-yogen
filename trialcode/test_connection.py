import grpc
import model_pb2
import model_pb2_grpc
import yaml
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_peer_connection(peer_address):
    try:
        channel = grpc.insecure_channel(peer_address)
        stub = model_pb2_grpc.FLPeerStub(channel)
        
        # Try health check
        request = model_pb2.HealthCheckRequest(
            peer_id="tester",
            timestamp=str(time.time())
        )
        
        response = stub.HealthCheck(request, timeout=5)
        logger.info(f"Successfully connected to {peer_address}: {response.status}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to connect to {peer_address}: {e}")
        return False
    finally:
        channel.close()

def test_all_peers():
    with open('host_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    results = []
    for peer in config['peers']:
        address = f"{peer['ip']}:{peer['port']}"
        success = test_peer_connection(address)
        results.append((peer['name'], address, success))
    
    print("\nConnection Test Results:")
    print("=" * 50)
    for name, addr, success in results:
        status = "✓ Connected" if success else "✗ Failed"
        print(f"{name} ({addr}): {status}")
    print("=" * 50)

if __name__ == "__main__":
    test_all_peers()
