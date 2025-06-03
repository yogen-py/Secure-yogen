import threading
import time
import grpc
import datetime
from tabulate import tabulate
import model_pb2
import model_pb2_grpc
import yaml
import os
from concurrent.futures import ThreadPoolExecutor

class PeerStatusMonitor:
    def __init__(self, own_address, check_interval=30):
        self.own_address = own_address
        self.check_interval = check_interval
        self.peer_status = {}
        self.running = False
        self.lock = threading.Lock()
        self.load_peers()
        
    def load_peers(self):
        """Load peer information from config file"""
        with open('host_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            self.peers = {
                f"{peer['ip']}:{peer['port']}": {
                    'name': peer['name'],
                    'status': 'Unknown',
                    'last_seen': None
                }
                for peer in config['peers']
                if f"{peer['ip']}:{peer['port']}" != self.own_address
            }

    def check_peer_health(self, peer_address):
        """Check health of a single peer"""
        try:
            channel = grpc.insecure_channel(peer_address)
            stub = model_pb2_grpc.FLPeerStub(channel)
            
            request = model_pb2.HealthCheckRequest(
                peer_id=self.own_address,
                timestamp=datetime.datetime.now().isoformat()
            )
            
            # Set a timeout of 5 seconds for the health check
            response = stub.HealthCheck(request, timeout=5)
            
            with self.lock:
                self.peers[peer_address].update({
                    'status': 'Online',
                    'last_seen': datetime.datetime.now()
                })
                
        except Exception as e:
            with self.lock:
                self.peers[peer_address].update({
                    'status': 'Offline',
                    'last_seen': self.peers[peer_address]['last_seen']
                })
        finally:
            channel.close()

    def check_all_peers(self):
        """Check health of all peers concurrently"""
        with ThreadPoolExecutor(max_workers=len(self.peers)) as executor:
            executor.map(self.check_peer_health, self.peers.keys())

    def display_status(self):
        """Display peer status in a formatted table"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n=== Peer Network Status ===")
        print(f"Local Address: {self.own_address}")
        print(f"Last Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        table_data = []
        with self.lock:
            for addr, info in self.peers.items():
                last_seen = info['last_seen'].strftime('%H:%M:%S') if info['last_seen'] else 'Never'
                status_symbol = '🟢' if info['status'] == 'Online' else '🔴'
                table_data.append([
                    info['name'],
                    addr,
                    f"{status_symbol} {info['status']}",
                    last_seen
                ])

        print(tabulate(
            table_data,
            headers=['Peer Name', 'Address', 'Status', 'Last Seen'],
            tablefmt='grid'
        ))
        print("\nPress Ctrl+C to stop monitoring\n")

    def start_monitoring(self):
        """Start the monitoring process"""
        self.running = True
        
        def monitor_loop():
            while self.running:
                self.check_all_peers()
                self.display_status()
                time.sleep(self.check_interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop the monitoring process"""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
