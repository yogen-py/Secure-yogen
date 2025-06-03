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
    def __init__(self, own_address, check_interval=5):
        self.own_address = own_address
        self.check_interval = check_interval
        self.peer_status = {}
        self.running = False
        self.lock = threading.Lock()
        self.last_display_time = 0
        self.display_interval = 1  # Minimum time between display updates
        self.load_peers()
        
    def load_peers(self):
        """Load peer information from config file"""
        with open('host_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            self.peers = {
                f"{peer['ip']}:{peer['port']}": {
                    'name': peer['name'],
                    'status': 'Unknown',
                    'last_seen': None,
                    'latency': None
                }
                for peer in config['peers']
                if f"{peer['ip']}:{peer['port']}" != self.own_address
            }

    def check_peer_health(self, peer_address):
        """Check health of a single peer"""
        try:
            start_time = time.time()
            channel = grpc.insecure_channel(peer_address)
            stub = model_pb2_grpc.FLPeerStub(channel)
            
            request = model_pb2.HealthCheckRequest(
                peer_id=self.own_address,
                timestamp=datetime.datetime.now().isoformat()
            )
            
            # Set a timeout of 2 seconds for the health check (reduced from 5)
            response = stub.HealthCheck(request, timeout=2)
            
            # Calculate latency
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            with self.lock:
                self.peers[peer_address].update({
                    'status': 'Online',
                    'last_seen': datetime.datetime.now(),
                    'latency': f"{latency:.0f}ms"
                })
                
        except Exception as e:
            with self.lock:
                self.peers[peer_address].update({
                    'status': 'Offline',
                    'last_seen': self.peers[peer_address]['last_seen'],
                    'latency': None
                })
        finally:
            channel.close()

    def check_all_peers(self):
        """Check health of all peers concurrently"""
        with ThreadPoolExecutor(max_workers=len(self.peers)) as executor:
            executor.map(self.check_peer_health, self.peers.keys())

    def should_update_display(self):
        """Check if enough time has passed to update the display"""
        current_time = time.time()
        if current_time - self.last_display_time >= self.display_interval:
            self.last_display_time = current_time
            return True
        return False

    def display_status(self):
        """Display peer status in a formatted table"""
        if not self.should_update_display():
            return

        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n=== Peer Network Status ===")
        print(f"Local Address: {self.own_address}")
        print(f"Last Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Update Interval: {self.check_interval}s\n")

        table_data = []
        with self.lock:
            for addr, info in self.peers.items():
                last_seen = info['last_seen'].strftime('%H:%M:%S') if info['last_seen'] else 'Never'
                status_symbol = '🟢' if info['status'] == 'Online' else '🔴'
                latency = info['latency'] if info['latency'] else 'N/A'
                table_data.append([
                    info['name'],
                    addr,
                    f"{status_symbol} {info['status']}",
                    last_seen,
                    latency
                ])

        print(tabulate(
            table_data,
            headers=['Peer Name', 'Address', 'Status', 'Last Seen', 'Latency'],
            tablefmt='grid'
        ))
        print("\nPress Ctrl+C to stop monitoring\n")

    def start_monitoring(self):
        """Start the monitoring process"""
        self.running = True
        
        def monitor_loop():
            while self.running:
                loop_start = time.time()
                
                self.check_all_peers()
                self.display_status()
                
                # Calculate sleep time to maintain consistent interval
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.check_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop the monitoring process"""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
