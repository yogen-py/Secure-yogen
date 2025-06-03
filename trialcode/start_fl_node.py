import os
import sys
import time
import subprocess
import logging
import socket
import netifaces
import yaml
import signal
import psutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_port_in_use(port):
    """Check if port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def kill_process_on_port(port):
    """Kill any process using the specified port"""
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        try:
            for conn in proc.connections():
                if conn.laddr.port == port:
                    os.kill(proc.pid, signal.SIGTERM)
                    logger.info(f"Killed process {proc.pid} using port {port}")
                    time.sleep(1)
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False

def regenerate_grpc_files():
    """Regenerate gRPC files"""
    logger.info("Regenerating gRPC files...")
    result = subprocess.run([sys.executable, 'regenerate_grpc.py'], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Failed to regenerate gRPC files:")
        logger.error(result.stderr)
        return False
    return True

def start_grpc_server(port=50051):
    """Start the gRPC server"""
    # Check if port is in use
    if check_port_in_use(port):
        logger.warning(f"Port {port} is already in use")
        if kill_process_on_port(port):
            logger.info("Cleared port for use")
        else:
            logger.error(f"Could not free port {port}")
            return None

    # Start the server
    logger.info("Starting gRPC server...")
    server_process = subprocess.Popen(
        [sys.executable, 'run_server.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    time.sleep(2)
    
    # Check if server is running
    if server_process.poll() is not None:
        logger.error("Server failed to start:")
        logger.error(server_process.stderr.read())
        return None
        
    return server_process

def test_connections():
    """Test connections to peers"""
    logger.info("Testing peer connections...")
    result = subprocess.run([sys.executable, 'test_connection.py'], capture_output=True, text=True)
    print(result.stdout)
    if "0/4 peers accessible" in result.stdout:
        logger.warning("No peers are accessible. Check if other nodes are running.")
    return True

def start_fl_process():
    """Start the main FL process"""
    logger.info("Starting main FL process...")
    fl_process = subprocess.Popen(
        [sys.executable, 'main.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return fl_process

def main():
    try:
        print("\n=== Starting Federated Learning Node ===\n")
        
        # 1. Regenerate gRPC files
        if not regenerate_grpc_files():
            logger.error("Failed to regenerate gRPC files. Exiting.")
            return
        
        # 2. Start gRPC server
        server_process = start_grpc_server()
        if not server_process:
            logger.error("Failed to start gRPC server. Exiting.")
            return
        
        # 3. Wait for server to initialize
        time.sleep(3)
        
        # 4. Test connections
        test_connections()
        
        # 5. Start FL process
        fl_process = start_fl_process()
        
        print("\nNode is running. Press Ctrl+C to stop.")
        
        # Monitor processes
        while True:
            if server_process.poll() is not None:
                logger.error("gRPC server has stopped unexpectedly!")
                break
            if fl_process.poll() is not None:
                logger.error("FL process has stopped unexpectedly!")
                break
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Cleanup
        if 'server_process' in locals():
            server_process.terminate()
        if 'fl_process' in locals():
            fl_process.terminate()
        
        # Wait for processes to close
        time.sleep(2)
        
        # Force kill if necessary
        for proc in [server_process, fl_process]:
            if proc and proc.poll() is None:
                proc.kill()

if __name__ == "__main__":
    main()
