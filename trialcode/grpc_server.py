import grpc
from concurrent import futures
import model_pb2
import model_pb2_grpc
import pickle
import torch
import logging
import threading
from queue import Queue
import ssl
import datetime
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Thread-safe queue for received models
received_models = []
model_lock = threading.Lock()

# Global current round
current_round = 1

SAVE_MODEL_DEBUG = True  # Toggle to save received models for inspection
NODE_ID = 'NodeB'  # Set this to a unique identifier for each node (e.g., from host_config.yaml)

def set_current_round(r):
    global current_round
    current_round = r

class FLPeerServicer(model_pb2_grpc.FLPeerServicer):
    def SendModel(self, request, context):
        peer_addr = context.peer()
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if hasattr(request, 'round') and request.round != current_round:
            logger.info(f"Ignored model from {peer_addr} for round {request.round} (current round: {current_round})")
            print(f"[SERVER][{now}] Ignored model from {peer_addr} for round {request.round} (current round: {current_round}) | Node: {NODE_ID}")
            return model_pb2.Ack(message="Ignored: wrong round")
        try:
            # Deserialize model weights
            state_dict = pickle.loads(request.weights)
            
            # Thread-safe append to received models
            with model_lock:
                received_models.append(state_dict)
                count = len(received_models)
            
            # Model size and checksum
            model_size = sum(v.numel() for v in state_dict.values() if torch.is_tensor(v)) * 4 / 1024
            checksum = hashlib.sha256()
            for k in sorted(state_dict.keys()):
                v = state_dict[k]
                if torch.is_tensor(v):
                    checksum.update(v.cpu().numpy().tobytes())
            checksum_str = checksum.hexdigest()[:12]
            logger.info(f"Received model weights from {peer_addr} (total received: {count}) | Size: {model_size:.2f} KB | Checksum: {checksum_str} | Node: {NODE_ID}")
            print(f"[SERVER][{now}] Received model from {peer_addr} (total received: {count}) | Size: {model_size:.2f} KB | Checksum: {checksum_str} | Node: {NODE_ID}")
            if SAVE_MODEL_DEBUG:
                fname = f"received_model_{NODE_ID}_from_{peer_addr.replace(':', '_')}_round{current_round}_idx{count}.pt"
                torch.save(state_dict, fname)
                print(f"[SERVER][DEBUG] Saved received model to {fname}")
            return model_pb2.Ack(message="Model received successfully")
            
        except pickle.UnpicklingError as e:
            logger.error(f"Failed to deserialize model from {context.peer()}: {str(e)}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Invalid model format")
            return model_pb2.Ack(message="Error: Invalid model format")
            
        except Exception as e:
            logger.error(f"Error processing model from {context.peer()}: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return model_pb2.Ack(message=f"Error: {str(e)}")

    def HealthCheck(self, request, context):
        """Handle health check requests"""
        try:
            return model_pb2.HealthCheckResponse(
                status="OK",
                peer_id=request.peer_id,
                timestamp=datetime.datetime.now().isoformat()
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return model_pb2.HealthCheckResponse(
                status="ERROR",
                peer_id=request.peer_id,
                timestamp=datetime.datetime.now().isoformat()
            )

def serve(port=50051, ssl_key=None, ssl_cert=None):
    """
    Start the gRPC server
    Args:
        port: Port number to listen on
        ssl_key: Path to SSL private key file
        ssl_cert: Path to SSL certificate file
    """
    try:
        grpc_options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=grpc_options)
        model_pb2_grpc.add_FLPeerServicer_to_server(FLPeerServicer(), server)
        
        if ssl_key and ssl_cert:
            # Secure connection
            with open(ssl_key, 'rb') as f:
                private_key = f.read()
            with open(ssl_cert, 'rb') as f:
                certificate_chain = f.read()
            
            credentials = grpc.ssl_server_credentials([(private_key, certificate_chain)])
            server.add_secure_port(f'[::]:{port}', credentials)
            logger.info(f"Starting secure gRPC server on port {port}")
        else:
            # Insecure connection (not recommended for production)
            server.add_insecure_port(f'[::]:{port}')
            logger.warning(f"Starting insecure gRPC server on port {port}")
        
        server.start()
        logger.info("Server started successfully")
        server.wait_for_termination()
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise

