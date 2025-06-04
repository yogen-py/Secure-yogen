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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Thread-safe queue for received models
received_models = []
model_lock = threading.Lock()

class FLPeerServicer(model_pb2_grpc.FLPeerServicer):
    def SendModel(self, request, context):
        try:
            # Deserialize model weights
            state_dict = pickle.loads(request.weights)
            
            # Thread-safe append to received models
            with model_lock:
                received_models.append(state_dict)
                count = len(received_models)
            
            peer_addr = context.peer()
            logger.info(f"Received model weights from {peer_addr} (total received: {count})")
            print(f"[SERVER] Received model from {peer_addr} (total received: {count})")
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
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
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

