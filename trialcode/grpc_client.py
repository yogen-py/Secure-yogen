import grpc
import model_pb2
import model_pb2_grpc
import pickle
import torch
import logging

logger = logging.getLogger(__name__)

def send_model(state_dict, address="localhost:50051", timeout=30, use_ssl=False, ssl_cert=None):
    """
    Send model weights to a peer
    Args:
        state_dict: PyTorch model state dictionary
        address: gRPC server address (host:port)
        timeout: Timeout in seconds
        use_ssl: Whether to use SSL/TLS
        ssl_cert: Path to SSL certificate file
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create channel with proper security
        if use_ssl and ssl_cert:
            with open(ssl_cert, 'rb') as f:
                credentials = grpc.ssl_channel_credentials(f.read())
            channel = grpc.secure_channel(address, credentials)
        else:
            channel = grpc.insecure_channel(address)

        # Create stub with timeout
        stub = model_pb2_grpc.FLPeerStub(channel)
        try:
            # Serialize and send model
            serialized = pickle.dumps(state_dict)
            response = stub.SendModel(
                model_pb2.ModelWeights(weights=serialized),
                timeout=timeout
            )
            logger.info(f"Model sent successfully to {address}: {response.message}")
            return True
        except grpc.RpcError as rpc_error:
            logger.error(f"RPC error when sending to {address}: {rpc_error.code()}: {rpc_error.details()}")
            return False
        finally:
            channel.close()
    except Exception as e:
        logger.error(f"Failed to send model to {address}: {str(e)}")
        return False

