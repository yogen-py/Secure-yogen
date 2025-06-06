import grpc
import model_pb2
import model_pb2_grpc
import pickle
import torch
import logging

logger = logging.getLogger(__name__)

def send_model(state_dict, address="localhost:50051", round_num=1, timeout=30, use_ssl=False, ssl_cert=None, node_id=None):
    """
    Send model weights to a peer
    Args:
        state_dict: PyTorch model state dictionary
        address: gRPC server address (host:port)
        round_num: Current federated round number
        timeout: Timeout in seconds
        use_ssl: Whether to use SSL/TLS
        ssl_cert: Path to SSL certificate file
        node_id: Node identifier for logging
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create channel with proper security and increased message size
        grpc_options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
        if use_ssl and ssl_cert:
            with open(ssl_cert, 'rb') as f:
                credentials = grpc.ssl_channel_credentials(f.read())
            channel = grpc.secure_channel(address, credentials, options=grpc_options)
        else:
            channel = grpc.insecure_channel(address, options=grpc_options)

        # Create stub with timeout
        stub = model_pb2_grpc.FLPeerStub(channel)
        try:
            # Serialize and send model
            serialized = pickle.dumps(state_dict)
            response = stub.SendModel(
                model_pb2.ModelWeights(round=round_num, weights=serialized),
                timeout=timeout
            )
            logger.info(f"Model sent successfully to {address} (Node: {node_id}): {response.message}")
            return True
        except grpc.RpcError as rpc_error:
            logger.error(f"RPC error when sending to {address} (Node: {node_id}): {rpc_error.code()}: {rpc_error.details()}")
            return False
        finally:
            channel.close()
    except Exception as e:
        logger.error(f"Failed to send model to {address} (Node: {node_id}): {str(e)}")
        return False

