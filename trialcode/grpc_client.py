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
        logger.info(f"[SEND_MODEL] Creating gRPC channel to {address} (Node: {node_id})")
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

        logger.info(f"[SEND_MODEL] Channel created successfully to {address} (Node: {node_id})")
        stub = model_pb2_grpc.FLPeerStub(channel)
        try:
            logger.info(f"[SEND_MODEL] Serializing model weights for Node: {node_id}")
            serialized = pickle.dumps(state_dict)
            logger.info(f"[SEND_MODEL] Sending model to {address} (Node: {node_id})")
            response = stub.SendModel(
                model_pb2.ModelWeights(round=round_num, weights=serialized),
                timeout=timeout
            )
            logger.info(f"[SEND_MODEL] Model sent successfully to {address} (Node: {node_id}): {response.message}")
            return True
        except grpc.RpcError as rpc_error:
            logger.error(f"[SEND_MODEL] RPC error when sending to {address} (Node: {node_id}): {rpc_error.code()}: {rpc_error.details()}")
            return False
        finally:
            channel.close()
    except Exception as e:
        logger.error(f"[SEND_MODEL] Failed to send model to {address} (Node: {node_id}): {str(e)}")
        return False

def test_health_check(address="localhost:50051", node_id=None):
    """
    Test the HealthCheck RPC to verify peer connectivity.
    Args:
        address: gRPC server address (host:port)
        node_id: Node identifier for logging
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"[HEALTH_CHECK] Creating gRPC channel to {address} (Node: {node_id})")
        grpc_options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
        channel = grpc.insecure_channel(address, options=grpc_options)
        stub = model_pb2_grpc.FLPeerStub(channel)
        logger.info(f"[HEALTH_CHECK] Sending health check to {address} (Node: {node_id})")
        response = stub.HealthCheck(
            model_pb2.HealthCheckRequest(peer_id=node_id),
            timeout=10
        )
        logger.info(f"[HEALTH_CHECK] Health check successful: {response.status} (Node: {node_id})")
        return True
    except grpc.RpcError as rpc_error:
        logger.error(f"[HEALTH_CHECK] RPC error when checking health of {address} (Node: {node_id}): {rpc_error.code()}: {rpc_error.details()}")
        return False
    except Exception as e:
        logger.error(f"[HEALTH_CHECK] Failed to check health of {address} (Node: {node_id}): {str(e)}")
        return False

