import grpc
from concurrent import futures
import model_pb2
import model_pb2_grpc
import pickle
import torch

# Shared store for received models (could also use a queue or database)
received_models = []

class FLPeerServicer(model_pb2_grpc.FLPeerServicer):
    def SendModel(self, request, context):
        try:
            state_dict = pickle.loads(request.weights)
            received_models.append(state_dict)
            print("Received model weights")
            return model_pb2.Ack(message="Model received successfully")
        except Exception as e:
            print(f"Failed to receive model: {e}")
            return model_pb2.Ack(message="Error receiving model")

def serve(port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_FLPeerServicer_to_server(FLPeerServicer(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"gRPC server started on port {port}")
    server.wait_for_termination()

