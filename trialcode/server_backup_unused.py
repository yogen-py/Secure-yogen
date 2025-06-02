# server.py
import grpc
from concurrent import futures
import model_pb2
import model_pb2_grpc
import torch
import pickle

class FLServer(model_pb2_grpc.FLPeerServicer):
    def __init__(self, model):
        self.model = model

    def SendModel(self, request, context):
        peer_state = pickle.loads(request.weights)
        self.aggregate_weights(peer_state)
        return model_pb2.Ack(message="Model received")

    def aggregate_weights(self, peer_state):
        for key in self.model.state_dict():
            self.model.state_dict()[key] = (
                self.model.state_dict()[key] + peer_state[key]
            ) / 2

def serve(model, port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_FLPeerServicer_to_server(FLServer(model), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    server.wait_for_termination()

