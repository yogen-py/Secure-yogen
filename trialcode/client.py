import grpc
import model_pb2
import model_pb2_grpc
import torch
import pickle
from model import SimpleModel

def send_model(model, peer_address):
    channel = grpc.insecure_channel(peer_address)
    stub = model_pb2_grpc.FLPeerStub(channel)
    data = pickle.dumps(model.state_dict())
    response = stub.SendModel(model_pb2.ModelWeights(weights=data))
    print(f"Sent to {peer_address}: {response.message}"
