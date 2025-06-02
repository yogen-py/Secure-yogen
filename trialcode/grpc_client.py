import grpc
import model_pb2
import model_pb2_grpc
import pickle
import torch

def send_model(state_dict, address="localhost:50051"):
    channel = grpc.insecure_channel(address)
    stub = model_pb2_grpc.FLPeerStub(channel)
    serialized = pickle.dumps(state_dict)
    response = stub.SendModel(model_pb2.ModelWeights(weights=serialized))
    print("Server says:", response.message)

