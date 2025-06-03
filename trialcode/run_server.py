import grpc
from concurrent import futures
import model_pb2
import model_pb2_grpc
import pickle
import torch
import logging
import threading
import socket
from grpc_server import serve

if __name__ == "__main__":
    try:
        print("\n" + "="*50)
        print("Starting Federated Learning Server")
        print("="*50 + "\n")
        
        # Start the server
        serve()
        
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nError: {str(e)}")