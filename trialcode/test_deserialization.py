import torch
import os

# Directory containing serialized model files
MODEL_DIR = "."

# Function to test deserialization of model files
def test_deserialization():
    for filename in os.listdir(MODEL_DIR):
        if filename.startswith("sent_model_PEER") and filename.endswith(".pt"):
            filepath = os.path.join(MODEL_DIR, filename)
            try:
                print(f"Testing deserialization for: {filename}")
                state_dict = torch.load(filepath, map_location="cpu")
                print(f"Successfully deserialized: {filename}")
                print(f"Keys in state_dict: {list(state_dict.keys())}")
            except Exception as e:
                print(f"Failed to deserialize: {filename}")
                print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_deserialization()
