import os
import torch
from model import SimpleBinaryClassifier
from data import get_data_loaders
from train import evaluate


def list_saved_models(save_dir="models"):
    """List all saved model files."""
    if not os.path.exists(save_dir):
        return []
    return [f for f in os.listdir(save_dir) if f.endswith('.pt')]


def load_model(model_path, input_dim):
    """Load a model from a .pt file."""
    model = SimpleBinaryClassifier(input_dim)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_saved_model(model_path, machine_id=0, total_machines=4, batch_size=64):
    """Evaluate a saved model on the test set."""
    train_loader, test_loader = get_data_loaders(machine_id=machine_id, total_machines=total_machines, batch_size=batch_size)
    input_dim = next(iter(train_loader))[0].shape[1]
    model = load_model(model_path, input_dim)
    acc, prec, rec, f1 = evaluate(model, test_loader)
    print(f"Evaluation for {model_path}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")


def predict(model_path, input_tensor):
    """Run inference on a single input tensor (1D torch tensor)."""
    input_dim = input_tensor.shape[0]
    model = load_model(model_path, input_dim)
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))
        prob = output.item()
        pred = int(prob > 0.5)
    return pred, prob


def main_cli():
    print("\nModel Interface CLI")
    print("==================\n")
    while True:
        print("Options:")
        print("  1. List saved models")
        print("  2. Evaluate a saved model")
        print("  3. Predict with a saved model")
        print("  4. Exit")
        choice = input("Select an option (1-4): ").strip()
        if choice == '1':
            models = list_saved_models()
            if not models:
                print("No saved models found.")
            else:
                print("Saved models:")
                for i, m in enumerate(models):
                    print(f"  {i+1}. {m}")
        elif choice == '2':
            models = list_saved_models()
            if not models:
                print("No saved models to evaluate.")
                continue
            for i, m in enumerate(models):
                print(f"  {i+1}. {m}")
            idx = input("Select model number: ").strip()
            try:
                idx = int(idx) - 1
                model_path = os.path.join("models", models[idx])
                evaluate_saved_model(model_path)
            except Exception as e:
                print(f"Invalid selection: {e}")
        elif choice == '3':
            models = list_saved_models()
            if not models:
                print("No saved models to use for prediction.")
                continue
            for i, m in enumerate(models):
                print(f"  {i+1}. {m}")
            idx = input("Select model number: ").strip()
            try:
                idx = int(idx) - 1
                model_path = os.path.join("models", models[idx])
                # Get input dimension
                train_loader, _ = get_data_loaders()
                input_dim = next(iter(train_loader))[0].shape[1]
                print(f"Enter {input_dim} comma-separated float values for input:")
                raw = input().strip()
                values = [float(x) for x in raw.split(",")]
                if len(values) != input_dim:
                    print(f"Expected {input_dim} values, got {len(values)}.")
                    continue
                input_tensor = torch.tensor(values, dtype=torch.float32)
                pred, prob = predict(model_path, input_tensor)
                print(f"Prediction: {pred} (probability: {prob:.4f})")
            except Exception as e:
                print(f"Invalid input or selection: {e}")
        elif choice == '4':
            print("Exiting.")
            break
        else:
            print("Invalid option. Please select 1-4.")

if __name__ == "__main__":
    main_cli() 