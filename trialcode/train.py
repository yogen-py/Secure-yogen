from model import SimpleBinaryClassifier, set_seed, get_loss, get_optimizer
from data import get_data_loaders
import torch
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

def evaluate(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in test_loader:
            outputs = model(xb)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    all_preds = [int(p[0]) for p in all_preds]
    all_labels = [int(l[0]) for l in all_labels]
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return acc, prec, rec, f1

def train_local(epochs=3, batch_size=64, save_dir="models", machine_id=0, total_machines=4):
    set_seed(42)
    os.makedirs(save_dir, exist_ok=True)
    train_loader, test_loader = get_data_loaders(machine_id=machine_id, total_machines=total_machines, batch_size=batch_size)
    input_dim = next(iter(train_loader))[0].shape[1]
    model = SimpleBinaryClassifier(input_dim)
    criterion = get_loss()
    optimizer = get_optimizer(model)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, ncols=100)
        for xb, yb in progress_bar:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
            progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
        avg_loss = epoch_loss / len(train_loader.dataset)
        # Save model checkpoint
        checkpoint_path = os.path.join(save_dir, f"model_machine{machine_id}_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        # Evaluate
        acc, prec, rec, f1 = evaluate(model, test_loader)
        tqdm.write(f"[TRAIN] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        tqdm.write(f"[EVAL]  Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    return model.state_dict()

