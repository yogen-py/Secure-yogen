from model import SimpleBinaryClassifier, set_seed, get_loss, get_optimizer
from data import load_and_preprocess_data
from torch.utils.data import DataLoader, TensorDataset

def train_local(epochs=3, batch_size=64):
    set_seed(42)
    X_train, y_train, _, _ = load_and_preprocess_data()
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = SimpleBinaryClassifier(X_train.shape[1])
    criterion = get_loss()
    optimizer = get_optimizer(model)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        avg_loss = epoch_loss / len(loader.dataset)
        print(f"[TRAIN] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    return model.state_dict()

