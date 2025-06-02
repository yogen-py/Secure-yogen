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
        for xb, yb in loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
    return model.state_dict()

