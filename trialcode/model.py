import torch
import torch.nn as nn
import numpy as np
import random


# ----------------------------
# Set seeds for reproducibility
# ----------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Binary Classifier Model
# ----------------------------
class SimpleBinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SimpleBinaryClassifier, self).__init__()
        self.output_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = torch.sigmoid(self.output_layer(x))
        return x


# ----------------------------
# Custom Metrics (F1, Precision, Recall)
# ----------------------------
def binary_metrics(y_true, y_pred, threshold=0.5):
    """
    Args:
        y_true (torch.Tensor): Ground truth labels (N,)
        y_pred (torch.Tensor): Predicted probabilities (N, 1)
    Returns:
        precision, recall, f1
    """
    y_pred_label = (y_pred > threshold).float()
    y_true = y_true.view(-1, 1)
    
    true_positive = (y_pred_label * y_true).sum()
    predicted_positive = y_pred_label.sum()
    actual_positive = y_true.sum()

    precision = true_positive / (predicted_positive + 1e-8)
    recall = true_positive / (actual_positive + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return precision.item(), recall.item(), f1.item()


# ----------------------------
# Loss and Optimizer Setup
# ----------------------------
def get_loss():
    return nn.BCELoss()

def get_optimizer(model, lr=0.001):
    return torch.optim.Adam(model.parameters(), lr=lr)

