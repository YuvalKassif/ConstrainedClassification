import torch
import torch.nn.functional as F
from losses import CustomLoss
import torch.nn as nn


# Simulated softmax outputs and labels
y_pred = F.softmax(torch.randn(10, 5), dim=1)
y_true = torch.randint(0, 5, (10,))

print(y_pred, '\n', y_true)

# Standard cross-entropy loss
ce_loss = F.cross_entropy(y_pred, y_true)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom loss with C_k = 1
criterion = CustomLoss(constrained_class_index=2, C=torch.tensor([1.0, 1.0, 30.0, 1.0, 1.0]), device=device)
custom_loss = criterion(y_pred, F.one_hot(y_true, num_classes=5).float())
print('Standard Cross-Entropy Loss:', ce_loss.item())
print('Custom Loss (C=1):', custom_loss.item())
print("Diff", ce_loss.item() - custom_loss.item())


