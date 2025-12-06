import torch
import torch.nn.functional as F
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, constrained_class_index, C, device='cpu'):
        super(CustomLoss, self).__init__()
        self.constrained_class_index = constrained_class_index
        self.C = C.to(device)
        self.device = device

    def forward(self, y_pred, y_true):
        # Convert logits to probabilities
        y_pred = F.softmax(y_pred, dim=1).to(self.device)
        # Check and convert y_true to one-hot if necessary
        if y_true.dtype == torch.int64:
            y_true = F.one_hot(y_true, num_classes=y_pred.shape[1]).float()
        y_true = y_true.to(self.device)

        # Access the probability of the constrained class and calculate max probability
        y_k_constrained = y_pred[:, self.constrained_class_index]
        max_y_t = torch.max(y_pred, dim=1)[0]

        # print("y_k_constrained", y_k_constrained, '\n', "max_y_t", max_y_t)

        # Calculate the tanh modulation term
        tanh_term = torch.tanh(50000000 * (max_y_t - y_k_constrained)).unsqueeze(1)

        # Added this normalization -> consider to erase
        # self.C = self.C / torch.max(self.C) * 5  # or torch.sum(self.C)

        # Calculate loss1
        # term1_log_scale = self.C.unsqueeze(0).detach() * torch.log(1 + (y_pred - 1) * (1 - tanh_term))
        term1_log_scale = self.C.unsqueeze(0).detach() * torch.log(1e-7 + torch.relu(1 + (y_pred - 1) * (1 - tanh_term)))
        loss1 = -torch.sum(y_true * term1_log_scale, dim=1)
        # print(y_true * term1_log_scale)

        # Calculate loss2y_true
        term2 = (y_pred - 1) * tanh_term
        # loss2 = -torch.sum(y_true * torch.log(1e-7 + 1 + term2), dim=1)
        loss2 = -torch.sum(y_true * torch.log(1e-7 + torch.relu(1 + term2)), dim=1)

        # Average loss over the batch
        total_loss = torch.mean(loss1 + loss2)
        return total_loss


