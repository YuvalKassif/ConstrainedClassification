import torch
import torch.nn.functional as F
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, constrained_class_index, C, device='cpu'):
        super(CustomLoss, self).__init__()
        self.constrained_class_index = constrained_class_index
        self.C = C.to(device)
        self.device = device
        # Expose recent tanh modulation summary for dynamic LR control
        self.last_tanh_mean = None

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
        # Cache a scalar summary for external use (e.g., LR selection)
        # Clamp to [0,1] and detach to avoid graph retention
        try:
            self.last_tanh_mean = torch.clamp(tanh_term.mean(), 0.0, 1.0).detach()
        except Exception:
            # Fallback in case of unexpected shapes
            self.last_tanh_mean = None

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


def _solve_exact_k_assignment(costs, constrained_class_index, k):
    """Solve per-sample label assignment with an exact K constraint on one class."""
    with torch.no_grad():
        batch_size, num_classes = costs.shape
        k = max(0, min(int(k), batch_size))

        cost_k = costs[:, constrained_class_index]
        costs_other = costs.clone()
        costs_other[:, constrained_class_index] = float("inf")
        best_other_cost, best_other_class = torch.min(costs_other, dim=1)

        if k <= 0:
            chosen = torch.zeros(batch_size, dtype=torch.bool, device=costs.device)
        elif k >= batch_size:
            chosen = torch.ones(batch_size, dtype=torch.bool, device=costs.device)
        else:
            diff = cost_k - best_other_cost
            _, idx = torch.topk(diff, k, largest=False)
            chosen = torch.zeros(batch_size, dtype=torch.bool, device=costs.device)
            chosen[idx] = True

        decisions = torch.zeros_like(costs)
        if chosen.any():
            decisions[chosen, constrained_class_index] = 1.0
        rest = ~chosen
        if rest.any():
            decisions[rest, best_other_class[rest]] = 1.0

        return decisions


class SpoPlusLoss(nn.Module):
    """SPO+ surrogate loss for constrained label assignment (exact K of one class)."""

    def __init__(self, constrained_class_index, constraint_ratio, num_classes, device="cpu"):
        super().__init__()
        self.constrained_class_index = int(constrained_class_index)
        self.constraint_ratio = float(constraint_ratio)
        self.num_classes = int(num_classes)
        self.device = device
        # Keep attributes used by existing training loop; SPO+ doesn't use them.
        self.last_tanh_mean = None
        self.C = torch.ones(self.num_classes, device=self.device)

    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1).to(self.device)
        if y_true.dtype == torch.int64:
            y_true = F.one_hot(y_true, num_classes=y_pred.shape[1]).float()
        y_true = y_true.to(self.device)

        true_cost = 1.0 - y_true
        pred_cost = 1.0 - y_pred

        batch_size = y_pred.shape[0]
        k = int(round(self.constraint_ratio * batch_size))

        z_true = _solve_exact_k_assignment(true_cost, self.constrained_class_index, k)
        z_spo = _solve_exact_k_assignment(2.0 * pred_cost - true_cost, self.constrained_class_index, k)

        loss = (2.0 * pred_cost - true_cost) * z_spo - pred_cost * z_true
        return loss.sum(dim=1).mean()


