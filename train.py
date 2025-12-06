import torch
import time
from pathlib import Path


class LRScheduler:
    def __init__(self, init_lr=1.0e-4, lr_decay_epoch=10):
        self.init_lr = init_lr
        self.lr_decay_epoch = lr_decay_epoch

    def __call__(self, optimizer, epoch):
        '''Decay learning rate by a factor of 0.8 every lr_decay_epoch epochs.'''
        lr = self.init_lr * (0.8 ** (epoch // self.lr_decay_epoch))
        lr = max(lr, 1e-8)
        if epoch % self.lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer

# Training and validation function
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_model(model, criterion, optimizer, scheduler, device, train_loader, val_loader, early_stopping_patience,
                timestamp, iteration_timestamp, num_epochs=25, debug=False):
    best_val_loss = float('inf')  # Initialize with infinity for tracking best validation loss
    early_stopping_counter = 0
    best_model = None

    start_time = time.time()
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, total = 0, 0, 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            if labels.dim() > 1:
                labels = labels.squeeze()
            if debug:
                print(f"\n[DEBUG] Epoch {epoch + 1} - Batch {batch_idx + 1}")
                print(f"  Inputs shape: {inputs.shape}")
                print(f"  Labels shape: {labels.shape}")
                print(f"  Labels unique values: {torch.unique(labels)}")
                print(f"  Model output before training step: {model(inputs).shape}")

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            if torch.isnan(loss):
                print("[ERROR] Loss is NaN. Check your loss function or model outputs.")
                print(f"[DEBUG] Model Outputs: {outputs}")
                print(f"[DEBUG] Labels: {labels}")
                print(f"[DEBUG] Loss Calculation Inputs: {loss.item()}")

            _, predicted = outputs.max(1)
            correct_predictions = predicted.eq(labels).sum().item()
            train_correct += correct_predictions
            total += labels.size(0)

        train_acc = 100. * train_correct / total
        history['train_loss'].append(train_loss / total)
        history['train_acc'].append(train_acc)

        print(f"\n[DEBUG] Epoch {epoch + 1} completed")
        print(f"  Train Loss: {train_loss / total:.4f}")
        print(f"  Train Accuracy: {train_acc:.2f}%")
        print(f"  Total examples processed: {total}")

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if labels.dim() > 1:
                    labels = labels.squeeze()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = 100. * val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"[DEBUG] Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

        # Adjust learning rate using PyTorch scheduler (StepLR for smoother decay)
        scheduler(optimizer, epoch)  # Call your custom scheduler directly

        # Early Stopping Logic - Based on Validation Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            early_stopping_counter = 0  # Reset patience counter if validation loss improves
        else:
            early_stopping_counter += 1  # Increment patience counter if no improvement

        if early_stopping_counter >= early_stopping_patience:
            print(
                f'Stopping early at epoch {epoch + 1} due to no improvement in validation loss for {early_stopping_patience} epochs.')
            break

    total_time = time.time() - start_time
    print('The total training time is ', total_time, 'seconds')
    return history, total_time, best_model
