import json
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
import timm
import matplotlib.pyplot as plt
import numpy as np
import random
import datetime
from datetime import datetime
import torch.nn.functional as F


def evaluate_test_accuracy(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def optimization_results(model, test_loader, device, constrained_class_idx, constraints_num):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def count_predictions_per_class(model, test_loader, device):
    model.eval()
    class_count = {}
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            for label in predicted:
                if label.item() in class_count:
                    class_count[label.item()] += 1
                else:
                    class_count[label.item()] = 1

    return class_count


# Set the random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def get_model(model_name, num_classes=5):
    if model_name == 'EfficientNetB0':
        model = timm.create_model('efficientnet_b0', pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'EfficientNetB5':
        model = timm.create_model('efficientnet_b5', pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'ResNet50':
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'ResNet101':
        model = models.resnet101(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'medmnist':
        model = SimpleCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    return model

def plot_training_results(history, model_choice, total_time, params, timestamp, iteration_timestamp):
    exp_name = params["exp_name"]

    # Plotting the training results with an overall title
    plt.figure(figsize=(12, 6))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Overall title
    plt.suptitle(f'Training and Validation Results for {model_choice}, Run Time: {round(total_time)}s')

    # Adjust layout
    plt.tight_layout()

    # Determine the save path
    save_path = Path(f'savedModels1/{timestamp}/{iteration_timestamp}')
    save_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    # Save the figure
    fig_filename = f"{exp_name}_training_results.png"
    plt.savefig(save_path / fig_filename)
    print(f"Plot saved to {save_path / fig_filename}")

    # Show plot
    plt.show()  # Show the plot after saving

    plt.close()  # Close the figure to free memory


def plot_comparison():
    # Data extracted from the image titles
    models = ['EfficientNetB0', 'EfficientNetB5', 'ResNet50', 'ResNet101']
    run_times = [357, 428, 436, 511]  # in seconds
    test_accuracies = [66.18, 68.78, 67.03, 65.94]  # in percentages

    # Create plots
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Plot for Run Time
    ax[0].bar(models, run_times, color=['blue', 'orange', 'green', 'red'])
    ax[0].set_title('Run Time of Each Model')
    ax[0].set_ylabel('Run Time (seconds)')
    ax[0].set_xlabel('Models')
    ax[0].set_xticklabels(models, rotation=45, ha='right')

    # Plot for Test Accuracy
    ax[1].bar(models, test_accuracies, color=['blue', 'orange', 'green', 'red'])
    ax[1].set_title('Test Accuracy of Each Model')
    ax[1].set_ylabel('Test Accuracy (%)')
    ax[1].set_xlabel('Models')
    ax[1].set_xticklabels(models, rotation=45, ha='right')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


def save_parameters(params, exp_name, timestamp, iteration_timestamp, filename="experiment_config.json"):
    # Convert any tensors in params to lists
    serializable_params = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in params.items()}

    # Adding a timestamp to each experiment for uniqueness
    file_path = Path(f'savedModels1/{timestamp}/{iteration_timestamp}/{filename}')
    # Create the directory if it does not exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(serializable_params, f, indent=4)

    print(f"Parameters saved to {file_path}")

def save_test_counts(class_count, exp_name, timestamp, iteration_timestamp, filename="class_count.json"):
    # Adding a timestamp to each experiment for uniqueness
    file_path = Path(f'savedModels1/{timestamp}/{iteration_timestamp}/{filename}')
    # Create the directory if it does not exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(class_count, f, indent=4)

    print(f"Class count saved to {file_path}")

def generate_exp_name():
    current_datetime = datetime.now()
    return current_datetime.strftime("%d-%m-%y---%H-%M")


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 128 * 3 * 3)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

