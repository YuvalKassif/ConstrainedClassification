import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import pandas as pd
from config import get_experiment_config
from load_data import get_dataloaders
from utils import get_model

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

import torch
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def constrained_classification(model, model_path, test_loader, constrained_class_idx, constraints_number, device, save_path='raw_results.csv'):
    # Load the model state dict if a model path is provided
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Store probabilities and labels for post-processing
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            all_probabilities.append(probabilities.cpu())
            all_labels.append(labels.cpu())

    all_probabilities = torch.cat(all_probabilities)
    all_labels = torch.cat(all_labels)

    # Flatten labels to 1D and get constrained class probabilities
    all_labels = all_labels.view(-1)
    constrained_probabilities = all_probabilities[:, constrained_class_idx].view(-1)

    # Create predictions array and keep track of constrained class assignments
    predictions = torch.zeros_like(all_labels)

    # Debug prints
    print(f"[DEBUG] all_labels shape: {all_labels.shape}")
    print(f"[DEBUG] constrained_probabilities shape: {constrained_probabilities.shape}")
    print(f"[DEBUG] predictions shape: {predictions.shape}")

    # Sort indices by probability of the constrained class in descending order
    sorted_indices = torch.argsort(constrained_probabilities, descending=True)

    # Assign predictions based on constraints
    constrained_class_count = 0
    for idx in sorted_indices:
        if constrained_class_count < constraints_number and constrained_probabilities[idx] > 0:
            predictions[idx] = constrained_class_idx
            constrained_class_count += 1
        else:
            probabilities = all_probabilities[idx]
            probabilities[constrained_class_idx] = 0  # Exclude constrained class
            predictions[idx] = torch.argmax(probabilities)

    # Save raw data to a CSV file for manual calculations
    try:
        raw_data = pd.DataFrame({
            'actual_label': all_labels.numpy(),
            'predicted_label': predictions.numpy(),
            'constrained_probability': constrained_probabilities.numpy()
        })

        # Add probabilities for each class to the DataFrame
        for class_idx in range(all_probabilities.size(1)):
            raw_data[f'probability_class_{class_idx}'] = all_probabilities[:, class_idx].numpy()

        raw_data.to_csv(save_path, index=False)
        print(f"Raw data saved to {save_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save raw data to CSV: {e}")

    # Evaluate with both macro and weighted averages
    accuracy = evaluate_test_accuracy_with_custom_predictions(predictions, all_labels)
    precision_macro = precision_score(all_labels.numpy(), predictions.numpy(), average='macro')
    recall_macro = recall_score(all_labels.numpy(), predictions.numpy(), average='macro')
    f1_macro = f1_score(all_labels.numpy(), predictions.numpy(), average='macro')

    precision_weighted = precision_score(all_labels.numpy(), predictions.numpy(), average='weighted')
    recall_weighted = recall_score(all_labels.numpy(), predictions.numpy(), average='weighted')
    f1_weighted = f1_score(all_labels.numpy(), predictions.numpy(), average='weighted')

    # Optionally print a detailed classification report
    print(classification_report(all_labels.numpy(), predictions.numpy()))

    # Return both macro and weighted metrics
    return {
        'accuracy': accuracy,
        'macro': {'precision': precision_macro, 'recall': recall_macro, 'f1': f1_macro},
        'weighted': {'precision': precision_weighted, 'recall': recall_weighted, 'f1': f1_weighted}
    }

def evaluate_test_accuracy_with_custom_predictions(predictions, actual_labels):
    correct = (predictions == actual_labels).sum().item()
    total = actual_labels.size(0)
    return 100 * correct / total
