import torch
import torch.nn as nn
import torch.optim as optim
from utils import set_seed, get_model, evaluate_test_accuracy, plot_training_results, count_predictions_per_class, save_test_counts, save_parameters
from train import train_model, LRScheduler
from load_data import get_dataloaders, get_weighted_sampler
from config import get_experiment_config
from losses import CustomLoss
from datetime import datetime
from update_weights import calculate_F_and_derivative
from optimization import constrained_classification
from pathlib import Path
from optimization_analysis import process_results
from torch.utils.data import DataLoader

seed = 42
set_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get experiment configuration
params, exp_name = get_experiment_config()

# Build dataloaders based on dataset selection
train_loader, val_loader, test_loader, data_meta = get_dataloaders(params)

# Dynamically determine number of classes
num_classes = data_meta.get('num_classes') or 4
print(f"[DEBUG] Dataset '{params.get('dataset')}' -> num_classes={num_classes}, classes={data_meta.get('class_names')}")

# print("got num classes:", data_meta.get('num_classes'))

# Initialize cost vector C per number of classes
if not params.get("C_k"):
    C = torch.ones(num_classes)
else:
    C_vals = params["C_k"]
    if len(C_vals) != num_classes:
        C = torch.ones(num_classes)
    else:
        C = torch.tensor(C_vals)

constrained_class_index = params["constrained_class_index"]
if constrained_class_index >= num_classes:
    constrained_class_index = num_classes - 1
    params["constrained_class_index"] = constrained_class_index
early_stopping_patience = params["patience"]

# Create a timestamp for the experiment folder, only once
base_timestamp = datetime.now().strftime("%m-%d/experiment_files/%H-%M")
N_K_val = None

# Compute true test counts dynamically from the test set
def _compute_test_counts(loader):
    label_counts = {}
    for _, labels in loader:
        if labels.dim() > 1:
            labels = labels.squeeze()
        for y in labels:
            yv = int(y.item())
            label_counts[yv] = label_counts.get(yv, 0) + 1
    # Fill missing classes with 0 up to num_classes
    return [label_counts.get(i, 0) for i in range(num_classes)]

true_test_counts = _compute_test_counts(test_loader)
print(f"[DEBUG] Test label counts (by class index): {true_test_counts}")
# constraints_percentage_list = [50]
constraints_percentage_list = [90, 70, 50, 30]
# constraints_percentage_list = [80,60, 40, 20]

# Placeholder for storing the first model
first_model = None

# Iterate over constraint percentages
for percent in constraints_percentage_list:
    print("#------ Constraints Percentage:", percent, "------#")
    N_K_val = true_test_counts[constrained_class_index] * percent / 100
    print("#------ N_K:", N_K_val, "------#")
    C = torch.ones(num_classes)
    print(C)

    constraints_percent = percent
    number_of_iterations = 0

    # Iterate until constraints are met
    while True:
        number_of_iterations += 1
        print("------- Iteration", number_of_iterations, "-------")
        print(C)
        print("Contstraint index:", constrained_class_index)
        model_choice = params["model_choice"]
        set_seed(seed)
        model = get_model(model_choice, num_classes=num_classes).to(device)
        criterion = CustomLoss(constrained_class_index=constrained_class_index, C=C, device=device)

        # Adjust learning rate
        current_lr = params["initial_learning_rate"] / torch.mean(C)
        optimizer = optim.Adam(model.parameters(), lr=current_lr, weight_decay=params["weight_decay"])

        if number_of_iterations == 1:
            first_model_timestamp = datetime.now().strftime("%d-%m-%H-%M")
        # Create a timestamp for the current iteration
        iteration_timestamp = datetime.now().strftime("%d-%m-%H-%M")

        # Save parameters with the same base timestamp folder but unique iteration timestamp folder
        params["C_k"] = C.tolist()  # Convert tensor to list for saving

        scheduler = LRScheduler(init_lr=params["initial_learning_rate"], lr_decay_epoch=params["decay_epoch"])

        # Train the model (inside the loop)
        history, total_time, current_model = train_model(model, criterion, optimizer, scheduler, device, train_loader,
                                                         val_loader, early_stopping_patience,
                                                         timestamp=base_timestamp,
                                                         iteration_timestamp=iteration_timestamp,
                                                         num_epochs=params["epochs"])

        # Save the first model (PTO should use this model)
        if number_of_iterations == 1:
            first_model = current_model

        # Evaluate on the test set for both PTO and PAO
        class_count = count_predictions_per_class(current_model, test_loader, device)
        save_test_counts(class_count, exp_name=params["exp_name"], timestamp=base_timestamp, iteration_timestamp=iteration_timestamp)
        print(class_count)

        # Plot the results
        plot_training_results(history, model_choice, total_time, params, timestamp=base_timestamp, iteration_timestamp=iteration_timestamp)

        N_K_p_val = class_count[constrained_class_index]
        b_val = 100.0

        params["constraints"] = N_K_val
        params["constraints_percent"] = constraints_percent

        save_parameters(params, exp_name=exp_name, timestamp=base_timestamp, iteration_timestamp=iteration_timestamp)

        F_val, dF_dN_K_p_val = calculate_F_and_derivative(N_K_val, N_K_p_val, b_val)



        # Predict Then Optimize (PTO)
        pto_results = constrained_classification(
            first_model, None, test_loader, constrained_class_index, N_K_val, device, save_path='raw_results_PTO.csv'
        )
        print("PTO Results:", pto_results)

        # Predict And Optimize (PAO)
        pao_results = constrained_classification(
            current_model, None, test_loader, constrained_class_index, N_K_val, device, save_path='raw_results_PAO.csv'
        )
        print("PAO Results:", pao_results)

        # Store results
        results = {
            "PTO_results": pto_results,
            "PAO_results": pao_results,
            "counts": class_count,
            "constrained_class": constrained_class_index,
            "N_K": N_K_val,
            "percent": percent,
        }


        print("results:\n", results)
        # Save results to a file
        save_parameters(results, exp_name, base_timestamp, iteration_timestamp, filename="results.json")

        # Check if constraints are met
        if F_val < 1e-5:
            break

        # Update C using optimization
        # mu = 8 / 600  # Learning rate for optimization
        mu = params["mu"]
        C += mu * dF_dN_K_p_val
        C[params["constrained_class_index"]] = 1
        params["C_k"] = C

    print('Achieved constraints with', number_of_iterations, 'iterations')
    process_results(base_dir=Path(f'savedModels1/{base_timestamp}'))

process_results(base_dir=Path(f'savedModels1/{base_timestamp}'))
