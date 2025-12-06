import os
import json
import statistics
import matplotlib.pyplot as plt


def find_experiment_json_files(base_dir):
    all_experiments = []

    # Walk through the directory structure
    for root, dirs, files in os.walk(base_dir):
        # Check if both 'results.json' and 'experiment_config.json' exist in the current folder
        if 'results.json' in files and 'experiment_config.json' in files:
            results_path = os.path.join(root, 'results.json')
            config_path = os.path.join(root, 'experiment_config.json')

            # Read and parse the JSON files
            with open(results_path, 'r') as results_file:
                results_data = json.load(results_file)
            with open(config_path, 'r') as config_file:
                config_data = json.load(config_file)

            # Append the parsed data to the list of experiments
            all_experiments.append({
                "results": results_data,
                "config": config_data
            })

    return all_experiments


# Example usage:
# base_directory = r"savedModels1/09-07/experiment_files/16-37"
base_directory = r"savedModels1"
all_experiments = find_experiment_json_files(base_directory)

# print(all_experiments[0]["results"]["PTO_results"]["accuracy"])

good_ones = []
C_k = []
C_k_mean = []
constrained_class = []
constraint_percent = []
constrained_diff = []

for i in range(len(all_experiments)):
    if all_experiments[i]["config"]["constrained_class_index"] == 2 or all_experiments[i]["config"]["constrained_class_index"] == 2:
        if all_experiments[i]["results"]["PTO_results"]["accuracy"] < all_experiments[i]["results"]["PAO_results"]["accuracy"]:
            constrained_class.append(all_experiments[i]["config"]["constrained_class_index"])

            constrained_diff.append(round(all_experiments[i]["results"]["counts"][str(constrained_class[-1])] - all_experiments[i]["results"]["N_K"]))

            if abs(constrained_diff[-1]) < 20:
                good_ones.append(all_experiments[i])
                C_k.append(all_experiments[i]["config"]["C_k"])
                C_k_mean.append(statistics.mean(all_experiments[i]["config"]["C_k"]))
                constraint_percent.append(all_experiments[i]["config"]["constraints_percent"])

# print(good_ones)
# print(C_k_mean)

# Sorting the data by C_k_mean in ascending order
sorted_data = sorted(zip(C_k_mean, constrained_class, constraint_percent, constrained_diff), key=lambda x: x[0])
C_k_mean_sorted, constrained_class_sorted, constraint_percent_sorted, constrained_diff_sorted = zip(*sorted_data)


# Create a scatter plot to visualize the sorted C_k_mean list with annotations for constrained_class, constraint_percent, and constrained_diff
plt.figure(figsize=(8, 6))
plt.scatter(range(len(C_k_mean_sorted)), C_k_mean_sorted, color='b')

# Annotate each point with constrained_class, constraint_percent, and constrained_diff
for idx, (mean_value, class_index, percent, diff) in enumerate(zip(C_k_mean_sorted, constrained_class_sorted, constraint_percent_sorted, constrained_diff_sorted)):
    annotation_text = f"Class: {class_index}\n%: {percent}\nDiff: {diff}"
    plt.annotate(annotation_text, (idx, mean_value), textcoords="offset points", xytext=(0,10), ha='center', fontsize=6)

plt.title('C_k Mean Values Over Selected Experiments (Sorted)')
plt.xlabel('Experiment Index (Sorted by C_k Mean)')
plt.ylabel('C_k Mean Value')
plt.grid(True)

# Show the scatter plot
plt.show()