import os
import json
import matplotlib.pyplot as plt


# Function to read JSON file
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


# Function to traverse directories and collect results data
def collect_results_data(base_dir):
    results_data = []

    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            results_file = os.path.join(dir_path, 'results.json')

            if os.path.exists(results_file):
                results_data.append(read_json_file(results_file))

    return results_data


# Function to plot histograms for results data
def plot_histograms(results_data):
    num_files = len(results_data)
    fig, axes = plt.subplots(num_files, 2, figsize=(12, 5 * num_files))

    for idx, data in enumerate(results_data):
        pto = data["PTO_results"]
        pao = data["PAO_results"]
        constrained_class = data["constrained_class"]
        n_k = data["N_K"]

        metrics = ['accuracy', 'precision', 'recall', 'f1']

        pto_values = [pto[m] for m in metrics]
        pao_values = [pao[m] for m in metrics]

        # Plot PTO results
        axes[idx, 0].bar(metrics, pto_values, color='b', alpha=0.7)
        axes[idx, 0].set_title(f'PTO Results - Class: {constrained_class}, N_K: {n_k}')
        axes[idx, 0].set_ylim(0, 1)

        # Plot PAO results
        axes[idx, 1].bar(metrics, pao_values, color='r', alpha=0.7)
        axes[idx, 1].set_title(f'PAO Results - Class: {constrained_class}, N_K: {n_k}')
        axes[idx, 1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


# Main function
def main():
    base_dir = r"C:\Users\Yuval\PycharmProjects\hybrid-cost-sensitive-ml-optimization-master\savedModels1\08-06\experiment_files\08-17"
    results_data = collect_results_data(base_dir)
    plot_histograms(results_data)


if __name__ == "__main__":
    main()
