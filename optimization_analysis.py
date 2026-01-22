import os
import json
import matplotlib.pyplot as plt
import numpy as np


# Function to read JSON file
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


# Function to parse time from directory name
def parse_time_from_dir_name(dir_name):
    day, month, hh, mm = map(int, dir_name.split('-'))
    # Convert the time to a comparable value (e.g., minutes since the start of the month)
    # Assuming 31 days max for simplicity in the computation.
    return ((day - 1) * 24 * 60) + (hh * 60) + mm



# Function to traverse directories and collect results data
def collect_results_data(base_dir):
    results_data = []
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            results_file = os.path.join(dir_path, 'results.json')

            if os.path.exists(results_file):
                data = read_json_file(results_file)
                try:
                    # Parse time from the directory name using the updated format
                    data['time'] = parse_time_from_dir_name(dir_name)
                except ValueError:
                    # Handle the case where the directory name does not match the expected format
                    print(f"Skipping directory with unexpected name format: {dir_name}")
                    continue
                results_data.append(data)

    # Sort results by time in reverse order
    results_data.sort(key=lambda x: x['time'], reverse=True)
    return results_data

# Function to plot and save line graphs for results data
def plot_and_save_graphs1(results_data, save_path):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 5 * num_metrics))

    if num_metrics == 1:
        axes = [axes]  # Ensure axes is always iterable

    for metric_idx, metric in enumerate(metrics):
        pto_values = []
        pao_values = []
        n_k_values = []
        constrained_class = None

        for data in results_data:
            pto_values.append(data["PTO_results"][metric])
            pao_values.append(data["PAO_results"][metric])
            n_k_values.append(data["N_K"])
            if constrained_class is None:
                constrained_class = data["constrained_class"]

        x = np.arange(len(pto_values))

        axes[metric_idx].plot(x, pto_values, marker='o', label='PTO', color='b', alpha=0.7)
        axes[metric_idx].plot(x, pao_values, marker='o', label='PAO', color='r', alpha=0.7)

        # Add text for labels, title and custom x-axis tick labels, etc.
        axes[metric_idx].set_xlabel('Experiments')
        axes[metric_idx].set_ylabel(metric.capitalize())
        axes[metric_idx].set_title(f'{metric.capitalize()} - Class: {constrained_class}')
        axes[metric_idx].set_xticks(x)
        axes[metric_idx].set_xticklabels([])  # No units on x-axis
        axes[metric_idx].legend()

        # Label the points with their N_K values
        for i, n_k in enumerate(n_k_values):
            axes[metric_idx].text(x[i], pto_values[i], str(round(n_k, 2)), ha='center', va='bottom', fontsize=8,
                                  color='blue')
            axes[metric_idx].text(x[i], pao_values[i], str(round(n_k, 2)), ha='center', va='bottom', fontsize=8,
                                  color='red')

    plt.tight_layout()

    # Save the plot to the provided path
    plt.savefig(save_path)
    plt.close()


def plot_and_save_graphs(results_data, save_path):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    variant_defs = [
        ("PTO", "PTO_results", "blue"),
        ("PTO_LP", "PTO_LP_results", "green"),
        ("PAO", "PAO_results", "red"),
        ("PAO_LP", "PAO_LP_results", "orange"),
    ]

    if not results_data:
        return

    available_variants = []
    for label, key, color in variant_defs:
        if all(key in data for data in results_data):
            available_variants.append((label, key, color))

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))

        n_k_values = [data["N_K"] for data in results_data]
        percent_values = [data["percent"] for data in results_data]
        constrained_class = results_data[0]["constrained_class"] if results_data else None

        x = np.arange(len(results_data))  # Experiment indices

        for label, key, color in available_variants:
            if metric == 'accuracy':
                values = [data[key]["accuracy"] for data in results_data]
                ax.plot(x, values, marker='o', label=f'{label} Accuracy', color=color, alpha=0.7)
            else:
                macro_values = [data[key]["macro"][metric] for data in results_data]
                weighted_values = [data[key]["weighted"][metric] for data in results_data]
                ax.plot(
                    x,
                    macro_values,
                    marker='o',
                    label=f'{label} Macro {metric.capitalize()}',
                    color=color,
                    alpha=0.7,
                )
                ax.plot(
                    x,
                    weighted_values,
                    marker='o',
                    label=f'{label} Weighted {metric.capitalize()}',
                    linestyle='--',
                    color=color,
                    alpha=0.7,
                )

        # Add text for labels, title and custom x-axis tick labels
        ax.set_xlabel('Experiments')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Analysis - Class: {constrained_class}')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{percent}%" for percent in percent_values], rotation=45)
        ax.legend()

        # Annotate points with N_K values using the first available variant to avoid clutter
        if available_variants:
            label, key, color = available_variants[0]
            if metric == 'accuracy':
                values = [data[key]["accuracy"] for data in results_data]
                for i, n_k in enumerate(n_k_values):
                    ax.text(x[i], values[i], f'{round(n_k, 2)}', ha='center', va='bottom', fontsize=8, color=color)
            else:
                values = [data[key]["macro"][metric] for data in results_data]
                for i, n_k in enumerate(n_k_values):
                    ax.text(x[i], values[i], f'{round(n_k, 2)}', ha='center', va='bottom', fontsize=8, color=color)

        ax.grid(True)
        plt.tight_layout()

        # Save the individual plot
        metric_save_path = os.path.join(save_path, f"{metric}_analysis.png")
        plt.savefig(metric_save_path)
        plt.close()
        print(f"Saved {metric.capitalize()} Analysis Plot to {metric_save_path}")


# Function to process results and save the plot in the base directory
def process_results1(base_dir):
    results_data = collect_results_data(base_dir)

    # Define the path to save the plot
    save_path = os.path.join(base_dir, "optimization_analysis.png")

    # Plot and save the graphs
    plot_and_save_graphs(results_data, save_path)

    print(f"Analysis Plot saved to {save_path}")


def process_results(base_dir):
    results_data = collect_results_data(base_dir)

    # Create a directory to save plots
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot and save graphs
    plot_and_save_graphs(results_data, plots_dir)

    print(f"All analysis plots saved to {plots_dir}")
