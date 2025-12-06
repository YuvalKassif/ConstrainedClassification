import os
import json
import matplotlib.pyplot as plt
import numpy as np


def plot_class_weights_vs_counts(base_dir):
    no_constraints_idx = 0
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    all_class_counts = []
    all_class_weights = []
    non_constraints_class_weights = []

    class_names = set()

    for subdir in subdirs:
        class_count_path = os.path.join(base_dir, subdir, 'class_count.json')
        experiment_config_path = os.path.join(base_dir, subdir, 'experiment_config.json')

        with open(class_count_path, 'r') as cc_file:
            class_counts = json.load(cc_file)

        with open(experiment_config_path, 'r') as ec_file:
            experiment_config = json.load(ec_file)

        class_weights = experiment_config.get('C_k', [])

        all_class_counts.append(class_counts)
        all_class_weights.append(class_weights)
        class_names.update(class_counts.keys())

    class_names = sorted(class_names, key=int)  # Sort class names numerically


    fig, axes = plt.subplots(len(class_names), 1, figsize=(10, 5 * len(class_names)))
    if len(class_names) == 1:
        axes = [axes]  # Ensure axes is iterable if only one subplot

    for idx, class_name in enumerate(class_names[::-1]):
        class_counts_list = []
        class_weights_list = []

        for class_counts, class_weights in zip(all_class_counts, all_class_weights):
            count = class_counts.get(class_name, 0)
            weight = class_weights[int(class_name)] if int(class_name) < len(class_weights) else 0

            class_counts_list.append(count)
            class_weights_list.append(weight)
            if idx == no_constraints_idx:
                non_constraints_class_weights.append(weight)

        # Scatter plot
        axes[idx].scatter(non_constraints_class_weights, class_counts_list)

        # Linear regression
        if len(non_constraints_class_weights) > 1:
            slope, intercept = np.polyfit(non_constraints_class_weights, class_counts_list, 1)
            regression_line = np.poly1d([slope, intercept])
            line_x = np.linspace(min(non_constraints_class_weights), max(non_constraints_class_weights), 100)
            line_y = regression_line(line_x)
            axes[idx].plot(line_x, line_y, color='red', linestyle='--')

            # Annotate the slope
            axes[idx].annotate(f'Slope: {slope:.2f}', xy=(0.7, 0.9), xycoords='axes fraction', fontsize=12, color='red')

        # Set y-axis limits
        max_count = max(class_counts_list)
        axes[idx].set_ylim(0, max_count * 1.2)

        axes[idx].set_title(f'Class {class_name}')
        axes[idx].set_xlabel('Weight')
        axes[idx].set_ylabel('Count')

    plt.tight_layout()
    plt.show()


# Example usage
base_dir = 'savedModels1/05-25/experiment_files/18-32'
plot_class_weights_vs_counts(base_dir)
