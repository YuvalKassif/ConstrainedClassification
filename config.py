from utils import save_parameters, generate_exp_name


def get_experiment_config(iteration=0, num_iterations=None):
    exp_name = generate_exp_name()

    # Choose one of: 'kneeKL224', 'medmnist_oct', 'medmnist_blood' (alias: 'bloodmnist'), 'lc25000'
    dataset = 'medmnist_blood'

    params = {
        "exp_name": exp_name,
        # Dataset selection and paths
        "dataset": dataset,
        # Update these paths as needed in your environment
        "data_dir_knee": '/home/dsi/kassify/Research2019/ordinal_dnn/yuval/kneeKL224',
        "data_dir_lc25000": '/home/dsi/kassify/RetinalDataSet/lc25000',

        # Optimization and training hyperparameters
        "C_k": None,  # Will be set dynamically based on num_classes
        "initial_learning_rate": 0.0001,
        "dropout_rate": 0.0,
        "decay_epoch": 5,
        "decay_factor": 0.8,
        "patience": 3,
        "num_iterations": num_iterations,
        "num_iteration": iteration,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "epochs": 25,

        # Model choice: 'medmnist' (SimpleCNN) or any of
        # 'EfficientNetB0', 'EfficientNetB5', 'ResNet50', 'ResNet101'
        # Defaults:
        # - medmnist_oct -> 'medmnist' (1-channel SimpleCNN)
        # - medmnist_blood -> 'ResNet50' (RGB). SimpleCNN expects 1 channel.
        # - imagefolder datasets -> 'ResNet50'
        "model_choice": ("medmnist" if dataset == 'medmnist_oct' else "EfficientNetB0"),

        # Constraints configuration (index validated dynamically against num_classes)
        "constrained_class_index": 1,
        "constraints": 180,
        "mu": 40 / 600 # 13/600
    }
    return params, exp_name
