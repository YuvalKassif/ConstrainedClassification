from utils import save_parameters, generate_exp_name


def get_experiment_config(iteration=0, num_iterations=None):
    exp_name = generate_exp_name()

    # Choose one of: 'kneeKL224', 'medmnist_oct', 'medmnist_blood' (alias: 'bloodmnist'),
    # 'medmnist_derma' (aliases: 'dermamnist', 'derma'), 'medmnist_tissue' (aliases: 'tissuemnist', 'tissue'),
    # 'medmnist_organ_c' (aliases: 'organ_cmnist', 'organcmnist', 'organ_c'), 'lc25000'
    dataset = 'medmnist_organ_c'

    params = {
        "exp_name": exp_name,
        # Dataset selection and paths
        "dataset": dataset,
        # GPU selection (index). Set to 1 to prefer GPU 1.
        "gpu_index": 1,
        # Update these paths as needed in your environment
        "data_dir_knee": '/home/dsi/kassify/Research2019/ordinal_dnn/yuval/kneeKL224',
        "data_dir_lc25000": '/home/dsi/kassify/RetinalDataSet/lc25000',

        # Optimization and training hyperparameters
        "C_k": None,  # Will be set dynamically based on num_classes
        "initial_learning_rate": 0.0001, # 0.0001
        "dropout_rate": 0.0,
        "decay_epoch": 5,
        "decay_factor": 0.8,
        "patience": 5,
        "num_iterations": num_iterations,
        "num_iteration": iteration,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "epochs": 75,

        # Model choice: 'medmnist' (SimpleCNN) or any of
        # 'EfficientNetB0', 'EfficientNetB5', 'ResNet50', 'ResNet101'
        # Defaults:
        # - medmnist_oct, medmnist_tissue, medmnist_organ_c -> 'medmnist' (1-channel SimpleCNN)
        # - medmnist_blood/derma -> EfficientNet/ResNet (RGB)
        # - imagefolder datasets -> EfficientNet/ResNet
        "model_choice": ("medmnist" if dataset in ('medmnist_oct', 'medmnist_tissue', 'tissuemnist', 'tissue', 'medmnist_organ_c', 'organ_cmnist', 'organcmnist', 'organ_c') else "EfficientNetB5"),

        # Constraints configuration (index validated dynamically against num_classes)
        "constrained_class_index": 2,
        "constraints": 180,
        "mu": 8 / 600 # 13/600
    }
    return params, exp_name
