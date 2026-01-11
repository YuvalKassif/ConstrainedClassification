from utils import save_parameters, generate_exp_name


def get_experiment_config(iteration=0, num_iterations=None):
    exp_name = generate_exp_name()

    # Choose one of: 'kneeKL224', 'medmnist_oct', 'medmnist_blood' (alias: 'bloodmnist'),
    # 'medmnist_derma' (aliases: 'dermamnist', 'derma'), 'medmnist_tissue' (aliases: 'tissuemnist', 'tissue'),
    # 'medmnist_organ_c' (aliases: 'organ_cmnist', 'organcmnist', 'organ_c'),
    # 'medmnist_organ_s' (aliases: 'organ_smnist', 'organsmnist', 'organ_s'), 'lc25000',
    # 'ham10000' (aliases: 'ham', 'isic_ham'), 'breakhis' (breast histology)
    dataset = 'medmnist_derma'

    params = {
        "exp_name": exp_name,
        # Dataset selection and paths
        "dataset": dataset,
        # GPU selection (index). Set to 1 to prefer GPU 1.
        "gpu_index": 1,
        # Update these paths as needed in your environment
        "data_dir_knee": '/home/dsi/kassify/Research2019/ordinal_dnn/yuval/kneeKL224',
        "data_dir_lc25000": '/home/dsi/kassify/RetinalDataSet/lc25000',
        "data_dir_ham10000": '/home/dsi/kassify/Research2019/datasets/HAM10000',  # Set to the folder containing HAM10000 images/metadata
        "data_dir_breakhis": None,  # Set to the BreaKHis dataset root

        # Optional backend controls
        "disable_cudnn": True,  # Set True to work around rare cuDNN internal errors
        "data_dir_breakhis": '/home/dsi/kassify/Research2019/datasets/BreaKHis',  # Set to the BreaKHis dataset root
        # Optional BreaKHis controls
        "breakhis_granularity": 'subtype',  # 'binary' or 'subtype'
        "breakhis_magnifications": None,   # e.g., [40, 100, 200, 400] or None for all

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
        "batch_size": 16,
        "epochs": 75,

        # Model choice: 'medmnist' (SimpleCNN) or any of
        # 'EfficientNetB0', 'EfficientNetB5', 'ResNet50', 'ResNet101'
        # Defaults:
        # - medmnist_oct, medmnist_tissue, medmnist_organ_c, medmnist_organ_s -> 'medmnist' (1-channel SimpleCNN)
        # - medmnist_blood/derma -> EfficientNet/ResNet (RGB)
        # - imagefolder datasets -> EfficientNet/ResNet
        "model_choice": ("medmnist" if dataset in (
            'medmnist_oct',
            'medmnist_tissue', 'tissuemnist', 'tissue',
            'medmnist_organ_c', 'organ_cmnist', 'organcmnist', 'organ_c',
            'medmnist_organ_s', 'organ_smnist', 'organsmnist', 'organ_s'
        ) else "EfficientNetB5"),

        # Constraints configuration (index validated dynamically against num_classes)
        "constrained_class_index": 2,
        "constraints": 180,
        "mu": 8 / 600 # 13/600
    }
    return params, exp_name
