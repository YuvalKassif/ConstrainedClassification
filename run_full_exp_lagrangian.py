import argparse
from datetime import datetime

import torch
import torch.optim as optim

from config import get_experiment_config
from utils import (
    set_seed,
    get_model,
    plot_training_results,
    count_predictions_per_class,
    save_test_counts,
    save_parameters,
)
from train import train_model, LRScheduler
from load_data import get_dataloaders
from losses import LagrangianConstraintLoss
from optimization import constrained_classification


def _compute_counts(loader, num_classes):
    label_counts = {}
    for _, labels in loader:
        if labels.dim() > 1:
            labels = labels.squeeze()
        for y in labels:
            yv = int(y.item())
            label_counts[yv] = label_counts.get(yv, 0) + 1
    return [label_counts.get(i, 0) for i in range(num_classes)]


def _select_device(params):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(params.get('gpu_index', 0))}")
    return torch.device("cpu")


def _maybe_override_dataset(params, dataset):
    if dataset is None:
        return
    params["dataset"] = dataset
    ds = params["dataset"].lower()
    if ds in (
        "medmnist_oct",
        "medmnist_tissue", "tissuemnist", "tissue",
        "medmnist_organ_c", "organ_cmnist", "organcmnist", "organ_c",
        "medmnist_organ_s", "organ_smnist", "organsmnist", "organ_s",
    ):
        params["model_choice"] = "medmnist"
    else:
        params["model_choice"] = "EfficientNetB5"


def run_lagrangian_for_class(
    params,
    exp_name,
    class_idx,
    train_loader,
    val_loader,
    test_loader,
    device,
    base_timestamp,
    constraints_percentage_list,
    num_classes,
    train_counts,
    test_counts,
    total_train,
):
    params["constrained_class_index"] = int(class_idx)
    lagrangian_lambda = float(params.get("lagrangian_lambda", 1.0))

    for percent in constraints_percentage_list:
        N_K_val = test_counts[class_idx] * percent / 100.0
        constraint_ratio = (
            (train_counts[class_idx] * percent / 100.0) / float(total_train)
            if total_train
            else 0.0
        )

        model = get_model(params["model_choice"], num_classes=num_classes).to(device)
        criterion = LagrangianConstraintLoss(
            constrained_class_index=class_idx,
            constraint_ratio=constraint_ratio,
            num_classes=num_classes,
            lam=lagrangian_lambda,
            device=device,
        )

        optimizer = optim.Adam(
            model.parameters(),
            lr=params["initial_learning_rate"],
            weight_decay=params["weight_decay"],
        )
        scheduler = LRScheduler(init_lr=params["initial_learning_rate"], lr_decay_epoch=params["decay_epoch"])

        iteration_timestamp = datetime.now().strftime("%d-%m-%H-%M")
        history, total_time, current_model = train_model(
            model,
            criterion,
            optimizer,
            scheduler,
            device,
            train_loader,
            val_loader,
            early_stopping_patience=params["patience"],
            timestamp=base_timestamp,
            iteration_timestamp=iteration_timestamp,
            num_epochs=params["epochs"],
        )

        class_count = count_predictions_per_class(current_model, test_loader, device)
        save_test_counts(
            class_count,
            exp_name=exp_name,
            timestamp=base_timestamp,
            iteration_timestamp=iteration_timestamp,
            dataset=params.get("dataset", "unknown_dataset"),
            constrained_class_index=class_idx,
        )

        plot_training_results(
            history,
            params["model_choice"],
            total_time,
            params,
            timestamp=base_timestamp,
            iteration_timestamp=iteration_timestamp,
        )

        lagrangian_results = constrained_classification(
            current_model,
            None,
            test_loader,
            class_idx,
            N_K_val,
            device,
            save_path="raw_results_lagrangian.csv",
        )

        results = {
            "Lagrangian_results": lagrangian_results,
            "counts": class_count,
            "constrained_class": class_idx,
            "N_K": N_K_val,
            "percent": percent,
        }

        save_parameters(
            results,
            exp_name,
            base_timestamp,
            iteration_timestamp,
            filename="results_lagrangian.json",
            dataset=params.get("dataset", "unknown_dataset"),
            constrained_class_index=class_idx,
        )


def main():
    parser = argparse.ArgumentParser(description="Run Lagrangian experiment for all classes in a dataset")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset override (else use config)")
    parser.add_argument("--constraints", type=int, nargs="+", default=[90, 70, 50, 30], help="Constraint percentages")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    seed = int(args.seed)
    set_seed(seed)

    params, exp_name = get_experiment_config()
    params["seed"] = seed
    _maybe_override_dataset(params, args.dataset)

    train_loader, val_loader, test_loader, data_meta = get_dataloaders(params)
    num_classes = data_meta.get("num_classes") or 4
    params["num_classes"] = num_classes

    device = _select_device(params)

    train_counts = _compute_counts(train_loader, num_classes)
    test_counts = _compute_counts(test_loader, num_classes)
    total_train = sum(train_counts)
    if total_train <= 0:
        total_train = len(train_loader.dataset)

    base_timestamp = datetime.now().strftime("%m-%d/experiment_files/%H-%M")

    for class_idx in range(num_classes):
        run_lagrangian_for_class(
            params=params,
            exp_name=exp_name,
            class_idx=class_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            base_timestamp=base_timestamp,
            constraints_percentage_list=args.constraints,
            num_classes=num_classes,
            train_counts=train_counts,
            test_counts=test_counts,
            total_train=total_train,
        )


if __name__ == "__main__":
    main()
