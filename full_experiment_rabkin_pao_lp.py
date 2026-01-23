import argparse
import json
from pathlib import Path
from datetime import datetime
import copy
import os
import subprocess

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
from losses import CustomLoss
from update_weights import calculate_F_and_derivative
from optimization import constrained_classification, constrained_classification_or
from optimization_analysis import process_results


def _select_smart_device():
    """Select the most 'free' CUDA device when available."""
    if not torch.cuda.is_available():
        return torch.device("cpu")

    gpu_env = os.environ.get("GPU_INDEX")
    # gpu_env = 2
    if gpu_env is not None:
        try:
            idx = int(gpu_env)
            torch.cuda.set_device(idx)
            print(f"[INFO] Using GPU from GPU_INDEX env: cuda:{idx}")
            return torch.device(f"cuda:{idx}")
        except Exception as e:
            print(f"[WARN] Invalid GPU_INDEX='{gpu_env}': {e}. Ignoring.")

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
            stderr=subprocess.STDOUT,
        )
        rows = []
        for line in output.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            idx, mem_free, mem_total, util = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            rows.append({"idx": idx, "free": mem_free, "total": mem_total, "util": util})

        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        allowed = None
        if visible:
            try:
                allowed = [int(t.strip()) for t in visible.split(",") if t.strip() != ""]
                rows = [r for r in rows if r["idx"] in allowed]
            except Exception:
                allowed = None

        if rows:
            rows.sort(key=lambda r: (r["util"], -r["free"]))
            best_physical = rows[0]["idx"]
            if allowed is not None and best_physical in allowed:
                best_logical = allowed.index(best_physical)
            else:
                best_logical = best_physical
            torch.cuda.set_device(best_logical)
            try:
                name = torch.cuda.get_device_name(best_logical)
                print(f"[INFO] Auto-selected GPU cuda:{best_logical} ({name}) via nvidia-smi")
            except Exception:
                print(f"[INFO] Auto-selected GPU cuda:{best_logical} via nvidia-smi")
            return torch.device(f"cuda:{best_logical}")
    except Exception as e:
        print(f"[WARN] nvidia-smi query failed: {e}. Falling back to torch API.")

    try:
        best_idx, best_free = 0, -1
        for i in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(i):
                    free, total = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
                if free > best_free:
                    best_idx, best_free = i, free
            except Exception:
                pass
        torch.cuda.set_device(best_idx)
        print(f"[INFO] Auto-selected GPU cuda:{best_idx} via torch CUDA API")
        return torch.device(f"cuda:{best_idx}")
    except Exception:
        print("[INFO] Falling back to default CUDA device")
        return torch.device("cuda")


def compute_test_counts(loader, num_classes):
    label_counts = {}
    for _, labels in loader:
        if labels.dim() > 1:
            labels = labels.squeeze()
        for y in labels:
            yv = int(y.item())
            label_counts[yv] = label_counts.get(yv, 0) + 1
    return [label_counts.get(i, 0) for i in range(num_classes)]


def grid_search_hparams(
    params,
    train_loader,
    val_loader,
    device,
    lr_grid,
    epoch_grid,
    base_timestamp,
):
    """Run a simple grid search over (initial_learning_rate, epochs)."""
    results = []
    best = None
    best_key = None

    constrained_idx = int(params.get("constrained_class_index", 0))

    for lr in lr_grid:
        for epochs in epoch_grid:
            model = get_model(params["model_choice"], num_classes=params.get("num_classes", 4)).to(device)
            C = torch.ones(params.get("num_classes", 4), device=device)
            criterion = CustomLoss(constrained_class_index=constrained_idx, C=C, device=device)
            optimizer = optim.Adam(model.parameters(), lr=float(lr), weight_decay=params["weight_decay"])
            scheduler = LRScheduler(init_lr=float(lr), lr_decay_epoch=params["decay_epoch"])

            iteration_timestamp = datetime.now().strftime("%d-%m-%H-%M")
            history, total_time, best_model = train_model(
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
                num_epochs=int(epochs),
            )

            try:
                val_acc = max(history.get("val_acc", []) or [0.0])
            except Exception:
                val_acc = 0.0

            record = {
                "lr": float(lr),
                "epochs": int(epochs),
                "val_acc_max": float(val_acc),
                "train_time_sec": float(total_time),
                "iteration_timestamp": iteration_timestamp,
            }
            results.append(record)

            if best is None or val_acc > best.get("val_acc_max", -1):
                best = record
                best_key = (float(lr), int(epochs))

    return best_key, best, results


def run_constraints_for_class(
    base_params,
    class_idx,
    train_loader,
    val_loader,
    test_loader,
    device,
    base_timestamp,
    constraints_percentage_list,
    num_classes,
):
    """Run constrained experiment for a single constrained class, incl. LP benchmarks."""
    params = copy.deepcopy(base_params)
    params["constrained_class_index"] = int(class_idx)

    first_model = None
    true_test_counts = compute_test_counts(test_loader, num_classes=num_classes)

    for percent in constraints_percentage_list:
        N_K_val = true_test_counts[class_idx] * percent / 100.0
        C = torch.ones(num_classes, device=device)

        number_of_iterations = 0
        while True:
            number_of_iterations += 1

            model = get_model(params["model_choice"], num_classes=num_classes).to(device)
            criterion = CustomLoss(constrained_class_index=class_idx, C=C, device=device)

            current_lr = params["initial_learning_rate"]
            optimizer = optim.Adam(model.parameters(), lr=current_lr, weight_decay=params["weight_decay"])

            iteration_timestamp = datetime.now().strftime("%d-%m-%H-%M")

            params["C_k"] = C.tolist()
            scheduler = LRScheduler(init_lr=params["initial_learning_rate"], lr_decay_epoch=params["decay_epoch"])

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

            if number_of_iterations == 1:
                first_model = current_model

            class_count = count_predictions_per_class(current_model, test_loader, device)
            # Ensure all classes exist to avoid KeyError when a class is never predicted.
            class_count = {i: class_count.get(i, 0) for i in range(num_classes)}
            save_test_counts(
                class_count,
                exp_name=params["exp_name"],
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

            N_K_p_val = class_count[class_idx]
            b_val = 100.0

            params["constraints"] = N_K_val
            params["constraints_percent"] = percent
            # save
            save_parameters(
                params,
                exp_name=params["exp_name"],
                timestamp=base_timestamp,
                iteration_timestamp=iteration_timestamp,
                dataset=params.get("dataset", "unknown_dataset"),
                constrained_class_index=class_idx,
            )

            F_val, dF_dN_K_p_val = calculate_F_and_derivative(N_K_val, N_K_p_val, b_val)

            pto_results = constrained_classification(
                first_model,
                None,
                test_loader,
                class_idx,
                N_K_val,
                device,
                save_path="raw_results_PTO.csv",
            )
            pto_lp_results = constrained_classification_or(
                first_model,
                None,
                test_loader,
                class_idx,
                N_K_val,
                device,
                save_path="raw_results_PTO_LP.csv",
            )
            pao_results = constrained_classification(
                current_model,
                None,
                test_loader,
                class_idx,
                N_K_val,
                device,
                save_path="raw_results_PAO.csv",
            )
            pao_lp_results = constrained_classification_or(
                current_model,
                None,
                test_loader,
                class_idx,
                N_K_val,
                device,
                save_path="raw_results_PAO_LP.csv",
            )

            results = {
                "PTO_results": pto_results,
                "PTO_LP_results": pto_lp_results,
                "PAO_results": pao_results,
                "PAO_LP_results": pao_lp_results,
                "counts": class_count,
                "constrained_class": class_idx,
                "N_K": N_K_val,
                "percent": percent,
            }

            save_parameters(
                results,
                params["exp_name"],
                base_timestamp,
                iteration_timestamp,
                filename="results.json",
                dataset=params.get("dataset", "unknown_dataset"),
                constrained_class_index=class_idx,
            )

            if F_val < 1e-5:
                break

            mu = params["mu"]
            C = C + mu * dF_dN_K_p_val
            C[class_idx] = 1

        process_results(
            base_dir=Path(f"results/{params.get('dataset', 'unknown_dataset')}/{class_idx}/{base_timestamp}")
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run full experiment with PTO/PAO and LP variants (PTO_LP, PAO_LP)"
    )
    parser.add_argument("--dataset", type=str, default=None, help="Dataset override (else use config)")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate override")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs override")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience override")
    parser.add_argument("--constraints", type=int, nargs="+", default=[90, 70, 50, 30], help="Constraint percentages")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    seed = int(args.seed)
    set_seed(seed)

    params, exp_name = get_experiment_config()
    params["seed"] = seed
    if args.dataset is not None:
        params["dataset"] = args.dataset
        ds = params["dataset"].lower()
        if ds in (
            "medmnist_oct",
            "medmnist_tissue", "tissuemnist", "tissue",
            "medmnist_organ_c", "organ_cmnist", "organcmnist", "organ_c",
            "medmnist_organ_s", "organ_smnist", "organsmnist", "organ_s",
        ):
            params["model_choice"] = "medmnist"
        elif ds in ("medmnist_path", "pathmnist", "path"):
            # PathMNIST is RGB; use 3-channel model.
            params["model_choice"] = "EfficientNetB5"
        else:
            params["model_choice"] = "EfficientNetB5"

    train_loader, val_loader, test_loader, meta = get_dataloaders(params)
    num_classes = meta.get("num_classes") or 4
    params["num_classes"] = num_classes

    device = _select_smart_device()

    base_timestamp = datetime.now().strftime("%m-%d/experiment_files/%H-%M")

    if args.lr is not None:
        params["initial_learning_rate"] = float(args.lr)
    if args.epochs is not None:
        params["epochs"] = int(args.epochs)
    if args.patience is not None:
        params["patience"] = int(args.patience)

    for class_idx in range(num_classes):
        run_constraints_for_class(
            base_params=params,
            class_idx=class_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            base_timestamp=base_timestamp,
            constraints_percentage_list=args.constraints,
            num_classes=num_classes,
        )

    manifest = {
        "dataset": params.get("dataset"),
        "base_timestamp": base_timestamp,
        "num_classes": num_classes,
        "hyperparams": {"lr": params["initial_learning_rate"], "epochs": params["epochs"]},
        "patience": params["patience"],
        "constraints": args.constraints,
        "variants": ["PTO", "PTO_LP", "PAO", "PAO_LP"],
    }
    manifest_dir = Path(f"results/{params.get('dataset', 'unknown_dataset')}/hyperparam_search/{base_timestamp}")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_dir / "full_experiment_manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)


if __name__ == "__main__":
    main()
