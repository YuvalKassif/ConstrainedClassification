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
from optimization import constrained_classification
from optimization_analysis import process_results


def _select_smart_device():
    """Select the most 'free' CUDA device when available.

    Priority:
    1) Respect `GPU_INDEX` env var if set and valid.
    2) Use `nvidia-smi` to pick GPU with lowest utilization, then most free mem.
    3) Fallback to largest free memory via torch API if available.
    4) Fallback to default CUDA or CPU.
    """
    # If CUDA not available, use CPU
    if not torch.cuda.is_available():
        return torch.device('cpu')

    # 1) Allow explicit override via env var
    gpu_env = os.environ.get('GPU_INDEX')
    if gpu_env is not None:
        try:
            idx = int(gpu_env)
            torch.cuda.set_device(idx)
            print(f"[INFO] Using GPU from GPU_INDEX env: cuda:{idx}")
            return torch.device(f'cuda:{idx}')
        except Exception as e:
            print(f"[WARN] Invalid GPU_INDEX='{gpu_env}': {e}. Ignoring.")

    # 2) Try to query via nvidia-smi
    try:
        output = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=index,memory.free,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits',
            ],
            encoding='utf-8',
            stderr=subprocess.STDOUT,
        )
        rows = []
        for line in output.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 4:
                continue
            idx, mem_free, mem_total, util = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            rows.append({'idx': idx, 'free': mem_free, 'total': mem_total, 'util': util})

        # Respect CUDA_VISIBLE_DEVICES if set
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        allowed = None
        if visible:
            try:
                allowed = [int(t.strip()) for t in visible.split(',') if t.strip() != '']
                rows = [r for r in rows if r['idx'] in allowed]
            except Exception:
                # If parsing fails, keep rows as is
                allowed = None

        if rows:
            # Prefer lowest utilization, then highest free memory
            rows.sort(key=lambda r: (r['util'], -r['free']))
            best_physical = rows[0]['idx']
            # Map to logical index when CUDA_VISIBLE_DEVICES is set
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
            return torch.device(f'cuda:{best_logical}')
    except Exception as e:
        # nvidia-smi not found or failed; continue to fallback
        print(f"[WARN] nvidia-smi query failed: {e}. Falling back to torch API.")

    # 3) Fallback: pick device with largest free memory via torch (if supported)
    try:
        best_idx, best_free = 0, -1
        for i in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(i):
                    # mem_get_info returns (free, total)
                    free, total = torch.cuda.mem_get_info()  # type: ignore[attr-defined]
                if free > best_free:
                    best_idx, best_free = i, free
            except Exception:
                # If mem_get_info not available for this device, skip
                pass
        torch.cuda.set_device(best_idx)
        print(f"[INFO] Auto-selected GPU cuda:{best_idx} via torch CUDA API")
        return torch.device(f'cuda:{best_idx}')
    except Exception:
        # 4) Final fallback: default CUDA device
        print("[INFO] Falling back to default CUDA device")
        return torch.device('cuda')

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
    """Run a simple grid search over (initial_learning_rate, epochs).

    For each combo, train once with C=1 (no iterative constraint updates) and record
    best validation accuracy. Returns best (lr, epochs) and a results list.
    """
    results = []
    best = None
    best_key = None

    # Use a fixed constrained class (from params) but C=1 to make this a general fit
    constrained_idx = int(params.get("constrained_class_index", 0))

    for lr in lr_grid:
        for epochs in epoch_grid:
            # Fresh model and loss per combo
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

            # Track max validation accuracy observed during training
            try:
                val_acc = max(history.get("val_acc", []) or [0.0])
            except Exception:
                val_acc = 0.0

            # Save a compact record
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
    """Run the existing constrained experiment for a single constrained class.

    Mirrors the core logic in main.py but parameterized for reuse.
    """
    params = copy.deepcopy(base_params)
    params["constrained_class_index"] = int(class_idx)

    # Placeholder for storing the first model (PTO)
    first_model = None

    # Compute N_K per constraint percent based on true test set counts
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
                save_path='raw_results_PTO.csv',
            )
            pao_results = constrained_classification(
                current_model,
                None,
                test_loader,
                class_idx,
                N_K_val,
                device,
                save_path='raw_results_PAO.csv',
            )

            results = {
                "PTO_results": pto_results,
                "PAO_results": pao_results,
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

        process_results(base_dir=Path(f'results/{params.get("dataset", "unknown_dataset")}/{class_idx}/{base_timestamp}'))


def main():
    parser = argparse.ArgumentParser(description="Run full experiment: grid search then per-class constrained runs")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset override (else use config)")
    parser.add_argument("--lr-grid", type=float, nargs="+", default=[1e-3, 1e-4, 5e-5], help="Learning rate grid")
    parser.add_argument("--epoch-grid", type=int, nargs="+", default=[30, 50, 75], help="Epoch grid")
    parser.add_argument("--constraints", type=int, nargs="+", default=[90, 70, 50, 30], help="Constraint percentages")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Global seeding for reproducibility
    seed = int(args.seed)
    set_seed(seed)

    params, exp_name = get_experiment_config()
    # Pass seed to dataloaders via params for consistent worker seeding
    params["seed"] = seed
    if args.dataset is not None:
        params["dataset"] = args.dataset
        # Align model choice with dataset override
        ds = params["dataset"].lower()
        if ds in ('medmnist_oct', 'medmnist_tissue', 'tissuemnist', 'tissue'):
            # Grayscale MedMNIST datasets → 1-channel SimpleCNN
            params["model_choice"] = 'medmnist'
        else:
            # Use a 3-channel model for RGB datasets
            params["model_choice"] = 'EfficientNetB5'

    # Build dataloaders and capture num_classes
    train_loader, val_loader, test_loader, meta = get_dataloaders(params)
    num_classes = meta.get('num_classes') or 4
    params["num_classes"] = num_classes

    # Smart device selection: choose the most 'free' GPU when possible
    device = _select_smart_device()

    # Single base timestamp to group this entire full run
    base_timestamp = datetime.now().strftime("%m-%d/experiment_files/%H-%M")

    # 1) Grid search hyperparameters per dataset
    best_key, best_record, grid_records = grid_search_hparams(
        params,
        train_loader,
        val_loader,
        device,
        lr_grid=args.lr_grid,
        epoch_grid=args.epoch_grid,
        base_timestamp=base_timestamp,
    )

    # Save grid search summary under a dedicated folder
    grid_dir = Path(f"results/{params.get('dataset', 'unknown_dataset')}/hyperparam_search/{base_timestamp}")
    grid_dir.mkdir(parents=True, exist_ok=True)
    with open(grid_dir / "grid_results.json", "w") as f:
        json.dump(grid_records, f, indent=4)
    with open(grid_dir / "best_hyperparams.json", "w") as f:
        json.dump({"best": best_record, "key": {"lr": best_key[0], "epochs": best_key[1]}}, f, indent=4)

    # Update params with best hyperparameters
    if best_key is not None:
        params["initial_learning_rate"] = float(best_key[0])
        params["epochs"] = int(best_key[1])

    # 2) Run constrained experiment for each class
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

    # Optionally save a small manifest for this full run
    manifest = {
        "dataset": params.get("dataset"),
        "base_timestamp": base_timestamp,
        "num_classes": num_classes,
        "best_hyperparams": {"lr": params["initial_learning_rate"], "epochs": params["epochs"]},
        "constraints": args.constraints,
    }
    with open(grid_dir / "full_experiment_manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)


if __name__ == "__main__":
    main()
