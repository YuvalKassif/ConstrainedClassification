import os
from collections import Counter, defaultdict
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import datasets, transforms
from medmnist import OCTMNIST, BloodMNIST, DermaMNIST

from utils import set_seed


def _flatten_target(y):
    try:
        if isinstance(y, torch.Tensor):
            return int(y.squeeze().item())
        import numpy as _np
        if isinstance(y, _np.ndarray):
            return int(_np.squeeze(y).item())
        if isinstance(y, (list, tuple)):
            return int(y[0])
        return int(y)
    except Exception:
        # Best-effort fallback
        return int(y) if not isinstance(y, (list, tuple)) else int(y[0])


def get_weighted_sampler(dataset) -> WeightedRandomSampler:
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, 'labels'):
        targets = dataset.labels
    elif hasattr(dataset, 'samples'):
        targets = [label for _, label in dataset.samples]
    else:
        raise ValueError("Dataset missing 'targets'/'labels'/'samples' for weighted sampling.")

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    elif isinstance(targets, list):
        targets = np.array(targets)

    if targets.ndim > 1:
        targets = targets.flatten()

    class_counts = Counter(targets)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = np.array([class_weights[label] for label in targets])
    return WeightedRandomSampler(torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True)


def _transforms_for(image_size: int, channels: int, mean: List[float], std: List[float]) -> Dict[str, transforms.Compose]:
    normalize = transforms.Normalize(mean=mean, std=std)
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((image_size, image_size))
    color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    if channels == 1:
        # For grayscale datasets (e.g., MedMNIST OCT)
        train_tf = [resize, transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(3), to_tensor, normalize]
        eval_tf = [resize, to_tensor, normalize]
    else:
        # For RGB datasets
        train_tf = [resize, transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(3),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    color_jitter, to_tensor, normalize]
        eval_tf = [resize, to_tensor, normalize]

    return {
        'train': transforms.Compose(train_tf),
        'val': transforms.Compose(eval_tf),
        'test': transforms.Compose(eval_tf)
    }


def _has_tv_splits(root: str) -> bool:
    return all(os.path.isdir(os.path.join(root, d)) for d in ['train', 'val', 'test'])


def _stratified_indices(targets: List[int], splits=(0.7, 0.15, 0.15), seed: int = 42):
    rng = np.random.RandomState(seed)
    by_class = defaultdict(list)
    for idx, y in enumerate(targets):
        by_class[int(y)].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for cls, idxs in by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(splits[0] * n))
        n_val = int(round(splits[1] * n))
        cls_train = idxs[:n_train]
        cls_val = idxs[n_train:n_train + n_val]
        cls_test = idxs[n_train + n_val:]
        train_idx.extend(cls_train)
        val_idx.extend(cls_val)
        test_idx.extend(cls_test)

    return train_idx, val_idx, test_idx


def load_imagefolder(root: str, batch_size: int, image_size: int = 224, channels: int = 3,
                     mean: List[float] = None, std: List[float] = None, seed: int = 42,
                     splits=(0.7, 0.15, 0.15)):
    if mean is None or std is None:
        if channels == 3:
            mean, std = [0.66133188] * 3, [0.21229856] * 3
        else:
            mean, std = [0.5], [0.5]

    tfs = _transforms_for(image_size=image_size, channels=channels, mean=mean, std=std)

    if _has_tv_splits(root):
        train_ds = datasets.ImageFolder(os.path.join(root, 'train'), transform=tfs['train'])
        val_ds = datasets.ImageFolder(os.path.join(root, 'val'), transform=tfs['val'])
        test_ds = datasets.ImageFolder(os.path.join(root, 'test'), transform=tfs['test'])
    else:
        # Single-folder with class subdirs. Build stratified splits.
        base_ds = datasets.ImageFolder(root, transform=tfs['train'])
        targets = base_ds.targets
        idx_train, idx_val, idx_test = _stratified_indices(targets, splits=splits, seed=seed)
        train_ds = Subset(base_ds, idx_train)
        # For val/test, re-wrap with eval transforms by setting dataset.transform appropriately
        val_base = datasets.ImageFolder(root, transform=tfs['val'])
        test_base = datasets.ImageFolder(root, transform=tfs['test'])
        val_ds = Subset(val_base, idx_val)
        test_ds = Subset(test_base, idx_test)

    num_workers = 4
    generator = lambda wid: np.random.seed(seed + wid)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              worker_init_fn=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            worker_init_fn=generator)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             worker_init_fn=generator)

    # Determine num_classes and names
    if hasattr(train_ds, 'dataset') and hasattr(train_ds.dataset, 'classes'):
        class_names = train_ds.dataset.classes
    elif hasattr(train_ds, 'classes'):
        class_names = train_ds.classes
    else:
        # Fallback
        class_names = []

    meta = {
        'num_classes': len(class_names) if class_names else None,
        'class_names': class_names,
        'channels': channels,
        'image_size': image_size
    }
    return train_loader, val_loader, test_loader, meta


def load_medmnist_oct(batch_size: int = 32, seed: int = 42):
    set_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Ensure labels are flattened to int via target_transform
    target_tf = transforms.Lambda(_flatten_target)

    train_dataset = OCTMNIST(split='train', transform=transform, target_transform=target_tf, download=False)
    val_dataset = OCTMNIST(split='val', transform=transform, target_transform=target_tf, download=False)
    test_dataset = OCTMNIST(split='test', transform=transform, target_transform=target_tf, download=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    meta = {
        'num_classes': 4,
        'class_names': [str(i) for i in range(4)],
        'channels': 1,
        'image_size': 28
    }
    return train_loader, val_loader, test_loader, meta


def load_medmnist_blood(batch_size: int = 32, seed: int = 42):
    """Load MedMNIST BloodMNIST (RGB, 8 classes)."""
    set_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Ensure labels are flattened to int via target_transform
    target_tf = transforms.Lambda(_flatten_target)

    train_dataset = BloodMNIST(split='train', transform=transform, target_transform=target_tf, download=False)
    val_dataset = BloodMNIST(split='val', transform=transform, target_transform=target_tf, download=False)
    test_dataset = BloodMNIST(split='test', transform=transform, target_transform=target_tf, download=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    meta = {
        'num_classes': 8,
        'class_names': [str(i) for i in range(8)],
        'channels': 3,
        'image_size': 28
    }
    return train_loader, val_loader, test_loader, meta


def load_medmnist_derma(batch_size: int = 32, seed: int = 42):
    """Load MedMNIST DermaMNIST (RGB, 7 classes)."""
    set_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    target_tf = transforms.Lambda(_flatten_target)

    train_dataset = DermaMNIST(split='train', transform=transform, target_transform=target_tf, download=False)
    val_dataset = DermaMNIST(split='val', transform=transform, target_transform=target_tf, download=False)
    test_dataset = DermaMNIST(split='test', transform=transform, target_transform=target_tf, download=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    meta = {
        'num_classes': 7,
        'class_names': [str(i) for i in range(7)],
        'channels': 3,
        'image_size': 28
    }
    return train_loader, val_loader, test_loader, meta


def get_dataloaders(params) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    dataset = params.get('dataset', 'medmnist_oct')
    batch_size = params.get('batch_size', 32)
    seed = 42
    set_seed(seed)

    if dataset == 'medmnist_oct':
        return load_medmnist_oct(batch_size=batch_size, seed=seed)
    elif dataset in ('medmnist_blood', 'bloodmnist'):
        return load_medmnist_blood(batch_size=batch_size, seed=seed)
    elif dataset in ('medmnist_derma', 'dermamnist', 'dermamnist_v2', 'derma'):
        return load_medmnist_derma(batch_size=batch_size, seed=seed)
    elif dataset == 'kneeKL224':
        root = params.get('data_dir_knee')
        if not root:
            raise ValueError("'data_dir_knee' must be set in params for kneeKL224 dataset.")
        return load_imagefolder(root=root, batch_size=batch_size, image_size=224, channels=3)
    elif dataset == 'lc25000':
        root = params.get('data_dir_lc25000')
        if not root:
            raise ValueError("'data_dir_lc25000' must be set in params for lc25000 dataset.")
        print(f"[DEBUG] LC25000 root: '{root}'")
        # LC25000 has a specific layout: 'Train and Validation Set' and 'Test Set'.
        loaders = load_lc25000(root=root, batch_size=batch_size, image_size=224, seed=seed)
        # Debug: summarize detected classes
        try:
            _, _, _, meta = loaders
            print(f"[DEBUG] LC25000 meta: classes={meta.get('class_names')} num_classes={meta.get('num_classes')}")
        except Exception as e:
            print(f"[DEBUG] LC25000 meta extraction failed: {e}")
        return loaders
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Choose from 'kneeKL224', 'medmnist_oct', 'medmnist_blood', 'medmnist_derma', 'lc25000'."
        )


def load_lc25000(root: str, batch_size: int, image_size: int = 224, seed: int = 42):
    """Load LC25000 where root contains 'Train and Validation Set' and 'Test Set' class subfolders.

    Train/Val are created via stratified split from 'Train and Validation Set'.
    Test is read from 'Test Set'.
    """
    set_seed(seed)
    try:
        print(f"[DEBUG] LC25000 root subdirs: {sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])}")
    except Exception as e:
        print(f"[DEBUG] LC25000 listdir failed for root: {e}")

    trainval_dir = os.path.join(root, 'Train and Validation Set')
    test_dir = os.path.join(root, 'Test Set')
    print(f"[DEBUG] LC25000 paths: trainval_dir='{trainval_dir}' exists={os.path.isdir(trainval_dir)}, test_dir='{test_dir}' exists={os.path.isdir(test_dir)}")
    if not os.path.isdir(trainval_dir) or not os.path.isdir(test_dir):
        # Fallback to generic loader if structure differs
        print("[DEBUG] LC25000 fallback to generic ImageFolder loader")
        return load_imagefolder(root=root, batch_size=batch_size, image_size=image_size, channels=3)

    tfs = _transforms_for(image_size=image_size, channels=3, mean=[0.66133188]*3, std=[0.21229856]*3)

    # Build train/val from the combined trainval directory using stratified split
    base_trainval = datasets.ImageFolder(trainval_dir, transform=tfs['train'])
    print(f"[DEBUG] LC25000 trainval classes detected: {base_trainval.classes} (n={len(base_trainval.classes)})")
    try:
        print(f"[DEBUG] LC25000 trainval immediate subdirs: {sorted([d for d in os.listdir(trainval_dir) if os.path.isdir(os.path.join(trainval_dir,d))])}")
    except Exception as e:
        print(f"[DEBUG] LC25000 listdir failed for trainval_dir: {e}")
    targets = getattr(base_trainval, 'targets', None)
    if targets is None:
        # Older torchvision may not expose .targets; build from samples
        targets = [label for _, label in base_trainval.samples]
    from collections import Counter as _Counter
    print(f"[DEBUG] LC25000 trainval label distribution: {_Counter(targets)} total={len(targets)}")
    idx_train, idx_val, _ = _stratified_indices(targets, splits=(0.85, 0.15, 0.0), seed=seed)

    train_ds = Subset(base_trainval, idx_train)

    # For val, re-wrap with eval transforms to avoid augmentation
    val_base = datasets.ImageFolder(trainval_dir, transform=tfs['val'])
    val_ds = Subset(val_base, idx_val)

    # Test set comes from dedicated folder
    test_ds = datasets.ImageFolder(test_dir, transform=tfs['test'])
    print(f"[DEBUG] LC25000 test classes detected: {test_ds.classes} (n={len(test_ds.classes)})")
    try:
        print(f"[DEBUG] LC25000 test immediate subdirs: {sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir,d))])}")
    except Exception as e:
        print(f"[DEBUG] LC25000 listdir failed for test_dir: {e}")

    num_workers = 4
    generator = lambda wid: np.random.seed(seed + wid)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=generator)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=generator)

    # Class names from the base dataset
    class_names = base_trainval.classes
    meta = {
        'num_classes': len(class_names),
        'class_names': class_names,
        'channels': 3,
        'image_size': image_size
    }
    return train_loader, val_loader, test_loader, meta
