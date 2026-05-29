import json
import os
import random

import numpy as np
import torch


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _assert_finite_tensor(name, tensor):
    if not torch.isfinite(tensor).all():
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        raise ValueError(f"{name} has {nan_count} NaNs and {inf_count} Infs")


def make_run_key(**kwargs):
    """Build a filename-safe string from hyperparameters."""
    parts = [f"{k}={v}" for k, v in sorted(kwargs.items())]
    return "__".join(parts)


def try_load_results(save_dir, key):
    """Return cached results dict if save_dir/key.json exists, else None."""
    if save_dir is None:
        return None
    path = os.path.join(save_dir, key + ".json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_results(save_dir, key, results):
    """Persist results dict as JSON to save_dir/key.json."""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, key + ".json"), "w") as f:
        json.dump(results, f, indent=2)
