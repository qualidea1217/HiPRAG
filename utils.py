import json
import os
import random
import re
import warnings
from typing import Iterable

import numpy as np
import torch
import transformers


def seed_worker(worker_id):
    # Base seed + worker_id ensures different seeds for different workers
    # Note: This relies on the global `initial_seed` being set before DataLoader init
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_everything(seed: int):
    """
    Sets the seed for generating random numbers in PyTorch, numpy, and Python
    across different platforms (CPU, NVIDIA GPU, AMD GPU, Apple Silicon, TPU)
    and attempts to enforce deterministic behavior.

    Also sets up a suggestion for seeding DataLoader workers.

    Args:
        seed (int): The seed value to use.

    Returns:
        Callable: A `worker_init_fn` suitable for `torch.utils.data.DataLoader`
                  to ensure reproducibility with multiple workers.
    """
    global initial_seed  # Store seed for worker_init_fn
    initial_seed = seed

    platforms_seeded = ["random", "numpy", "torch"]

    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch on CPU
    torch.manual_seed(seed)

    # Use Hugging Face Transformers' seeding
    transformers.set_seed(seed)

    # Set seed for PyTorch on GPU (if available)
    # Handles NVIDIA CUDA and AMD ROCm GPUs if PyTorch is built with support
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # Set seed for all GPUs if using multi-GPU setup
        torch.cuda.manual_seed_all(seed)
        platforms_seeded.append("cuda/rocm")

        # Settings for CUDA/cuDNN (primarily for NVIDIA GPUs)
        # These might improve reproducibility on NVIDIA GPUs but can impact performance.
        # They might not have an effect or could raise warnings on ROCm setups.
        try:
            # Attempt to set cuDNN flags if available and relevant
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except AttributeError:
            # cudnn backend might not be available (e.g., ROCm)
            print("torch.backends.cudnn not available, skipping deterministic/benchmark settings.")

    # Set seed for Apple Silicon (MPS) backend (if available)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        platforms_seeded.append("mps")

    # Set seed for PyTorch/XLA (TPU)
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
        xm.set_rng_state(seed)
        platforms_seeded.append("tpu")
    except ImportError:
        # If torch_xla is not installed, just skip
        pass

    # Attempt to use deterministic algorithms
    # Note: This can potentially slow down operations or cause errors if a
    # deterministic implementation is not available for a specific operation.
    try:
        # Set related environment variable (needed for some operations)
        # Must be set before 'use_deterministic_algorithms'
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # Enable deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
        platforms_seeded.append("torch_deterministic")
    except Exception as e:
        warnings.warn(f"Could not enable deterministic algorithms: {e}")

    # Set seed for Python hash seed (optional, for certain hash-based operations)
    os.environ['PYTHONHASHSEED'] = str(seed)
    platforms_seeded.append("hashseed")

    print(f"Set seed to {seed} for: {', '.join(platforms_seeded)}.")
    print("NOTE: For reproducible DataLoader results with num_workers > 0,")
    print("      use the returned 'worker_init_fn'. Example:")
    print(f"      loader = DataLoader(..., worker_init_fn=seed_worker)")

    # Return the worker_init_fn for convenience with DataLoaders
    return seed_worker


def load_jsonl(data_dir: str) -> list[dict]:
    with open(data_dir, encoding="utf-8") as data_file:
        data_list = data_file.readlines()
    return [json.loads(line) for line in data_list]


def write_jsonl(json_lines: Iterable[dict], data_dir: str):
    with open(data_dir, encoding="utf-8", mode='w') as data_file:
        for data in json_lines:
            data_file.write(json.dumps(data) + '\n')


def extract_str_between(text: str, str_before: str, str_after: str):
    pattern = f"{re.escape(str_before)}(.*?){re.escape(str_after)}"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return matches
