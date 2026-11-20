"""Inspect the supplied MATLAB datasets without changing them."""

from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASETS = ("SwissRollData.mat", "PeaksData.mat", "GMMData.mat")
EXPECTED = {"Yt", "Ct", "Yv", "Cv"}


def describe_split(name, features, targets):
    """Check column-oriented data and report its class distribution."""
    print(f"  {name}:")
    if features.ndim != 2 or targets.ndim != 2:
        print("    WARNING: expected two-dimensional feature and target matrices")
        return None
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        print("    WARNING: non-finite values present")
        return None
    if features.shape[1] != targets.shape[1]:
        print("    WARNING: feature and label matrices have different sample counts")
        return None
    binary = np.all((targets == 0) | (targets == 1))
    if not binary or not np.all(targets.sum(axis=0) == 1):
        print("    WARNING: each label column must be one-hot")
        return None
    sample_count = targets.shape[1]
    class_counts = targets.sum(axis=1).astype(int)
    dimension = features.shape[0]
    print(f"    samples: {sample_count}; features: {dimension}; classes: {len(class_counts)}")
    print("    samples are columns in both matrices")
    print(f"    one-hot labels: yes; class counts: {class_counts.tolist()}")
    return dimension, len(class_counts)


def inspect(path):
    print(f"\n{path.name}")
    if not path.is_file():
        print("  ERROR: file not found")
        return False
    try:
        arrays = {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
    except (OSError, ValueError, NotImplementedError) as error:
        print(f"  ERROR: could not load file: {error}")
        return False
    for key in sorted(arrays):
        value = arrays[key]
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    missing = EXPECTED - arrays.keys()
    if missing:
        print(f"  ERROR: missing arrays: {', '.join(sorted(missing))}")
        return False
    extras = arrays.keys() - EXPECTED
    if extras:
        print(f"  NOTE: additional arrays: {', '.join(sorted(extras))}")
    train = describe_split("training", arrays["Yt"], arrays["Ct"])
    validation = describe_split("validation", arrays["Yv"], arrays["Cv"])
    if train is None or validation is None:
        return False
    if train != validation:
        print("  WARNING: training and validation feature/class dimensions differ")
        return False
    print("  training/validation feature and class dimensions match: yes")
    return True


def main():
    results = [inspect(DATA_DIR / name) for name in DATASETS]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
