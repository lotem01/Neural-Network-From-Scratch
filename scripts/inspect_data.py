"""Inspect the supplied MATLAB datasets without changing them."""

from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASETS = ("SwissRollData.mat", "PeaksData.mat", "GMMData.mat")
EXPECTED = {"Yt", "Ct", "Yv", "Cv"}


def describe_split(name, features, targets):
    """Identify the sample axis from a unique one-hot interpretation."""
    print(f"  {name}:")
    if features.ndim != 2 or targets.ndim != 2:
        print("    WARNING: expected two-dimensional feature and target matrices")
        return None
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        print("    WARNING: non-finite values present")
        return None
    binary = np.all((targets == 0) | (targets == 1))
    candidates = []
    if binary:
        for target_axis in (0, 1):
            if np.all(targets.sum(axis=1 - target_axis) == 1):
                for feature_axis in (0, 1):
                    if features.shape[feature_axis] == targets.shape[target_axis]:
                        candidates.append((feature_axis, target_axis))
    if len(candidates) != 1:
        print("    WARNING: cannot uniquely infer one-hot sample axes; inspect manually")
        return None
    feature_axis, target_axis = candidates[0]
    sample_count = targets.shape[target_axis]
    class_counts = targets.sum(axis=target_axis).astype(int)
    dimension = features.shape[1 - feature_axis]
    print(f"    samples: {sample_count}; features: {dimension}; classes: {len(class_counts)}")
    print(f"    sample axis: features={feature_axis}, labels={target_axis} (zero-based)")
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
