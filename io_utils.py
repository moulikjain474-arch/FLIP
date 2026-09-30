"""File naming shared by every script, so results can never mix item sets.

Result file:  <model with / -> __>__<revision>__s<seed>n<n>__<task>.npz
Items file:   items_s<seed>n<n>.json
"""
import json
import os

import numpy as np


def tag(seed, n):
    return f"s{seed}n{n}"


def items_path(runs, seed, n):
    return os.path.join(runs, f"items_{tag(seed, n)}.json")


def result_path(runs, model, revision, seed, n, task):
    return os.path.join(runs, f"{model.replace('/', '__')}__{revision}__{tag(seed, n)}__{task}.npz")


def parse_result_name(path):
    """-> (model, revision, item_tag, task)."""
    parts = os.path.basename(path)[: -len(".npz")].split("__")
    return "/".join(parts[:-3]), parts[-3], parts[-2], parts[-1]


def is_result(path):
    return path.endswith(".npz") and not path.endswith(".tmp.npz")


def save_npz_atomic(path, **arrays):
    """Write to a temp file, then rename, so a crash never leaves a truncated result."""
    tmp = path[: -len(".npz")] + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_items(runs, item_tag):
    with open(os.path.join(runs, f"items_{item_tag}.json")) as f:
        return json.load(f)
