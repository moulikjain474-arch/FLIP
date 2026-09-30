"""File naming shared by every script, so results can never mix item sets.

Result file:  <model with / -> __>__<revision>__<tag>__<task>.npz
Items file:   items_<tag>.json
tag = s<seed>n<n>, plus k<shots> and/or "blocked" when the variable-tracking
format differs from the default (so different formats never mix).
"""
import json
import os

import numpy as np


def tag(seed, n, shots=2, layout="interleaved"):
    t = f"s{seed}n{n}"
    if shots != 2:
        t += f"k{shots}"
    if layout != "interleaved":
        t += layout
    return t


def items_path(runs, item_tag):
    return os.path.join(runs, f"items_{item_tag}.json")


def result_path(runs, model, revision, item_tag, task):
    return os.path.join(runs, f"{model.replace('/', '__')}__{revision}__{item_tag}__{task}.npz")


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
