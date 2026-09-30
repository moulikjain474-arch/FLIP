"""Run every task on every checkpoint; save candidate logits and activations.

  python sweep.py --model EleutherAI/pythia-410m --out runs/
  python sweep.py --model EleutherAI/pythia-160m-seed1 --out runs/
  python sweep.py --model EleutherAI/pythia-1.4b --steps 143000 --n 200 --out pilot/   # day-2 gate

One .npz per (model, checkpoint, item set, task). Files are written atomically
and finished ones are skipped, so a crashed Colab session can simply be rerun.
Each checkpoint's download is deleted afterwards to save disk (--keep-cache to keep).
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch

import io_utils as iou
import model_utils as mu
from tasks import build_all

# 13 log-spaced checkpoints that exist for every Pythia and PolyPythias run
DEFAULT_STEPS = [0, 32, 128, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000, 143000]


def hf_cache_dir(model):
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        HF_HUB_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
    return os.path.join(HF_HUB_CACHE, "models--" + model.replace("/", "--"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, nargs="*", default=DEFAULT_STEPS)
    ap.add_argument("--revision-format", default="step{}")
    ap.add_argument("--n", type=int, default=600, help="items per task")
    ap.add_argument("--seed", type=int, default=0, help="item-generation seed")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--keep-cache", action="store_true")
    args = ap.parse_args()

    tasks = build_all(args.n, args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(iou.items_path(args.out, args.seed, args.n), "w") as f:
        json.dump(tasks, f)  # the exact items, for the data book and for errors.py

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    for step in args.steps:
        rev = args.revision_format.format(step)
        path = lambda t: iou.result_path(args.out, args.model, rev, args.seed, args.n, t)
        todo = [t for t in tasks if not os.path.exists(path(t))]
        if not todo:
            print(f"[skip] {rev}: already done")
            continue
        tok, model = mu.load(args.model, rev, args.device, dtype)
        layers = mu.default_layers(model)
        for t in todo:
            items = tasks[t]
            cids = mu.candidate_ids(tok, items[0]["candidates"])
            logits, acts = mu.run(tok, model, [it["prompt"] for it in items], cids,
                                  read_layers=layers, batch_size=args.batch_size, device=args.device)
            labels = np.array([it["label"] for it in items])
            iou.save_npz_atomic(
                path(t), logits=logits.numpy(), labels=labels, layers=np.array(layers),
                step=step, tokens=step * mu.TOKENS_PER_STEP,
                **{f"acts_L{l}": a.numpy().astype(np.float16) for l, a in acts.items()})
            acc = float((logits.argmax(-1).numpy() == labels).mean())
            print(f"{args.model} {rev} {t}: behavior acc={acc:.3f} (chance {1 / len(cids):.3f})",
                  flush=True)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
        if not args.keep_cache:
            shutil.rmtree(hf_cache_dir(args.model), ignore_errors=True)


if __name__ == "__main__":
    main()
