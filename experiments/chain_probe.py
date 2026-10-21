"""Does the residual stream keep track of WHICH chain the question points at, even when the answer flips? (R2b)

For one checkpoint: read the last-token residual stream at several layers and fit a cross-validated linear probe
for "the queried variable belongs to the FIRST chain / the SECOND chain" (the label the model needs in order to choose
correctly). Also records the model's own preference: among answers equal to one of the two chain start values, the
share that take the second chain's value (picks_later).
Across checkpoints:
  probe steady while picks_later swings  -> the reversal is in how the choice is read out, not in the representation
  probe moves with picks_later           -> the representation changed
Exploratory (mentor suggestion 8 Oct). Probe settings are fixed in advance (C=0.1, 5 folds, standardised), as in probes.py.

  python chain_probe.py --model EleutherAI/pythia-1.4b --step 16000 --out chain_probe_results
"""
import argparse
import json
import os

import numpy as np

from perturb import chain_values


def labels_and_picks(prompts, pred_idx, cands):
    """(which chain is queried [0/1] per item, picks_later rate or None) from prompts and predicted candidate indices."""
    target, first, second = [], 0, 0
    for p, pi in zip(prompts, pred_idx):
        vals, t = chain_values(p)
        target.append(t)
        a = cands[int(pi)]
        if vals[0] == vals[1]:            # ambiguous (never happens in tasks.py, which draws distinct values)
            continue
        if a == vals[1]:
            second += 1
        elif a == vals[0]:
            first += 1
    n = first + second
    return np.array(target), (second / n if n else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tasks", nargs="+", default=["vartrack_d2", "vartrack_d3"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="chain_probe_results")
    args = ap.parse_args()
    if (args.step is None) == (args.revision is None):
        raise SystemExit("give exactly one of --step or --revision")
    revision = args.revision or f"step{args.step}"

    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import model_utils as mu
    from tasks import build_all

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.float16 if "pythia" in args.model else torch.bfloat16) if dev == "cuda" else torch.float32
    tok, model = mu.load(args.model, revision, dev, dtype)
    L = mu.n_layers(model)
    layers = sorted(set(range(args.every, L + 1, args.every)) | {1, L})
    items_all = build_all(args.n, args.seed, shots=2, layout="blocked")
    out = dict(model=args.model, revision=revision, n_layers=L, layers=layers, tasks={})

    def cv_acc(X, y, seed=0):
        correct = np.zeros(len(y), bool)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
            clf.fit(X[tr], y[tr])
            correct[te] = clf.predict(X[te]) == y[te]
        return float(correct.mean())

    for t in args.tasks:
        items = items_all[t]
        prompts = [it["prompt"] for it in items]
        cands = [str(c).strip() for c in items[0]["candidates"]]
        cids = mu.candidate_ids(tok, items[0]["candidates"])
        logits, acts = mu.run(tok, model, prompts, cids, read_layers=layers, batch_size=args.batch_size, device=dev)
        y, picks_later = labels_and_picks(prompts, logits.argmax(-1).numpy(), cands)
        rng = np.random.default_rng(args.seed)
        res = dict(n=int(len(y)), picks_later=picks_later, share_second=float(y.mean()), layers={})
        for l in layers:
            X = acts[l].numpy()
            res["layers"][l] = dict(acc=cv_acc(X, y), acc_shuffled=cv_acc(X, rng.permutation(y)))
            print(t, "layer", l, res["layers"][l], flush=True)
        print(t, "picks_later", picks_later, flush=True)
        out["tasks"][t] = res
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.model.split('/')[-1]}__{revision}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
