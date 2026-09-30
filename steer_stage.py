"""Causal tests at one checkpoint: erase or plant the chain values the model represents.

At the coin-flip stage the model answers with one of the two chain values almost at random.
This script asks whether that choice runs through what the last-token residual stream
encodes about each chain's value:

  ERASE  remove the distractor chain's value and see whether the model picks the right chain.
  PLANT  overwrite the distractor (or answer) value with a different number and see whether
         the model starts answering with exactly that planted number. This predicts WHICH
         answer changes, not just whether accuracy moves (the test proposed, but not run,
         by Ravulapalli & Chadha 2026, Sections 7.1 and 9.5).

Subspaces are spanned by class-mean offsets (LDA-style), computed on a TRAIN half of the
items, separately at the main layer and every later layer; every number is measured on the
held-out TEST half. An edit sets the coordinates inside a subspace to a target (the train
mean for erasure, one class mean for planting), at the main layer and all later layers
(--scope single: main layer only).

Conditions
  natural          no edit
  erase_dis_orth   distractor-value subspace minus the answer-value subspace   (PRIMARY erase)
  erase_dis        raw distractor-value subspace
  erase_cor        answer-value subspace (sanity check: should hurt)
  random[k]        random subspaces of the PRIMARY rank, --n-rand draws        (control)
  pca_top          top principal components of the PRIMARY rank               (harsh control)
  plant_dis        distractor-value coordinates set to the class mean of a third number p
                   (p differs from the answer and the distractor)              (PRIMARY plant)
  plant_cor        the same inside the answer-value subspace
Primary tests (EXPERIMENTS.md, C4 and C4b), paired over held-out items:
  erase: log p(correct) under erase_dis_orth vs the mean of the random draws
  plant: rate of answering p under plant_dis vs under natural

  python steer_stage.py --model EleutherAI/pythia-1.4b --step 5000 --task vartrack_d1
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy import stats

import io_utils as iou
import model_utils as mu
from steer import frac_var_removed, split
from tasks import build_all

DISTRACTOR_KEYS = ("distractor", "distractor_label", "distractor_idx", "distractor_index",
                   "distractor_value")


def distractor_index(it, key=None):
    """Index (into it['candidates']) of the distractor chain's value.

    tasks.py stores it as it['meta']['distractor'] (the value as a string, e.g. '4');
    top-level fields are checked too, for other item formats."""
    keys = (key,) if key else DISTRACTOR_KEYS
    cands = [str(c).strip() for c in it["candidates"]]
    for source in (it.get("meta") or {}, it):
        for k in keys:
            if k not in source:
                continue
            v = source[k]
            if isinstance(v, (int, np.integer)) and k.endswith(("label", "idx", "index")):
                if 0 <= int(v) < len(cands):
                    return int(v)
            if str(v).strip() in cands:
                return cands.index(str(v).strip())
    raise KeyError(f"no usable distractor field; item keys are {sorted(it)}, "
                   f"meta keys are {sorted(it.get('meta') or {})}. "
                   f"Pass --distractor-key with the right one.")


def class_geometry(H, labels, K, tol=1e-6):
    """Class means (K x d) and an orthonormal basis of their offsets from the overall mean."""
    mean = H.mean(0)
    cm = torch.stack([H[labels == c].mean(0) if (labels == c).any() else mean for c in range(K)])
    U, S, _ = torch.linalg.svd((cm - mean).T, full_matrices=False)
    r = int((S > tol * S.max()).sum()) if S.numel() and S.max() > 0 else 0
    return cm, U[:, :r]


def orth_complement(B, A, tol=1e-3):
    """Orthonormal basis of span(B) with span(A) projected out."""
    if B.shape[1] == 0:
        return B
    R = B - A @ (A.T @ B)
    U, S, _ = torch.linalg.svd(R, full_matrices=False)
    r = int((S > tol * S.max()).sum()) if S.max() > 0 else 0
    return U[:, :r]


def paired(a, b):
    """Paired comparison of a vs b over the same items: mean difference, t, p, Cohen's d_z."""
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1)
    t = stats.ttest_rel(a, b) if sd > 0 else None
    return dict(mean_diff=float(d.mean()), n=int(len(d)),
                t=float(t.statistic) if t else float("nan"),
                p=float(t.pvalue) if t else float("nan"),
                d_z=float(d.mean() / sd) if sd > 0 else float("nan"))


@torch.no_grad()
def evaluate(tok, model, items, cids, edit_layers, make_edit, batch_size, device):
    """Per-item log p(correct) and predicted candidate index under one condition.
    make_edit(item_indices) -> fn(h, layer) or None (no edit)."""
    logps, preds = [], []
    cand = torch.tensor(cids, device=device)
    for i in range(0, len(items), batch_size):
        idx = list(range(i, min(i + batch_size, len(items))))
        fn = make_edit(idx) if make_edit else None
        edit = (edit_layers, fn) if fn else None
        enc, pos = mu.encode(tok, [items[j]["prompt"] for j in idx], device)
        with mu.hooks(model, (), edit):
            logits = mu.last_logits(model, enc, pos).index_select(-1, cand).float().cpu()
        labels = torch.tensor([items[j]["label"] for j in idx])
        logps.append(torch.log_softmax(logits, -1)[torch.arange(len(idx)), labels])
        preds.append(logits.argmax(-1))
    return torch.cat(logps).numpy(), torch.cat(preds).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--revision", default=None, help="instead of --step, e.g. main")
    ap.add_argument("--task", required=True, help="vartrack_d1 / vartrack_d2 / vartrack_d3")
    ap.add_argument("--layer", type=int, default=None, help="default: the middle layer")
    ap.add_argument("--scope", choices=("downstream", "single"), default="downstream")
    ap.add_argument("--n-rand", type=int, default=5)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=2)
    ap.add_argument("--layout", choices=("interleaved", "blocked"), default="blocked")
    ap.add_argument("--distractor-key", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="steer_stage_results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if (args.step is None) == (args.revision is None):
        raise SystemExit("give exactly one of --step or --revision")
    revision = args.revision or f"step{args.step}"

    # --- items: answer, distractor and a planted third value per item
    items = build_all(args.n, args.seed, shots=args.shots, layout=args.layout)[args.task]
    rng = np.random.default_rng(args.seed + 1)
    for it in items:
        it["_dis"] = distractor_index(it, args.distractor_key)
        others = [c for c in range(len(it["candidates"])) if c not in (it["label"], it["_dis"])]
        it["_plant"] = int(rng.choice(others))
    items = [it for it in items if it["_dis"] != it["label"]]
    train, test = split(items, args.seed)

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    tok, model = mu.load(args.model, revision, args.device, dtype)
    n_layers = mu.n_layers(model)
    layer = n_layers // 2 if args.layer is None else args.layer
    if not 1 <= layer <= n_layers:
        raise SystemExit(f"--layer must be in 1..{n_layers}")
    edit_layers = set(range(layer, n_layers + 1)) if args.scope == "downstream" else {layer}
    cids = mu.candidate_ids(tok, items[0]["candidates"])
    K = len(cids)

    # --- geometry from the TRAIN half, per edited layer
    read = sorted(edit_layers | {layer})
    _, acts = mu.run(tok, model, [it["prompt"] for it in train], cids, read_layers=read,
                     batch_size=args.batch_size, device=args.device)
    y_cor = torch.tensor([it["label"] for it in train])
    y_dis = torch.tensor([it["_dis"] for it in train])
    g = torch.Generator().manual_seed(args.seed)
    geo = {}
    for l in read:
        H = acts[l]
        mean = H.mean(0)
        cm_cor, Q_cor = class_geometry(H, y_cor, K)
        cm_dis, Q_dis = class_geometry(H, y_dis, K)
        Q_orth = orth_complement(Q_dis, Q_cor)
        r = Q_orth.shape[1]
        _, _, Vt = torch.linalg.svd(H - mean, full_matrices=False)
        geo[l] = dict(mean=mean, cm_cor=cm_cor, cm_dis=cm_dis, Q_cor=Q_cor, Q_dis=Q_dis,
                      Q_orth=Q_orth, Q_pca=Vt[:r].T.contiguous(),
                      Q_rand=[torch.linalg.qr(torch.randn(H.shape[1], r, generator=g))[0]
                              for _ in range(args.n_rand)])
    gm = geo[layer]
    rank = int(gm["Q_orth"].shape[1])
    if rank == 0:
        print("WARNING: primary subspace has rank 0 at the main layer; erase tests are no-ops here.")
    frac_var = {k: frac_var_removed(acts[layer], gm["mean"], gm[k])
                for k in ("Q_orth", "Q_dis", "Q_cor", "Q_pca")}
    frac_var["Q_rand"] = [frac_var_removed(acts[layer], gm["mean"], Q) for Q in gm["Q_rand"]]

    # --- edits
    def erase(get):
        def make(_idx):
            def fn(h, l):
                B = get(geo[l]).to(h.device, h.dtype)
                m = geo[l]["mean"].to(h.device, h.dtype)
                return h - ((h - m) @ B) @ B.T
            return fn
        return make

    def plant(qkey, cmkey):
        def make(idx):
            targets = torch.tensor([test[j]["_plant"] for j in idx])
            def fn(h, l):
                G = geo[l]
                B = G[qkey].to(h.device, h.dtype)
                m = G["mean"].to(h.device, h.dtype)
                tgt = (G[cmkey][targets] - G["mean"]).to(h.device, h.dtype)
                return h - ((h - m) @ B) @ B.T + (tgt @ B) @ B.T
            return fn
        return make

    conditions = {
        "natural": None,
        "erase_dis_orth": erase(lambda G: G["Q_orth"]),
        "erase_dis": erase(lambda G: G["Q_dis"]),
        "erase_cor": erase(lambda G: G["Q_cor"]),
        "pca_top": erase(lambda G: G["Q_pca"]),
        "plant_dis": plant("Q_dis", "cm_dis"),
        "plant_cor": plant("Q_cor", "cm_cor"),
    }
    conditions.update({f"random{k}": erase(lambda G, k=k: G["Q_rand"][k]) for k in range(args.n_rand)})

    labels = np.array([it["label"] for it in test])
    dis = np.array([it["_dis"] for it in test])
    plant_idx = np.array([it["_plant"] for it in test])
    res = {}
    for name, make in conditions.items():
        logp, pred = evaluate(tok, model, test, cids, edit_layers, make, args.batch_size, args.device)
        res[name] = dict(logp=logp, correct=(pred == labels).astype(float),
                         distractor=(pred == dis).astype(float),
                         planted=(pred == plant_idx).astype(float))
        r_ = res[name]
        print(f"{name:15s} correct={r_['correct'].mean():.3f}  distractor={r_['distractor'].mean():.3f}  "
              f"planted={r_['planted'].mean():.3f}  log p(correct)={r_['logp'].mean():.3f}", flush=True)
    rand_names = [f"random{k}" for k in range(args.n_rand)]
    res["random_mean"] = {m: np.mean([res[n][m] for n in rand_names], axis=0)
                          for m in ("logp", "correct", "distractor", "planted")}

    summary = dict(
        model=args.model, revision=revision, step=args.step, task=args.task, layer=layer,
        scope=args.scope, n_test=len(test), K=K, rank_primary=rank, n_rand=args.n_rand,
        rates={n: {m: float(np.mean(v)) for m, v in r_.items()} for n, r_ in res.items()},
        frac_var_removed_main_layer=frac_var,
        primary_erase=dict(test="log p(correct): erase_dis_orth vs random_mean",
                           **paired(res["erase_dis_orth"]["logp"], res["random_mean"]["logp"])),
        primary_plant=dict(test="answers = planted value: plant_dis vs natural",
                           **paired(res["plant_dis"]["planted"], res["natural"]["planted"])),
        secondary=dict(
            erase_accuracy=paired(res["erase_dis_orth"]["correct"], res["random_mean"]["correct"]),
            erase_distractor_rate=paired(res["erase_dis_orth"]["distractor"], res["random_mean"]["distractor"]),
            plant_cor=paired(res["plant_cor"]["planted"], res["natural"]["planted"]),
            erase_cor_sanity=paired(res["erase_cor"]["logp"], res["natural"]["logp"]),
        ),
    )
    os.makedirs(args.out, exist_ok=True)
    tag = iou.tag(args.seed, args.n, args.shots, args.layout)
    name = f"{args.model.replace('/', '__')}__{revision}__{tag}__{args.task}__L{layer}__{args.scope}.json"
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("primary_erase", "primary_plant", "rank_primary")}, indent=2))
    print(f"wrote {os.path.join(args.out, name)}")


if __name__ == "__main__":
    main()
