"""Causal test: can we fix the model's wrong-chain mistake by erasing the wrong chain?

At a checkpoint in the wrong-chain stage, the model often answers with the distractor
chain's value. We find the directions that encode the DISTRACTOR value (from class means
grouped by distractor value, train half only), remove from them anything shared with the
CORRECT-value directions, and erase what is left at the main layer and every later layer.
If the model is using that information to pick the wrong chain, correct answers should go
up and distractor answers down, more than for control erasures of the same rank.

Conditions (last-token residual stream, erasure = set coordinates to their train mean):
  natural          no edit
  erase_dis_orth   distractor subspace minus the correct-answer subspace        (PRIMARY)
  erase_dis        raw distractor subspace
  erase_cor        correct-answer subspace (should hurt; sanity check)
  random[k]        random subspaces, same rank as PRIMARY, --n-rand draws        (control)
  pca_top          top principal components, same rank as PRIMARY               (harsh control)
Primary test (EXPERIMENTS.md): mean log p(correct), erase_dis_orth vs the average of the
random draws, paired over held-out items.

Check your item fields first (the distractor key is looked up automatically, or pass
--distractor-key):
  python -c "from tasks import build_all; it=build_all(4,0,shots=2,layout='blocked')['vartrack_d1'][0]; print({k:v for k,v in it.items() if k!='prompt'})"

  python steer_stage.py --model EleutherAI/pythia-1.4b --step 4000 --task vartrack_d1 --layout blocked
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

DISTRACTOR_KEYS = ("distractor_label", "distractor_idx", "distractor_index",
                   "distractor_value", "distractor")


def distractor_index(it, key=None):
    """Index (into it['candidates']) of the distractor chain's value."""
    keys = (key,) if key else DISTRACTOR_KEYS
    for k in keys:
        if k not in it:
            continue
        v = it[k]
        cands = [str(c).strip() for c in it["candidates"]]
        if str(v).strip() in cands and not k.endswith(("label", "idx", "index")):
            return cands.index(str(v).strip())
        if isinstance(v, (int, np.integer)) and 0 <= int(v) < len(cands):
            return int(v)
        if str(v).strip() in cands:
            return cands.index(str(v).strip())
    raise KeyError(f"no usable distractor field; item keys are {sorted(it)}. "
                   f"Pass --distractor-key with the right one.")


def class_subspace(H, labels, K, tol=1e-6):
    mean = H.mean(0)
    cm = torch.stack([H[labels == c].mean(0) if (labels == c).any() else mean for c in range(K)])
    U, S, _ = torch.linalg.svd((cm - mean).T, full_matrices=False)
    r = int((S > tol * S.max()).sum())
    return U[:, :r]


def orth_complement(B, A, tol=1e-3):
    """Orthonormal basis of span(B) with span(A) projected out."""
    R = B - A @ (A.T @ B)
    U, S, _ = torch.linalg.svd(R, full_matrices=False)
    r = int((S > tol * S.max()).sum()) if S.numel() else 0
    return U[:, :r]


@torch.no_grad()
def evaluate(tok, model, items, cids, dis_idx, edit_layers, edit_fn, bs, device):
    lp, pred = [], []
    cand = torch.tensor(cids, device=device)
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        edit = None if edit_fn is None else (edit_layers, edit_fn)
        enc, pos = mu.encode(tok, [it["prompt"] for it in chunk], device)
        with mu.hooks(model, (), edit):
            logits = mu.last_logits(model, enc, pos).index_select(-1, cand).float().cpu()
        labels = torch.tensor([it["label"] for it in chunk])
        lp.append(torch.log_softmax(logits, -1)[torch.arange(len(chunk)), labels])
        pred.append(logits.argmax(-1))
    lp, pred = torch.cat(lp).numpy(), torch.cat(pred).numpy()
    labels = np.array([it["label"] for it in items])
    return dict(logp=lp, correct=(pred == labels).astype(float),
                distractor=(pred == dis_idx).astype(float))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--task", required=True, help="vartrack_d1 / d2 / d3")
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

    items = build_all(args.n, args.seed, shots=args.shots, layout=args.layout)[args.task]
    for it in items:
        it["_dis"] = distractor_index(it, args.distractor_key)
    items = [it for it in items if it["_dis"] != it["label"]]
    train, test = split(items, args.seed)

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    tok, model = mu.load(args.model, f"step{args.step}", args.device, dtype)
    n_layers = mu.n_layers(model)
    layer = args.layer or n_layers // 2
    edit_layers = set(range(layer, n_layers + 1)) if args.scope == "downstream" else {layer}
    cids = mu.candidate_ids(tok, items[0]["candidates"])
    K = len(cids)

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
        Q_cor = class_subspace(H, y_cor, K)
        Q_dis = class_subspace(H, y_dis, K)
        Q_orth = orth_complement(Q_dis, Q_cor)
        r = Q_orth.shape[1]
        _, _, Vt = torch.linalg.svd(H - mean, full_matrices=False)
        geo[l] = dict(mean=mean, Q_cor=Q_cor, Q_dis=Q_dis, Q_orth=Q_orth,
                      Q_pca=Vt[:r].T.contiguous(),
                      Q_rand=[torch.linalg.qr(torch.randn(H.shape[1], r, generator=g))[0]
                              for _ in range(args.n_rand)])

    gm, Hm = geo[layer], acts[layer]
    frac_var = {k: frac_var_removed(Hm, gm["mean"], gm[k]) for k in ("Q_orth", "Q_dis", "Q_cor", "Q_pca")}
    frac_var["Q_rand"] = [frac_var_removed(Hm, gm["mean"], Q) for Q in gm["Q_rand"]]

    def erase(get):
        def fn(h, l):
            B = get(geo[l]).to(h.device, h.dtype)
            mu_l = geo[l]["mean"].to(h.device, h.dtype)
            return h - ((h - mu_l) @ B) @ B.T
        return fn

    dis_test = np.array([it["_dis"] for it in test])
    run = lambda fn: evaluate(tok, model, test, cids, dis_test, edit_layers, fn, args.batch_size, args.device)
    res = {"natural": run(None),
           "erase_dis_orth": run(erase(lambda G: G["Q_orth"])),
           "erase_dis": run(erase(lambda G: G["Q_dis"])),
           "erase_cor": run(erase(lambda G: G["Q_cor"])),
           "pca_top": run(erase(lambda G: G["Q_pca"]))}
    rand = [run(erase(lambda G, k=k: G["Q_rand"][k])) for k in range(args.n_rand)]
    res["random_mean"] = {m: np.mean([r[m] for r in rand], axis=0) for m in ("logp", "correct", "distractor")}

    for name, r in res.items():
        print(f"{name:15s} correct={r['correct'].mean():.3f}  distractor={r['distractor'].mean():.3f}  "
              f"log p(correct)={r['logp'].mean():.3f}", flush=True)

    t_lp = stats.ttest_rel(res["erase_dis_orth"]["logp"], res["random_mean"]["logp"])
    t_acc = stats.ttest_rel(res["erase_dis_orth"]["correct"], res["random_mean"]["correct"])
    summary = dict(
        model=args.model, step=args.step, task=args.task, layer=layer, scope=args.scope,
        n_test=len(test), K=K, rank_primary=int(gm["Q_orth"].shape[1]), n_rand=args.n_rand,
        rates={n: dict(correct=float(r["correct"].mean()), distractor=float(r["distractor"].mean()),
                       logp=float(r["logp"].mean())) for n, r in res.items()},
        random_draws_logp=[float(r["logp"].mean()) for r in rand],
        frac_var_removed_main_layer=frac_var,
        primary=dict(test="erase_dis_orth vs random_mean, paired t on log p(correct)",
                     t=float(t_lp.statistic), p=float(t_lp.pvalue),
                     mean_diff=float((res["erase_dis_orth"]["logp"] - res["random_mean"]["logp"]).mean())),
        secondary_accuracy=dict(t=float(t_acc.statistic), p=float(t_acc.pvalue),
                                diff=float((res["erase_dis_orth"]["correct"] - res["random_mean"]["correct"]).mean())),
    )
    os.makedirs(args.out, exist_ok=True)
    item_tag = iou.tag(args.seed, args.n, args.shots, args.layout)
    name = f"{args.model.replace('/', '__')}__step{args.step}__{item_tag}__{args.task}__L{layer}__{args.scope}.json"
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
