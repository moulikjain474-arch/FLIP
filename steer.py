"""Causal test: does the model USE what the probe finds?

Run it at checkpoints before and after behavior appears. Before onset, accuracy
is near chance, so read the log p(correct) numbers, not accuracy: a drop in
log p(correct) under suppress (beyond the random control) means the model is
already using sub-threshold task information.

Conditions (last-token residual stream). Erasure is applied at the main layer
AND every later layer, each with its own subspace, so later layers cannot simply
re-read the information (--scope single: main layer only). Pushes are applied
once, at the main layer:
  natural          no edit
  suppress         erase the task subspace: set its coordinates to their mean   (PRIMARY)
  random_suppress  the same for random subspaces of equal rank, averaged over
                   --n-rand independent draws                                  (control)
  enhance          push toward the correct class direction                       (oracle)
  random_enhance   push along a random direction of the same size               (control)
  wrong_enhance    push toward a wrong class direction                          (control)
The task subspace is spanned by the class-mean offsets (an LDA-style subspace),
not the logistic probe's weights. It and the class directions come from a
separate TRAIN half of the items; all numbers are on the held-out TEST half.
'enhance' uses each item's label, so it shows the direction is usable, not a
deployable fix, and its effect is large almost by construction. The primary
test for H3 is therefore suppress vs random_suppress. The summary also reports
the fraction of activation variance each erasure removes, so a larger effect
cannot be explained away as "the task subspace just removed more variance"
without that being visible.

  python steer.py --model EleutherAI/pythia-1.4b --step 143000 --task vartrack_d1 --layout blocked
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy import stats

import io_utils as iou
import model_utils as mu
from tasks import build_all

MAIN_CONDITIONS = ("suppress", "natural", "enhance")


def split(items, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))
    half = len(items) // 2
    return [items[i] for i in idx[:half]], [items[i] for i in idx[half:]]


def rm_anova(conditions):
    """One-way repeated-measures ANOVA (the same items appear in every condition).
    Returns F, p and partial eta^2 = SS_condition / (SS_condition + SS_error)."""
    Y = np.stack(conditions, axis=1)                       # items x conditions
    n, k = Y.shape
    grand = Y.mean()
    ss_cond = n * ((Y.mean(0) - grand) ** 2).sum()
    ss_item = k * ((Y.mean(1) - grand) ** 2).sum()
    ss_err = ((Y - grand) ** 2).sum() - ss_cond - ss_item
    df_cond, df_err = k - 1, (k - 1) * (n - 1)
    F = (ss_cond / df_cond) / (ss_err / df_err)
    return float(F), float(stats.f.sf(F, df_cond, df_err)), float(ss_cond / (ss_cond + ss_err))


def frac_var_removed(H, mean, B):
    """Share of the centred activation variance lying in subspace B (orthonormal columns)."""
    Hc = H - mean
    return float(((Hc @ B) ** 2).sum() / (Hc ** 2).sum().clamp(min=1e-12))


@torch.no_grad()
def run_condition(tok, model, items, cids, edit_layers, edit_fn, batch_size, device):
    """Returns per-item log p(correct | K candidates) and correctness under one edit."""
    logps, oks = [], []
    cand = torch.tensor(cids, device=device)
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        labels = torch.tensor([it["label"] for it in chunk])
        edit = None if edit_fn is None else (edit_layers, lambda h, l, lab=labels: edit_fn(h, l, lab))
        enc, pos = mu.encode(tok, [it["prompt"] for it in chunk], device)
        with mu.hooks(model, (), edit):
            logits = mu.last_logits(model, enc, pos).index_select(-1, cand).float().cpu()
        logps.append(torch.log_softmax(logits, -1)[torch.arange(len(chunk)), labels])
        oks.append(logits.argmax(-1) == labels)
    return torch.cat(logps).numpy(), torch.cat(oks).numpy().astype(float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--layer", type=int, default=None, help="default: the middle layer")
    ap.add_argument("--scope", choices=("downstream", "single"), default="downstream",
                    help="edit the main layer and all later layers (default), or the main layer only")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="push size in units of the typical class-mean offset")
    ap.add_argument("--n-rand", type=int, default=5,
                    help="independent random subspaces averaged for random_suppress")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=2, help="must match the sweep")
    ap.add_argument("--layout", choices=("interleaved", "blocked"), default="interleaved",
                    help="must match the sweep")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="steer_results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    items = build_all(args.n, args.seed, shots=args.shots, layout=args.layout)[args.task]
    train, test = split(items, args.seed)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    tok, model = mu.load(args.model, f"step{args.step}", args.device, dtype)
    n_layers = mu.n_layers(model)
    layer = args.layer or n_layers // 2
    edit_layers = set(range(layer, n_layers + 1)) if args.scope == "downstream" else {layer}
    cids = mu.candidate_ids(tok, items[0]["candidates"])
    K = len(cids)

    # --- task subspaces and class directions from the TRAIN half only, per edited layer
    read = sorted(edit_layers | {layer})
    _, acts = mu.run(tok, model, [it["prompt"] for it in train], cids, read_layers=read,
                     batch_size=args.batch_size, device=args.device)
    y_train = torch.tensor([it["label"] for it in train])
    g = torch.Generator().manual_seed(args.seed)
    geo = {}
    for l in read:
        H = acts[l]
        mean = H.mean(0)
        class_means = torch.stack([H[y_train == c].mean(0) if (y_train == c).any() else mean
                                   for c in range(K)])
        D = class_means - mean                               # K x d class offsets
        U, S, _ = torch.linalg.svd(D.T, full_matrices=False)
        rank = int((S > 1e-6 * S.max()).sum())
        Q_rands = [torch.linalg.qr(torch.randn(H.shape[1], rank, generator=g))[0]
                   for _ in range(args.n_rand)]
        geo[l] = dict(mean=mean, Q=U[:, :rank], Q_rands=Q_rands, rank=rank,
                      unit=D / D.norm(dim=1, keepdim=True).clamp(min=1e-8),
                      scale=float(D.norm(dim=1).mean()))  # typical class offset, not ||h||
    rand_unit = torch.randn(K, acts[layer].shape[1], generator=g)
    rand_unit = rand_unit / rand_unit.norm(dim=1, keepdim=True)

    # variance removed at the main layer (train half), task vs random subspaces
    H_main, gm = acts[layer], geo[layer]
    var_task = frac_var_removed(H_main, gm["mean"], gm["Q"])
    var_rand = [frac_var_removed(H_main, gm["mean"], Qr) for Qr in gm["Q_rands"]]

    def to(t, h):
        return t.to(h.device, h.dtype)

    def erase(get):
        # mean-ablation inside each layer's own subspace (keeps activations on-distribution)
        def fn(h, l, lab):
            B, mu_l = to(get(geo[l]), h), to(geo[l]["mean"], h)
            return h - ((h - mu_l) @ B) @ B.T
        return fn

    def push(directions, pick):
        # a single push at the main layer only
        def fn(h, l, lab):
            if l != layer:
                return h
            return h + args.alpha * geo[layer]["scale"] * to(directions[pick(lab)], h)
        return fn

    conditions = {
        "natural": None,
        "suppress": erase(lambda gl: gl["Q"]),
        "enhance": push(geo[layer]["unit"], lambda lab: lab),
        "random_enhance": push(rand_unit, lambda lab: lab),
        "wrong_enhance": push(geo[layer]["unit"], lambda lab: (lab + 1) % K),
    }
    rank = geo[layer]["rank"]
    results = {}
    for name, fn in conditions.items():
        logp, ok = run_condition(tok, model, test, cids, edit_layers, fn, args.batch_size, args.device)
        results[name] = dict(logp=logp, acc=ok)
        print(f"{name:16s} acc={ok.mean():.3f}  mean log p(correct)={logp.mean():.3f}", flush=True)

    # random control: several independent random subspaces, averaged per item
    rand_logp, rand_acc = [], []
    for r in range(args.n_rand):
        fn = erase(lambda gl, r=r: gl["Q_rands"][r])
        logp, ok = run_condition(tok, model, test, cids, edit_layers, fn, args.batch_size, args.device)
        rand_logp.append(logp)
        rand_acc.append(ok)
        print(f"random_suppress[{r}] acc={ok.mean():.3f}  mean log p(correct)={logp.mean():.3f}", flush=True)
    results["random_suppress"] = dict(logp=np.mean(rand_logp, axis=0), acc=np.mean(rand_acc, axis=0))

    # --- statistics
    lp = {c: v["logp"] for c, v in results.items()}
    F, p_anova, eta2 = rm_anova([lp[c] for c in MAIN_CONDITIONS])
    paired = lambda a, b: stats.ttest_rel(lp[a], lp[b])
    t_sup, t_enh, t_wrong = (paired("suppress", "random_suppress"),
                             paired("enhance", "random_enhance"),
                             paired("wrong_enhance", "natural"))
    summary = dict(
        model=args.model, step=args.step, task=args.task, layer=layer, scope=args.scope,
        alpha=args.alpha, n_test=len(test), rank=rank, chance=1 / K, n_rand=args.n_rand,
        acc={c: float(v["acc"].mean()) for c, v in results.items()},
        mean_logp={c: float(v.mean()) for c, v in lp.items()},
        random_suppress_draws_mean_logp=[float(x.mean()) for x in rand_logp],
        frac_var_removed_main_layer=dict(task=var_task, random=var_rand),
        primary_test="suppress_vs_random",
        suppress_vs_random=dict(t=float(t_sup.statistic), p=float(t_sup.pvalue),
                                mean_diff=float((lp["suppress"] - lp["random_suppress"]).mean())),
        anova=dict(F=F, p=p_anova, partial_eta2=eta2, conditions=list(MAIN_CONDITIONS),
                   note="repeated measures; 'enhance' is an oracle edit"),
        enhance_vs_random=dict(t=float(t_enh.statistic), p=float(t_enh.pvalue)),
        wrong_vs_natural=dict(t=float(t_wrong.statistic), p=float(t_wrong.pvalue)),
    )
    os.makedirs(args.out, exist_ok=True)
    item_tag = iou.tag(args.seed, args.n, args.shots, args.layout)
    name = (f"{args.model.replace('/', '__')}__step{args.step}__{item_tag}__"
            f"{args.task}__L{layer}__{args.scope}__a{args.alpha}.json")
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
