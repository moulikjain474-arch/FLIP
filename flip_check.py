"""flip_check: how order-sensitive is a language model on a small variable-tracking task?

One command, any Hugging Face causal LM:

  python flip_check.py --model EleutherAI/pythia-1.4b --step 143000
  python flip_check.py --model Qwen/Qwen2.5-7B --revision main --n 300

Every question is asked three ways with the SAME correct answer: as written (orig), with the two
variable chains in reverse order (swap), and with the variables renamed (rename; a control that
changes the wording but not the order). The report shows:
  * accuracy in each version,
  * flip rate = share of questions answered correctly in exactly one of two versions
    (orig vs swap, and orig vs rename as the control),
  * accuracy when the model is asked in both orders and the answers are combined
    (sum of log-probabilities), with a 95% bootstrap interval of the gain over a single order,
  * the same for the equal-cost control that combines orig + rename.
The verdict line is descriptive (it is the rule used for claim C12 in EXPERIMENTS.md, but this tool
registers nothing). Needs the repo's task generator (tasks.py), perturb.py and model_utils.py.
"""
import argparse
import json
import os

import numpy as np

import flipstats as fs


def summarise(logp, y):
    """logp: {'orig','swap','rename'} -> [N, K] log-probs; y: labels. Pure function."""
    y = np.asarray(y)
    ok = {k: logp[k].argmax(-1) == y for k in ("orig", "swap", "rename")}
    ens_swap = fs.ensemble_correct([logp["orig"], logp["swap"]], y)
    ens_ren = fs.ensemble_correct([logp["orig"], logp["rename"]], y)
    single_swap = (ok["orig"].astype(float) + ok["swap"]) / 2
    single_ren = (ok["orig"].astype(float) + ok["rename"]) / 2
    g_swap = fs.paired_bootstrap(ens_swap, single_swap)
    g_ren = fs.paired_bootstrap(ens_ren, single_ren)
    fixed, broke = int((ens_swap & ~ens_ren).sum()), int((~ens_swap & ens_ren).sum())
    s = dict(n=int(len(y)),
             acc_orig=float(ok["orig"].mean()), acc_swap=float(ok["swap"].mean()), acc_rename=float(ok["rename"].mean()),
             flip_swap=fs.flip_rate(ok["orig"], ok["swap"]), flip_rename=fs.flip_rate(ok["orig"], ok["rename"]),
             acc_both_orders=float(ens_swap.mean()), acc_orig_plus_rename=float(ens_ren.mean()),
             gain_swap=g_swap[0], gain_swap_lo=g_swap[1], gain_swap_hi=g_swap[2],
             gain_rename=g_ren[0], gain_rename_lo=g_ren[1], gain_rename_hi=g_ren[2],
             swap_ens_only_right=fixed, rename_ens_only_right=broke, mcnemar_p=fs.mcnemar_exact(fixed, broke))
    d = s["flip_swap"] - s["flip_rename"]
    s["flip_diff"] = d
    s["verdict"] = ("ORDER-SENSITIVE: reordering changes the answer clearly more often than renaming does" if d >= 0.05 - 1e-9
                    else "NO CLEAR EXTRA ORDER EFFECT: reordering changes answers about as often as renaming does")
    return s


def report(task, s):
    pc = lambda x: f"{100 * x:.1f}%"
    lines = [f"[{task}]  n={s['n']}",
             f"  accuracy        orig {pc(s['acc_orig'])} | swapped order {pc(s['acc_swap'])} | renamed {pc(s['acc_rename'])}",
             f"  answer changes  swap {pc(s['flip_swap'])} | rename (control) {pc(s['flip_rename'])} | difference {100 * s['flip_diff']:+.1f} points",
             f"  both orders     accuracy {pc(s['acc_both_orders'])}  (gain {100 * s['gain_swap']:+.1f} points, 95% interval "
             f"{100 * s['gain_swap_lo']:+.1f} to {100 * s['gain_swap_hi']:+.1f})",
             f"  orig + rename   accuracy {pc(s['acc_orig_plus_rename'])}  (gain {100 * s['gain_rename']:+.1f} points, 95% interval "
             f"{100 * s['gain_rename_lo']:+.1f} to {100 * s['gain_rename_hi']:+.1f})",
             f"  both-orders vs orig+rename: right only with both orders {s['swap_ens_only_right']}, only with rename {s['rename_ens_only_right']} (exact p={s['mcnemar_p']:.3g})",
             f"  -> {s['verdict']}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--step", type=int, default=None, help="Pythia-style checkpoint step")
    ap.add_argument("--revision", default=None, help="e.g. main")
    ap.add_argument("--tasks", nargs="+", default=["vartrack_d2"])
    ap.add_argument("--n", type=int, default=300, help="questions per task (300 takes a few minutes on a small model)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dtype", choices=("fp16", "bf16"), default=None)
    ap.add_argument("--out", default=None, help="optional folder for a JSON copy of the numbers")
    args = ap.parse_args()
    if args.step is not None and args.revision is not None:
        raise SystemExit("give --step or --revision, not both")
    revision = args.revision or (f"step{args.step}" if args.step is not None else "main")

    import torch
    import model_utils as mu
    from perturb import variants
    from tasks import build_all

    items_all = build_all(args.n, args.seed, shots=2, layout="blocked")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = args.dtype or ("fp16" if "pythia" in args.model else "bf16")
    tok, model = mu.load(args.model, revision, dev,
                         (torch.float16 if dtype == "fp16" else torch.bfloat16) if dev == "cuda" else torch.float32)
    result = dict(model=args.model, revision=revision, seed=args.seed, tasks={})
    print(f"\nflip_check: {args.model} @ {revision}\n")
    for t in args.tasks:
        items = items_all[t]
        vs = [variants(it["prompt"], seed=args.seed * 100003 + i) for i, it in enumerate(items)]
        y = np.array([it["label"] for it in items])
        cids = mu.candidate_ids(tok, items[0]["candidates"])
        logp = {}
        for k in ("orig", "swap", "rename"):
            lo, _ = mu.run(tok, model, [v[k] for v in vs], cids, batch_size=args.batch_size, device=dev)
            logp[k] = torch.log_softmax(lo, -1).numpy()
        s = summarise(logp, y)
        result["tasks"][t] = s
        print(report(t, s) + "\n", flush=True)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"{args.model.split('/')[-1]}__{revision}.json"), "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
