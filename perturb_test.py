"""Is the order effect special? Compare reordering the chains with renaming the variables.

Each question is asked in four versions that all have the same correct answer:
  orig, swap (chains reversed), rename (new variable names), swap_rename (both).
Reports accuracy of each, how often the answer's correctness flips between versions, and
the accuracy of two-version and four-version ensembles (sum of log-probabilities).
The key comparisons: flip(orig, swap) vs flip(orig, rename), and the ensemble orig+swap vs
the equal-cost ensemble orig+rename.

  python perturb_test.py --model EleutherAI/pythia-12b --step 143000
  python perturb_test.py --model allenai/OLMo-2-1124-7B --revision main
  python perturb_test.py --model EleutherAI/pythia-12b --step 143000 --seed 1 --out $PERSIST/fullday/perturb_seed1
(--seed 1 draws a fresh question set; keep fresh-seed runs in their own folder, see claim C23.)
"""
import argparse
import json
import os

import numpy as np

import flipstats as fs
from perturb import INSTRUCT_VARIANTS, VARIANTS, final_answer, variants


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tasks", nargs="+", default=["vartrack_d1", "vartrack_d2", "vartrack_d3"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dtype", choices=("fp16", "bf16"), default=None)
    ap.add_argument("--instruct", action="store_true",
                    help="also ask with a plain instruction to attend to the asked variable (both orders)")
    ap.add_argument("--out", default="perturb_results")
    args = ap.parse_args()
    if (args.step is None) == (args.revision is None):
        raise SystemExit("give exactly one of --step or --revision")
    revision = args.revision or f"step{args.step}"
    names = VARIANTS + (INSTRUCT_VARIANTS if args.instruct else ())

    import torch
    import model_utils as mu
    from tasks import build_all

    # Build and check every version BEFORE loading the model: each must keep the same answer.
    items_all = build_all(args.n, args.seed, shots=2, layout="blocked")
    versions = {}
    for t in args.tasks:
        versions[t] = [variants(it["prompt"], seed=args.seed * 100003 + i, instruct=args.instruct) for i, it in enumerate(items_all[t])]
        want = [final_answer(v["orig"]) for v in versions[t]]
        if None in want:
            print(f"WARNING {t}: answer could not be computed from the prompt; skipping the answer check")
        else:
            bad = [(i, k) for i, v in enumerate(versions[t]) for k in names if final_answer(v[k]) != want[i]]
            assert not bad, f"{t}: a version changed the answer, e.g. {bad[:3]}"
            wrong = [i for i, it in enumerate(items_all[t]) if want[i] != str(it["candidates"][it["label"]]).strip()]
            if wrong:
                print(f"WARNING {t}: the computed answer differs from the item label for {len(wrong)} items, e.g. {wrong[:3]}")
    first = versions[args.tasks[0]][0]
    print(f"--- tails of the {len(names)} versions of item 0 ---")
    for k in names:
        print(f"[{k}]\n{first[k][-90:]}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = args.dtype or ("fp16" if "pythia" in args.model else "bf16")
    tok, model = mu.load(args.model, revision, dev,
                         (torch.float16 if dtype == "fp16" else torch.bfloat16) if dev == "cuda" else torch.float32)
    out = dict(model=args.model, revision=revision, step=args.step, seed=args.seed, tasks={})
    for t in args.tasks:
        items = items_all[t]
        y = np.array([it["label"] for it in items])
        cids = mu.candidate_ids(tok, items[0]["candidates"])
        logp = {}
        for k in names:
            lo, _ = mu.run(tok, model, [v[k] for v in versions[t]], cids, batch_size=args.batch_size, device=dev)
            logp[k] = torch.log_softmax(lo, -1).numpy()
        ok = {k: logp[k].argmax(-1) == y for k in names}
        length = {k: np.array([len(tok(v[k], add_special_tokens=False)["input_ids"]) for v in versions[t]]) for k in VARIANTS}
        ens = dict(swap=fs.ensemble_correct([logp["orig"], logp["swap"]], y),
                   rename=fs.ensemble_correct([logp["orig"], logp["rename"]], y),
                   all4=fs.ensemble_correct([logp[k] for k in VARIANTS], y))
        single_swap = (ok["orig"].astype(float) + ok["swap"]) / 2
        s = {f"acc_{k}": float(v.mean()) for k, v in ok.items()}
        s.update(acc_ens_swap=float(ens["swap"].mean()), acc_ens_rename=float(ens["rename"].mean()),
                 acc_ens_all4=float(ens["all4"].mean()), acc_single_mean_swap=float(single_swap.mean()),
                 flip_swap=fs.flip_rate(ok["orig"], ok["swap"]), flip_rename=fs.flip_rate(ok["orig"], ok["rename"]),
                 flip_swap_rename=fs.flip_rate(ok["orig"], ok["swap_rename"]),
                 same_swap=float((logp["orig"].argmax(-1) == logp["swap"].argmax(-1)).mean()),
                 same_rename=float((logp["orig"].argmax(-1) == logp["rename"].argmax(-1)).mean()),
                 tokens_swap_minus_orig=float((length["swap"] - length["orig"]).mean()),
                 tokens_rename_minus_orig=float((length["rename"] - length["orig"]).mean()), n=int(len(y)))
        if args.instruct:
            s.update(acc_instruct_mean=float((ok["instruct"].mean() + ok["instruct_swap"].mean()) / 2),
                     flip_instruct=fs.flip_rate(ok["instruct"], ok["instruct_swap"]))
        out["tasks"][t] = dict(summary=s, correct={k: v.astype(int).tolist() for k, v in ok.items()},
                               ens_correct={k: v.astype(int).tolist() for k, v in ens.items()})
        print(t, {k: round(v, 3) for k, v in s.items()}, flush=True)
    os.makedirs(args.out, exist_ok=True)
    # seed 0 keeps the original file name; any other seed (fresh question set, claim C23) gets a suffix
    # so it can never overwrite the seed-0 result. Run fresh seeds into a separate --out folder.
    name = f"{args.model.split('/')[-1]}__{revision}" + (f"__seed{args.seed}" if args.seed else "") + ".json"
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
