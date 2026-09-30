"""Why does the model pick the wrong chain? Test the simplest explanation: position.

In the blocked layout the two chains appear one after the other, in random order. If the
model has not learned WHICH chain the question asks about, it may take the value from the
chain that comes later (or earlier) in the prompt. This script splits questions by whether
the queried chain's value appears after or before the distractor chain's value, and reports:

  correct / distractor rate in each group
  picks_later: among answers that name one of the two chain values, the share taken from
               the later chain, with a 95% Wilson interval and a two-sided binomial test vs 0.5

  No position shortcut -> similar correct rates in both groups, picks_later near 0.5.
  Position shortcut    -> correct rate high in one group and low in the other,
                          picks_later far from 0.5 (EXPERIMENTS.md, C5: >= 0.65 or <= 0.35
                          and a correct-rate gap >= 0.2).

Works for a training checkpoint (--step) or a final model (--revision main). The first task
prints an example prompt: check that the order detection is right.

  python position_test.py --model EleutherAI/pythia-1.4b --step 5000
  python position_test.py --model allenai/OLMo-2-1124-7B --revision main
"""
import argparse
import json
import os
import re

import numpy as np
import torch
from scipy import stats

import model_utils as mu
from steer_stage import distractor_index
from tasks import build_all


def last_pos(prompt, value):
    """Position of the last standalone occurrence of a number in the prompt (-1 if none).
    The queried problem comes after the worked examples, so the last occurrence is in it."""
    hits = [m.start() for m in re.finditer(rf"(?<!\d){re.escape(value)}(?!\d)", prompt)]
    return hits[-1] if hits else -1


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(mid - half), float(mid + half))


def group_rates(rows, later):
    g = [r for r in rows if r["query_later"] == later]
    if not g:
        return dict(n=0, correct=None, distractor=None)
    return dict(n=len(g), correct=float(np.mean([r["correct"] for r in g])),
                distractor=float(np.mean([r["distractor"] for r in g])))


def analyse(items, pred, distractor_key):
    rows = []
    for it, p in zip(items, pred):
        dis = distractor_index(it, distractor_key)
        cands = [str(c).strip() for c in it["candidates"]]
        pc, pd_ = last_pos(it["prompt"], cands[it["label"]]), last_pos(it["prompt"], cands[dis])
        if pc < 0 or pd_ < 0 or dis == it["label"]:
            continue
        rows.append(dict(query_later=pc > pd_, correct=bool(p == it["label"]), distractor=bool(p == dis)))
    later, earlier = group_rates(rows, True), group_rates(rows, False)
    chain = [r for r in rows if r["correct"] or r["distractor"]]
    k = sum((r["correct"] and r["query_later"]) or (r["distractor"] and not r["query_later"]) for r in chain)
    n = len(chain)
    gap = (later["correct"] - earlier["correct"]) if later["n"] and earlier["n"] else None
    return dict(
        n_usable=len(rows), query_chain_later=later, query_chain_earlier=earlier,
        correct_gap_later_minus_earlier=gap,
        picks_later=dict(k=int(k), n=n, rate=float(k / n) if n else None, ci95=wilson(k, n),
                         p_vs_half=float(stats.binomtest(k, n, 0.5).pvalue) if n else None),
        shortcut_by_c5=bool(n and gap is not None and abs(gap) >= 0.2
                            and (k / n >= 0.65 or k / n <= 0.35)),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--revision", default=None, help="instead of --step, e.g. main")
    ap.add_argument("--tasks", nargs="+", default=["vartrack_d1", "vartrack_d2", "vartrack_d3"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=2)
    ap.add_argument("--layout", choices=("interleaved", "blocked"), default="blocked")
    ap.add_argument("--distractor-key", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="position_results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if (args.step is None) == (args.revision is None):
        raise SystemExit("give exactly one of --step or --revision")
    revision = args.revision or f"step{args.step}"

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    tok, model = mu.load(args.model, revision, args.device, dtype)
    all_items = build_all(args.n, args.seed, shots=args.shots, layout=args.layout)
    summary = dict(model=args.model, revision=revision, layout=args.layout, tasks={})

    for i, task in enumerate(args.tasks):
        items = all_items[task]
        if i == 0:
            ex = items[0]
            print(f"--- example prompt tail ({task}) ---\n{ex['prompt'][-400:]}")
            print(f"--- answer={ex['candidates'][ex['label']]!r}  "
                  f"distractor={ex['candidates'][distractor_index(ex, args.distractor_key)]!r}\n")
        cids = mu.candidate_ids(tok, items[0]["candidates"])
        logits, _ = mu.run(tok, model, [it["prompt"] for it in items], cids,
                           batch_size=args.batch_size, device=args.device)
        res = analyse(items, logits.argmax(-1).numpy(), args.distractor_key)
        summary["tasks"][task] = res
        pl = res["picks_later"]
        print(f"{task}: correct later={res['query_chain_later']['correct']} "
              f"earlier={res['query_chain_earlier']['correct']}  picks_later={pl['rate']} "
              f"CI={pl['ci95']} p={pl['p_vs_half']}  shortcut(C5)={res['shortcut_by_c5']}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    name = f"{args.model.replace('/', '__')}__{revision}__{args.layout}.json"
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {os.path.join(args.out, name)}")


if __name__ == "__main__":
    main()
