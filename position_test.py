"""Why does the model pick the wrong chain? Test the simplest explanation: position.

In the blocked layout the two chains appear one after the other. If the model has not
learned WHICH chain the question is about, it may just take the value from the chain that
appears later (or earlier) in the prompt. This script splits items by whether the queried
chain's value appears after or before the distractor chain's value in the prompt, and
reports correct / distractor rates for each group, plus the overall "picks the later
chain" rate among answers that name one of the two chain values.

  No position shortcut -> correct rate is about the same in both groups.
  Position shortcut    -> correct rate is high in one group and low in the other, and
                          "picks later" is far from 0.5.

Works for any checkpoint (--step) or any final model (--revision main).
Check a printed example prompt to confirm the order detection looks right.

  python position_test.py --model EleutherAI/pythia-1.4b --step 5000
  python position_test.py --model allenai/OLMo-2-1124-7B --revision main
"""
import argparse
import json
import os
import re

import numpy as np
import torch

import model_utils as mu
from steer_stage import distractor_index
from tasks import build_all


def last_pos(prompt, value):
    """Position of the last standalone occurrence of a number in the prompt (-1 if none)."""
    hits = [m.start() for m in re.finditer(rf"(?<!\d){re.escape(value)}(?!\d)", prompt)]
    return hits[-1] if hits else -1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--revision", default=None, help="e.g. main, for models without step checkpoints")
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

    for task in args.tasks:
        items = all_items[task]
        cids = mu.candidate_ids(tok, items[0]["candidates"])
        logits, _ = mu.run(tok, model, [it["prompt"] for it in items], cids,
                           batch_size=args.batch_size, device=args.device)
        pred = logits.argmax(-1).numpy()
        rows = []
        for it, p in zip(items, pred):
            dis = distractor_index(it, args.distractor_key)
            cands = [str(c).strip() for c in it["candidates"]]
            pc, pd_ = last_pos(it["prompt"], cands[it["label"]]), last_pos(it["prompt"], cands[dis])
            if pc < 0 or pd_ < 0 or dis == it["label"]:
                continue
            rows.append(dict(query_later=pc > pd_, correct=p == it["label"], distractor=p == dis))
        if task == args.tasks[0] and items:
            print(f"--- example prompt tail ({task}): check which chain is queried and where ---")
            print(items[0]["prompt"][-400:])
            print(f"--- label={items[0]['candidates'][items[0]['label']]!r}  "
                  f"distractor={items[0]['candidates'][distractor_index(items[0], args.distractor_key)]!r}")
        res = {}
        for later in (True, False):
            g = [r for r in rows if r["query_later"] == later]
            key = "query_chain_later" if later else "query_chain_earlier"
            res[key] = dict(n=len(g),
                            correct=float(np.mean([r["correct"] for r in g])) if g else None,
                            distractor=float(np.mean([r["distractor"] for r in g])) if g else None)
        chain_answers = [r for r in rows if r["correct"] or r["distractor"]]
        picks_later = [(r["correct"] and r["query_later"]) or (r["distractor"] and not r["query_later"])
                       for r in chain_answers]
        res["picks_later_chain"] = float(np.mean(picks_later)) if picks_later else None
        res["n_chain_answers"] = len(chain_answers)
        summary["tasks"][task] = res
        print(f"{task}: later={res['query_chain_later']}  earlier={res['query_chain_earlier']}  "
              f"picks_later_chain={res['picks_later_chain']}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    name = f"{args.model.replace('/', '__')}__{revision}__{args.layout}.json"
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
