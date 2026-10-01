"""Order-Flip Test on short Python programs (code_task.py): harder and more realistic than
the variable chains. Each program is asked with its definitions in two orders; the printed
answer is the same in both (checked by running it).

  python code_test.py --model EleutherAI/pythia-12b --step 143000
  python code_test.py --model Qwen/Qwen2.5-72B --revision main --batch-size 8
"""
import argparse
import json
import os

import flipstats as fs
from code_task import make_code


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="code_results")
    args = ap.parse_args()
    if (args.step is None) == (args.revision is None):
        raise SystemExit("give exactly one of --step or --revision")
    revision = args.revision or f"step{args.step}"

    import torch
    import model_utils as mu

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.float16 if "pythia" in args.model else torch.bfloat16) if dev == "cuda" else torch.float32
    tok, model = mu.load(args.model, revision, dev, dtype)
    out = dict(model=args.model, revision=revision, tasks={})
    for level in args.levels:
        items = make_code(args.n, level, args.seed)
        if level == args.levels[0]:
            print(items[0]["prompt"][-220:] + items[0]["candidates"][items[0]["label"]])
        cids = mu.candidate_ids(tok, items[0]["candidates"])
        y = [it["label"] for it in items]
        lo, _ = mu.run(tok, model, [it["prompt"] for it in items], cids, batch_size=args.batch_size, device=dev)
        ls, _ = mu.run(tok, model, [it["prompt_swapped"] for it in items], cids, batch_size=args.batch_size, device=dev)
        r = fs.pair_metrics(torch.log_softmax(lo, -1).numpy(), torch.log_softmax(ls, -1).numpy(), y)
        out["tasks"][f"code_l{level}"] = r
        print(f"code_l{level}", {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.model.split('/')[-1]}__{revision}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
