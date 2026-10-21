"""Does distance to the question drive the first-or-last habit? (exploratory)

Each question is asked with no-op lines ('>>> pass') inserted either between the two chains
(the first chain moves away from the question) or after both chains (both move away). The
correct answer never changes. Reports accuracy and the "picks the later chain" rate for each
condition, using the same analysis as position_test.py.

  python distance_test.py --model EleutherAI/pythia-12b --step 143000
"""
import argparse
import json
import os

CONDITIONS = [("none", 0), ("between", 2), ("between", 4), ("before_query", 2), ("before_query", 4)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tasks", nargs="+", default=["vartrack_d1", "vartrack_d2", "vartrack_d3"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="distance_results")
    args = ap.parse_args()
    if (args.step is None) == (args.revision is None):
        raise SystemExit("give exactly one of --step or --revision")
    revision = args.revision or f"step{args.step}"

    import torch
    import model_utils as mu
    from perturb import add_filler, final_answer, strip_filler
    from position_test import analyse
    from tasks import build_all

    items_all = build_all(args.n, args.seed, shots=2, layout="blocked")
    versions = {}
    for t in args.tasks:
        for where, k in CONDITIONS:
            vs = [dict(it, prompt=it["prompt"] if k == 0 else add_filler(it["prompt"], where, k)) for it in items_all[t]]
            for it, v in zip(items_all[t], vs):     # the answer must not change
                assert final_answer(strip_filler(v["prompt"])) == str(it["candidates"][it["label"]]).strip()
            versions[(t, where, k)] = vs
    t0 = args.tasks[0]
    print("--- tail with 2 lines between the chains ---\n" + versions[(t0, "between", 2)][0]["prompt"][-120:])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.float16 if "pythia" in args.model else torch.bfloat16) if dev == "cuda" else torch.float32
    tok, model = mu.load(args.model, revision, dev, dtype)
    out = dict(model=args.model, revision=revision, tasks={})
    for t in args.tasks:
        cids = mu.candidate_ids(tok, items_all[t][0]["candidates"])
        out["tasks"][t] = {}
        for where, k in CONDITIONS:
            vs = versions[(t, where, k)]
            logits, _ = mu.run(tok, model, [v["prompt"] for v in vs], cids, batch_size=args.batch_size, device=dev)
            pred = logits.argmax(-1).numpy()
            res = analyse(vs, pred, None)
            res["accuracy"] = float((pred == [v["label"] for v in vs]).mean())
            res["n"] = len(vs)
            out["tasks"][t][f"{where}{k}"] = res
            print(t, f"{where}{k}", "acc", round(res["accuracy"], 3), "picks_later", res["picks_later"]["rate"], flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.model.split('/')[-1]}__{revision}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
