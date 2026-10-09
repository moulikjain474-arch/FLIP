"""Absolute position vs distance to the question (exploratory; answers a mech-interp reviewer's question).

Same items, answer never changes. Padding with no-op lines ('>>> pass') is put in three places:
  start64   64 no-op lines at the very start of the prompt (everything shifts later in absolute position,
            all distances between chains and to the question stay the same)
  head8     8 no-op lines directly before the first chain of the final problem
            (both chains shift later, gap between the chains and to the question unchanged)
  between8  8 no-op lines between the two chains (first chain moves away from the question)
If the "picks the later chain" habit follows absolute position, start64/head8 change it and between8 does not;
if it follows distance to the question or order, the reverse.

  python3 head_pad_test.py --model EleutherAI/pythia-12b --step 143000
"""
import argparse
import json
import os

CONDITIONS = [("none", 0), ("start", 64), ("head", 8), ("between", 8)]


def pad(prompt, where, k):
    from perturb import FILLER, split_problem
    if k == 0:
        return prompt
    if where == "start":
        return "\n".join([FILLER] * k) + "\n" + prompt
    head, a, b, tail = split_problem(prompt)
    fill = [FILLER] * k
    body = fill + a + b if where == "head" else a + fill + b
    return head + "\n".join(body + tail)


def unpad(prompt):
    from perturb import FILLER
    return prompt.replace("\n" + FILLER, "").replace(FILLER + "\n", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tasks", nargs="+", default=["vartrack_d1", "vartrack_d2", "vartrack_d3"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dry", action="store_true", help="build and check prompts only, no model")
    ap.add_argument("--out", default="headpad_results")
    args = ap.parse_args()
    if not args.dry and (args.step is None) == (args.revision is None):
        raise SystemExit("give exactly one of --step or --revision")
    revision = args.revision or (f"step{args.step}" if args.step else None)

    from perturb import final_answer
    from tasks import build_all

    items_all = build_all(args.n, args.seed, shots=2, layout="blocked")
    versions = {}
    for t in args.tasks:
        for where, k in CONDITIONS:
            vs = [dict(it, prompt=pad(it["prompt"], where, k)) for it in items_all[t]]
            for it, v in zip(items_all[t], vs):
                assert final_answer(unpad(v["prompt"])) == str(it["candidates"][it["label"]]).strip()
            versions[(t, where, k)] = vs
    t0 = args.tasks[0]
    for where, k in CONDITIONS:
        p = versions[(t0, where, k)][0]["prompt"]
        print(f"--- {where}{k}: first 90 chars / last 140 chars\n{p[:90]!r}\n{p[-140:]!r}")
    if args.dry:
        print("dry run ok")
        return

    import model_utils as mu
    import torch
    from position_test import analyse
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
