"""Flatten the full-day results into one small CSV (long format: one number per row).

  python export_csv.py --dir $PERSIST/fullday --out /shared-docker/earlymind_fullday.csv
Columns: kind, model, revision, task, layer, metric, value, n
"""
import argparse
import csv
import glob
import json
import os


def rows(root):
    for f in glob.glob(os.path.join(root, "position*", "**", "*.json"), recursive=True):
        s = json.load(open(f))
        kind = "pos_interleaved" if "position_interleaved" in f else "pos"
        for t, v in s["tasks"].items():
            pl = v["picks_later"]
            m = s["model"].split("/")[-1]
            yield kind, m, s["revision"], t, "", "picks_later", pl["rate"], pl["n"]
            yield kind, m, s["revision"], t, "", "correct_query_later", v["query_chain_later"]["correct"], pl["n"]
            yield kind, m, s["revision"], t, "", "correct_query_earlier", v["query_chain_earlier"]["correct"], pl["n"]
    for f in glob.glob(os.path.join(root, "swap", "*.json")):
        s = json.load(open(f))
        for t, r in s["tasks"].items():
            for k, v in r.items():
                if k != "n":
                    yield "swap", s["model"].split("/")[-1], s.get("revision") or f"step{s['step']}", t, "", k, v, r["n"]
    for f in glob.glob(os.path.join(root, "perturb", "*.json")):
        s = json.load(open(f))
        for t, r in s["tasks"].items():
            for k, v in r["summary"].items():
                if k != "n":
                    yield "perturb", s["model"].split("/")[-1], s["revision"], t, "", k, v, r["summary"]["n"]
    for f in glob.glob(os.path.join(root, "benchmark", "*.json")):
        s = json.load(open(f))
        for k, v in s["summary"].items():
            if k != "n":
                yield "benchmark", s["model"].split("/")[-1], s.get("revision") or "main", "mmlu", "", k, v, s["summary"]["n"]
    for f in glob.glob(os.path.join(root, "steer_layers", "*.json")):
        s = json.load(open(f))
        for cond in ("natural", "plant_cor", "plant_dis", "erase_cor"):
            if cond in s["rates"]:
                for k, v in s["rates"][cond].items():
                    yield "layer", s["model"].split("/")[-1], s["revision"], s["task"], s["layer"], f"{cond}.{k}", v, s["n_test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "model", "revision", "task", "layer", "metric", "value", "n"])
        n = 0
        for r in rows(args.dir):
            w.writerow(r)
            n += 1
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
