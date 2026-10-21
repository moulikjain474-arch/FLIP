"""Flatten the full-day results into one small CSV (long format: one number per row).

  python export_csv.py --dir $PERSIST/fullday --out /shared-docker/earlymind_fullday.csv
Columns: kind, model, revision, task, layer, metric, value, n
"""
import argparse
import csv
import glob
import json
import os

import extra_stats


def rows(root):
    for f in glob.glob(os.path.join(root, "position*", "**", "*.json"), recursive=True):
        s = json.load(open(f))
        kind = ("pos_interleaved" if "position_interleaved" in f else "pos_olmo" if "position_olmo" in f else "pos")
        for t, v in s["tasks"].items():
            pl = v["picks_later"]
            m = s["model"].split("/")[-1]
            yield kind, m, s["revision"], t, "", "picks_later", pl["rate"], pl["n"]
            yield kind, m, s["revision"], t, "", "correct_query_later", v["query_chain_later"]["correct"], v["query_chain_later"].get("n")
            yield kind, m, s["revision"], t, "", "correct_query_earlier", v["query_chain_earlier"]["correct"], v["query_chain_earlier"].get("n")
    for f in glob.glob(os.path.join(root, "swap", "*.json")):
        s = json.load(open(f))
        for t, r in s["tasks"].items():
            for k, v in r.items():
                if k != "n":
                    yield "swap", s["model"].split("/")[-1], s.get("revision") or f"step{s['step']}", t, "", k, v, r["n"]
    for f in glob.glob(os.path.join(root, "perturb", "*.json")) + glob.glob(os.path.join(root, "perturb_olmo_traj", "*.json")):
        s = json.load(open(f))
        kind = "perturb_traj" if "perturb_olmo_traj" in os.path.dirname(f) else "perturb"
        for t, r in s["tasks"].items():
            for k, v in r["summary"].items():
                if k != "n":
                    yield kind, s["model"].split("/")[-1], s["revision"], t, "", k, v, r["summary"]["n"]
    for f in glob.glob(os.path.join(root, "benchmark", "*.json")):
        s = json.load(open(f))
        for k, v in s["summary"].items():
            if k != "n":
                yield "benchmark", s["model"].split("/")[-1], s.get("revision") or "main", "mmlu", "", k, v, s["summary"]["n"]
    for f in glob.glob(os.path.join(root, "distance", "*.json")):
        s = json.load(open(f))
        for t, conds in s["tasks"].items():
            for c, r in conds.items():
                m = s["model"].split("/")[-1]
                yield "distance", m, s["revision"], t, "", f"{c}.accuracy", r["accuracy"], r.get("n", r["n_usable"])
                yield "distance", m, s["revision"], t, "", f"{c}.picks_later", r["picks_later"]["rate"], r["picks_later"]["n"]
    for f in glob.glob(os.path.join(root, "patch", "*.json")) + glob.glob(os.path.join(root, "patch_choice", "*.json")):
        s = json.load(open(f))
        src = "_choice" if os.path.basename(os.path.dirname(f)) == "patch_choice" else ""
        for t, r in s["tasks"].items():
            for layer, v in r["layers"].items():
                for k, x in v.items():
                    yield "patch" + src, s["model"].split("/")[-1], s["revision"], t, layer, k, x, r["n_differ"]
            for layer, v in r.get("control", {}).get("layers", {}).items():
                for k, x in v.items():
                    yield "patch_control" + src, s["model"].split("/")[-1], s["revision"], t, layer, k, x, r["control"]["n"]
            for k in ("margin_first_minus_second_orig", "margin_first_minus_second_swap"):
                if r.get(k):
                    yield "patch_margin" + src, s["model"].split("/")[-1], s["revision"], t, "", k + ".mean", sum(r[k]) / len(r[k]), len(r[k])
    for f in glob.glob(os.path.join(root, "code", "*.json")) + glob.glob(os.path.join(root, "story", "*.json")):
        s = json.load(open(f))
        for t, r in s["tasks"].items():
            for k, v in r.items():
                if k != "n":
                    yield ("code" if os.path.basename(os.path.dirname(f)) == "code" else "story"), s["model"].split("/")[-1], s["revision"], t, "", k, v, r["n"]
    for f in glob.glob(os.path.join(root, "attention", "*.json")):
        s = json.load(open(f))
        for t, r in s["tasks"].items():
            m = s["model"].split("/")[-1]
            for i, (v, g) in enumerate(zip(r["layer_later_minus_earlier"], r["layer_gap_by_choice"]), start=1):
                yield "attention", m, s["revision"], t, i, "later_minus_earlier", v, r["n"]
                yield "attention", m, s["revision"], t, i, "gap_by_choice", g, r["n_picked_later"] + r["n_picked_earlier"]
    for f in glob.glob(os.path.join(root, "steer_layers", "*.json")):
        s = json.load(open(f))
        for cond in ("natural", "plant_cor", "plant_dis", "erase_cor"):
            if cond in s["rates"]:
                for k, v in s["rates"][cond].items():
                    yield "layer", s["model"].split("/")[-1], s["revision"], s["task"], s["layer"], f"{cond}.{k}", v, s["n_test"]


def gate_rows(root):
    """Claim C22 (uncertainty_gate.py) and C23 (fresh-seed perturb runs)."""
    for f in glob.glob(os.path.join(root, "uncertainty", "*.json")):
        s = json.load(open(f))
        m = s["model"].split("/")[-1]
        for t, r in s["tasks"].items():
            sm = r["summary"]
            for gname in ("gate_orig_margin", "gate_swap_margin", "gate_min_margin"):
                g = sm[gname]
                for k in ("low_flip", "high_flip", "diff", "diff_lo", "diff_hi", "ratio", "fisher_p", "spearman_margin_flip"):
                    if g.get(k) is not None:
                        yield "gate", m, s["revision"], t, "", f"{gname}.{k}", g[k], sm["n"]
                for i, v in enumerate(g["group_flip"]):
                    yield "gate", m, s["revision"], t, "", f"{gname}.group{i}_flip", v, g["group_n"][i]
            for i, v in enumerate(sm["acc_orig_by_group"]):
                yield "gate", m, s["revision"], t, "", f"acc_orig_group{i}", v, sm["n"]
    for f in glob.glob(os.path.join(root, "perturb_seed*", "*.json")):
        s = json.load(open(f))
        for t, r in s["tasks"].items():
            for k, v in r["summary"].items():
                if k != "n":
                    yield "perturb_seed%s" % s.get("seed", "?"), s["model"].split("/")[-1], s["revision"], t, "", k, v, r["summary"]["n"]


def extra_rows(root):
    for kind, m, rev, t, k, v, n in extra_stats.rows(root):
        yield kind, m, rev, t, "", k, v, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "model", "revision", "task", "layer", "metric", "value", "n"])
        n = 0
        for r in list(rows(args.dir)) + list(gate_rows(args.dir)) + list(extra_rows(args.dir)):
            w.writerow(r)
            n += 1
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
