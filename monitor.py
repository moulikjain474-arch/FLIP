"""Do the model's mistakes warn us before it becomes competent? And the locked prediction.

Part 1 (evaluate): for every run x task, how many doublings of training tokens before
competence (T_cor) does each pre-registered signal fire, compared with the plain-accuracy
baseline (B_acc)? A signal is only useful if it fires EARLIER than the baseline.

Part 2 (--predict, EXPLORATORY ONLY, see EXPERIMENTS.md "Changes"): writes predictions.json from two rules
fixed in EXPERIMENTS.md. Commit that file to GitHub BEFORE running those models; the
commit time is the proof the prediction came first.

  python monitor.py --errors figs/error_types.csv figs/error_types_dense.csv
  python monitor.py --errors figs/error_types.csv figs/error_types_dense.csv --predict pythia-6.9b pythia-12b
"""
import argparse
import datetime
import hashlib
import json

import numpy as np
import pandas as pd

from stages import PARAMS, events, is_seed_run, load, size_key

TOKENS_PER_STEP = 2_097_152
SIGNALS = ("S_err", "S_copy", "B_acc")


def lead(t_signal, t_cor):
    """Doublings of training tokens between the signal and competence (positive = warning).
    NaN if either never fired, or if one fired at step 0 (0 tokens has no logarithm)."""
    if t_signal is None or t_cor is None or t_signal <= 0 or t_cor <= 0:
        return np.nan
    return float(np.log2(t_cor / t_signal))


def evaluate(df):
    rows = []
    for (model, items, task), g in df.groupby(["model", "items", "task"]):
        ev = events(g)
        t_cor = ev["T_cor"][1]
        row = dict(model=model, task=task, T_cor_step=ev["T_cor"][0])
        for s in SIGNALS:
            row[f"{s}_step"] = ev[s][0]
            row[f"{s}_lead"] = lead(ev[s][1], t_cor)
            row[f"{s}_fired_no_competence"] = ev[s][0] is not None and t_cor is None
        for s in ("S_err", "S_copy"):
            row[f"{s}_minus_baseline"] = row[f"{s}_lead"] - row["B_acc_lead"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(ev):
    print("\n=== signal vs baseline (lead in doublings of tokens before competence) ===")
    has_cor = ev["T_cor_step"].notna()
    print(f"runs x tasks: {len(ev)}, with competence reached: {int(has_cor.sum())}")
    for s in SIGNALS:
        leads = ev.loc[has_cor, f"{s}_lead"].dropna()
        fa = int(ev[f"{s}_fired_no_competence"].sum())
        med = f"{leads.median():.2f}" if len(leads) else "n/a"
        print(f"{s:7s} fired-before-competence n={len(leads)}  median lead={med}  "
              f"fired but no competence by end of training={fa}")
    for s in ("S_err", "S_copy"):
        diff = ev.loc[has_cor, f"{s}_minus_baseline"].dropna()
        if len(diff):
            print(f"{s} earlier than baseline in {int((diff > 0).sum())}/{len(diff)} runs, "
                  f"same checkpoint in {int((diff == 0).sum())}, later in {int((diff < 0).sum())}; "
                  f"median extra warning = {diff.median():.2f} doublings")


def predict(ev, targets):
    base = ev[~ev["model"].map(is_seed_run)]
    out = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               note="Rules and thresholds fixed in EXPERIMENTS.md. Commit before running targets.",
               rules={}, predictions={})

    # Rule A: per depth, straight line through log(T_cor tokens) vs log(params) over base models.
    out["rules"]["A"] = "per depth: least-squares line of log T_cor vs log params (base models with T_cor), extrapolated"
    # Rule B: T_cor = T_err * 2^m, m = median observed lead of S_err, range = min..max.
    m = base["S_err_lead"].dropna()
    out["rules"]["B"] = ("T_cor = T_err x 2^m, m = median S_err lead over base models with competence; "
                         f"m = {m.median():.2f}, range {m.min():.2f}..{m.max():.2f}, n = {len(m)}"
                         if len(m) else "no data")
    for target in targets:
        pred = {}
        for task, g in base.groupby("task"):
            pts = [(PARAMS[size_key(r.model)], r.T_cor_step * TOKENS_PER_STEP)
                   for r in g.itertuples() if pd.notna(r.T_cor_step) and size_key(r.model) in PARAMS]
            entry = dict(n_points=len(pts))
            if len(pts) >= 2:
                x, y = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
                slope, icpt = np.polyfit(x, y, 1)
                tok = float(np.exp(icpt + slope * np.log(PARAMS[target])))
                entry.update(slope=float(slope), rule_A_tokens=tok,
                             rule_A_step=int(round(tok / TOKENS_PER_STEP / 1000) * 1000))
            if len(m):
                entry["rule_B"] = (f"once {target} shows S_err at tokens T, predict T_cor at "
                                   f"T x {2 ** m.median():.1f} (range x{2 ** m.min():.1f}..x{2 ** m.max():.1f})")
            pred[task] = entry
        out["predictions"][target] = pred
    text = json.dumps(out, indent=2)
    with open("predictions.json", "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nsha256(predictions.json) = {hashlib.sha256((text + chr(10)).encode()).hexdigest()}")
    print("Write this hash + today's date in the data book, then commit and push predictions.json.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--errors", nargs="+", required=True)
    ap.add_argument("--out", default="figs/monitor.csv")
    ap.add_argument("--predict", nargs="*", default=[], help="model sizes to predict, e.g. pythia-6.9b pythia-12b")
    args = ap.parse_args()
    df = load(args.errors)
    ev = evaluate(df)
    ev.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(ev[["model", "task", "S_copy_step", "S_err_step", "B_acc_step", "T_cor_step",
                  "S_err_minus_baseline", "S_copy_minus_baseline"]].to_string(index=False))
    summarize(ev)
    if args.predict:
        unknown = [t for t in args.predict if t not in PARAMS]
        if unknown:
            raise SystemExit(f"unknown size(s) {unknown}; known: {sorted(PARAMS)}")
        predict(ev, args.predict)


if __name__ == "__main__":
    main()
