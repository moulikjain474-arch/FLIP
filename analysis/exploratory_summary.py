"""Position decomposition (exploratory, post hoc) and the C22 confidence gate, from saved JSONs.
  python exploratory_summary.py --dir results_08Oct/merged"""
import argparse, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
M = ap.parse_args().dir
print("=== DECOMPOSITION (exploratory, written after seeing C22 at 12B) ===")
for f in sorted(glob.glob(os.path.join(M, "decomp", "*.json"))):
    d = json.load(open(f)); print(d["model"].split("/")[-1], d["revision"])
    for t, v in d["tasks"].items():
        s = v["decomposition"]["share"]
        print("  ", t, " ".join(f"{k} {s[k]:.2f}" for k in s), "| position-driven %.2f" % v["decomposition"]["position_driven"],
              "| by confidence low->high", [round(c["position_driven"], 2) for c in v["decomposition"].get("by_confidence", [])])
print("=== C22 CONFIDENCE GATE (registered; flip rate by confidence third, low->high) ===")
for f in sorted(glob.glob(os.path.join(M, "uncertainty", "*.json"))):
    d = json.load(open(f)); print(os.path.basename(f))
    for t, v in d["tasks"].items():
        g = v["summary"]["gate_orig_margin"]
        print("  ", t, [round(x, 3) for x in g["group_flip"]], "diff %+.3f [%+.3f,%+.3f]" % (g["diff"], g["diff_lo"], g["diff_hi"]),
              "ratio", g["ratio"], "p=%.2g" % g["fisher_p"])
