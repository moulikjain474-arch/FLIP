"""Summarise head_pad_test.py (audit A3): does the first-or-last habit follow absolute position or distance?

  python3 headpad_report.py --dir $PERSIST/fullday/headpad

Per model and task: accuracy and the "picks the later chain" rate under four conditions (none; start64 = 64 no-op lines
at the very start; head8 = 8 lines before the first chain; between8 = 8 lines between the chains) and the change in the
later-chain rate against "none". A condition CHANGES the habit if the rate moves by >= 0.05 (fixed in advance).
  start64 or head8 changes it, between8 does not  -> follows absolute position
  between8 changes it, start64 and head8 do not   -> follows distance to the question / order
  both kinds change it                            -> both
  nothing changes it                              -> neither (at these paddings)
"""
import argparse
import glob
import json
import os

CHANGE = 0.05


def classify(delta):
    """delta = {condition: change in picks_later vs none}."""
    ch = {k: v is not None and abs(v) >= CHANGE for k, v in delta.items()}
    absolute = ch.get("start64", False) or ch.get("head8", False)
    distance = ch.get("between8", False)
    return ("both" if absolute and distance else "absolute position" if absolute
            else "distance / order" if distance else "neither")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    root = ap.parse_args().dir
    files = sorted(glob.glob(os.path.join(root, "*.json")))
    if not files:
        print("A3: NOT RUN")
        return
    for f in files:
        s = json.load(open(f))
        for t, conds in s["tasks"].items():
            base = conds["none0"]["picks_later"]["rate"]
            delta, cells = {}, []
            for key, name in (("none0", "none"), ("start64", "start64"), ("head8", "head8"), ("between8", "between8")):
                r = conds.get(key)
                if r is None:
                    continue
                rate = r["picks_later"]["rate"]
                if name != "none":
                    delta[name] = None if rate is None or base is None else rate - base
                cells.append(f"{name}: acc {r['accuracy']:.3f}, later {('%.3f' % rate) if rate is not None else '-'}"
                             + ("" if name == "none" or delta[name] is None else f" ({delta[name]:+.3f})"))
            print(f"{s['model'].split('/')[-1]} {s['revision']} {t}: " + " | ".join(cells) + f"  -> {classify(delta)}")


if __name__ == "__main__":
    main()
