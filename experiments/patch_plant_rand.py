"""One-time additive edit of steer_stage.py: the random-subspace PLANT control (claim C24; defaults unchanged).

  --plant-rand   also plant the same distractor-value class means inside --n-rand random subspaces of the
                 primary rank (one per draw, same layers, same items). The summary gets
                 summary["plant_rand"] = {rate, plant_dis_minus_rand (paired test)} and the per-draw rates appear
                 in summary["rates"] as plant_rand0, plant_rand1, ...

Run once in the repo, AFTER patch_steer_stage.py:  python3 patch_plant_rand.py
Without the flag the script behaves exactly as before. Nothing is changed unless every anchor is found exactly once.
"""
SRC = "steer_stage.py"
EDITS = [
    ('''    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
''',
     '''    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--plant-rand", action="store_true",
                    help="also plant the distractor class means inside random subspaces of the primary rank (C24)")
'''),
    ('''    conditions.update({f"random{k}": erase(lambda G, k=k: G["Q_rand"][k]) for k in range(args.n_rand)})
''',
     '''    conditions.update({f"random{k}": erase(lambda G, k=k: G["Q_rand"][k]) for k in range(args.n_rand)})
    if args.plant_rand:
        for k in range(args.n_rand):
            for l in geo:
                geo[l][f"Q_rp{k}"] = geo[l]["Q_rand"][k]
            conditions[f"plant_rand{k}"] = plant(f"Q_rp{k}", "cm_dis")
'''),
    ('''    os.makedirs(args.out, exist_ok=True)
    tag = iou.tag(args.seed, args.n, args.shots, args.layout)
''',
     '''    if args.plant_rand:
        res["plant_rand_mean"] = dict(planted=np.mean(
            [res[f"plant_rand{k}"]["planted"] for k in range(args.n_rand)], axis=0))
        summary["plant_rand"] = dict(
            rate=float(res["plant_rand_mean"]["planted"].mean()),
            plant_dis_minus_rand=paired(res["plant_dis"]["planted"], res["plant_rand_mean"]["planted"]))
        print("plant_rand mean planted rate", round(summary["plant_rand"]["rate"], 3),
              "| plant_dis", round(float(res["plant_dis"]["planted"].mean()), 3), flush=True)
    os.makedirs(args.out, exist_ok=True)
    tag = iou.tag(args.seed, args.n, args.shots, args.layout)
'''),
]
MARK = "--plant-rand"
src = open(SRC).read()
if MARK in src:
    print(SRC, "already patched")
else:
    for old, _ in EDITS:
        assert src.count(old) == 1, f"anchor not found exactly once in {SRC}; nothing changed:\n{old}"
    for old, new in EDITS:
        src = src.replace(old, new)
    compile(src, SRC, "exec")                      # refuse to write a file that does not parse
    open(SRC, "w").write(src)
    print(SRC, "patched")
