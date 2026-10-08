"""One-time additive edit of steer_stage.py (exploratory options P3 and P4; defaults unchanged).

  --save-items   also store, per held-out item: the model's natural log p(answer) - log p(distractor)
                 (its confidence in the chain choice) and whether planting worked (P3: does planting flip
                 near-ties more than confident questions?)
  --rank-sweep   also plant inside only the top-r directions of the primary subspace, r = 1..rank (P4: how
                 many directions are enough?)

Without the flags the script behaves exactly as before. Run once in the repo:  python3 patch_steer_stage.py
Nothing is changed unless every anchor is found exactly once.
"""
SRC = "steer_stage.py"
EDITS = [
    ('''    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
''',
     '''    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-items", action="store_true", help="also store per-item margins and outcomes (P3)")
    ap.add_argument("--rank-sweep", action="store_true", help="also plant inside the top-r components, r=1..rank (P4)")
'''),
    ('''    conditions.update({f"random{k}": erase(lambda G, k=k: G["Q_rand"][k]) for k in range(args.n_rand)})
''',
     '''    conditions.update({f"random{k}": erase(lambda G, k=k: G["Q_rand"][k]) for k in range(args.n_rand)})
    if args.rank_sweep:
        for r in range(1, rank + 1):
            for l in geo:
                geo[l][f"Q_orth_r{r}"] = geo[l]["Q_orth"][:, :r]
            conditions[f"plant_dis_orth_r{r}"] = plant(f"Q_orth_r{r}", "cm_dis")
'''),
    ('''    os.makedirs(args.out, exist_ok=True)
    tag = iou.tag(args.seed, args.n, args.shots, args.layout)
''',
     '''    if args.save_items:
        lo_nat, _ = mu.run(tok, model, [it["prompt"] for it in test], cids, batch_size=args.batch_size,
                           device=args.device)
        lp_nat = torch.log_softmax(lo_nat, -1).numpy()
        ar = np.arange(len(test))
        summary["items"] = dict(
            margin_natural=(lp_nat[ar, labels] - lp_nat[ar, dis]).tolist(),
            correct_natural=res["natural"]["correct"].tolist(),
            planted_natural=res["natural"]["planted"].tolist(),
            planted_plant_dis=res["plant_dis"]["planted"].tolist(),
            planted_plant_dis_orth=res["plant_dis_orth"]["planted"].tolist())
    os.makedirs(args.out, exist_ok=True)
    tag = iou.tag(args.seed, args.n, args.shots, args.layout)
'''),
]
MARK = "--rank-sweep"
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
