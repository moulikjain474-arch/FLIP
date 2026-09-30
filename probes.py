"""Score every saved (model, checkpoint, task) file produced by sweep.py.

All measures use the same items, the same 5-fold split and the same classifier.

10-way (or K-way) measures, chance = 1/K:
  behavior        the model's own zero-shot choice (argmax over the K candidates)
  logit_decoder   classifier on the model's K candidate logits
                  ("what the output layer already knows"; the probe must beat it)
  probe_main      classifier on the middle-layer residual stream (pre-registered layer)
  probe_shuffled  same probe on shuffled labels (should sit at chance)
  probe_L{l}      probes at every recorded layer (non-main layers only with --all-layers;
                  used for the processing-depth figure)
  keyword         bag-of-words classifier on the prompt text (surface cues)

Target-vs-distractor measures, variable-tracking tasks only, chance = 0.5:
  *_td            "is the TARGET chain's value scored above the DISTRACTOR's?"
  A model that only knows which digits appear in the prompt gets ~50% here, so
  these are the measures used for the main depth result (H1).

The step-0 checkpoint is the untrained-network control for every measure.

  python probes.py --runs runs/ --out results.csv            # main layer only
  python probes.py --runs runs/ --out results.csv --all-layers
"""
import argparse
import glob
import os
import zlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import io_utils as iou

C = 0.1        # pre-registered probe regularisation: never tune this on results
FOLDS = 5
N_BOOT = 1000


def boot_ci(correct, rng, n=N_BOOT):
    """95% percentile bootstrap interval for mean accuracy (resampling items)."""
    idx = rng.integers(0, len(correct), size=(n, len(correct)))
    means = correct[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cv_predict(X, y, K, seed=0):
    """Out-of-fold class probabilities from a standardised logistic probe."""
    proba = np.zeros((len(y), K))
    folds = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
    for tr, te in folds.split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        clf.fit(X[tr], y[tr])
        proba[np.ix_(te, clf.classes_)] = clf.predict_proba(X[te])
    return proba


def row(meta, measure, layer, correct, rng, K=None, logloss=np.nan):
    lo, hi = boot_ci(correct, rng)
    m = dict(meta) if K is None else dict(meta, K=K)
    return dict(**m, measure=measure, layer=layer, acc=float(correct.mean()), lo=lo, hi=hi,
                logloss=logloss)


def kway(meta, measure, layer, y, rng, proba=None, pred=None):
    """K-way accuracy row (log-loss too when probabilities are available)."""
    ll = np.nan
    if proba is not None:
        pred = proba.argmax(1)
        ll = log_loss(y, np.clip(proba, 1e-6, 1), labels=list(range(proba.shape[1])))
    return row(meta, measure, layer, (pred == y).astype(float), rng, logloss=ll)


def td(meta, measure, layer, scores, a, b, rng):
    """Target-vs-distractor accuracy: score of the target digit > score of the distractor."""
    i = np.arange(len(a))
    return row(meta, measure + "_td", layer, (scores[i, a] > scores[i, b]).astype(float), rng, K=2)


def td_indices(items):
    digits = items[0]["candidates"]
    a = np.array([digits.index(it["meta"]["answer"]) for it in items])
    b = np.array([digits.index(it["meta"]["distractor"]) for it in items])
    return a, b


def score_file(path, all_layers):
    model, rev, item_tag, task = iou.parse_result_name(path)
    rng = np.random.default_rng(zlib.crc32(os.path.basename(path).encode()))  # reproducible
    d = np.load(path)
    y, logits = d["labels"], d["logits"]
    layers = [int(l) for l in d["layers"]]
    K = logits.shape[1]
    main_layer = max(layers) // 2
    meta = dict(model=model, step=int(d["step"]), tokens=int(d["tokens"]), task=task,
                items=item_tag, K=K, n_layers=max(layers))

    td_ab = None
    if task.startswith("vartrack"):
        items = iou.load_items(os.path.dirname(path), item_tag)[task]
        assert [it["label"] for it in items] == y.tolist(), f"item/label mismatch in {path}"
        td_ab = td_indices(items)

    decoder = cv_predict(logits, y, K)
    rows = [kway(meta, "behavior", -1, y, rng, pred=logits.argmax(1)),
            kway(meta, "logit_decoder", -1, y, rng, proba=decoder)]
    if td_ab:
        rows += [td(meta, "behavior", -1, logits, *td_ab, rng),
                 td(meta, "logit_decoder", -1, decoder, *td_ab, rng)]

    for l in layers:
        if l != main_layer and not all_layers:
            continue
        X = d[f"acts_L{l}"].astype(np.float32)
        proba = cv_predict(X, y, K)
        rows.append(kway(meta, f"probe_L{l}", l, y, rng, proba=proba))
        if td_ab:
            rows.append(td(meta, f"probe_L{l}", l, proba, *td_ab, rng))
        if l == main_layer:
            rows.append(kway(meta, "probe_main", l, y, rng, proba=proba))
            if td_ab:
                rows.append(td(meta, "probe_main", l, proba, *td_ab, rng))
            y_shuf = rng.permutation(y)
            rows.append(kway(meta, "probe_shuffled", l, y_shuf, rng, proba=cv_predict(X, y_shuf, K)))
    print(f"done {model} {rev} {task}", flush=True)
    return rows


def keyword_rows(runs):
    """Bag-of-words on the prompt text. It does not depend on the checkpoint: once per item set."""
    rows, rng = [], np.random.default_rng(0)
    for path in sorted(glob.glob(os.path.join(runs, "items_*.json"))):
        item_tag = os.path.basename(path)[len("items_"):-len(".json")]
        for task, items in iou.load_items(runs, item_tag).items():
            y = np.array([i["label"] for i in items])
            K = len(items[0]["candidates"])
            vec = CountVectorizer(token_pattern=r"\S+")
            X = vec.fit_transform(i["prompt"] for i in items).toarray().astype(np.float32)
            proba = cv_predict(X, y, K)
            meta = dict(model="-", step=-1, tokens=-1, task=task, items=item_tag, K=K)
            rows.append(kway(meta, "keyword", -1, y, rng, proba=proba))
            if task.startswith("vartrack"):
                rows.append(td(meta, "keyword", -1, proba, *td_indices(items), rng))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--all-layers", action="store_true", help="also probe the non-main layers")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel workers (-1 = all CPU cores)")
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(args.runs, "*.npz")) if iou.is_result(f))
    if not files:
        raise SystemExit(f"no result files in {args.runs!r}; run sweep.py first")
    per_file = Parallel(n_jobs=args.jobs)(delayed(score_file)(f, args.all_layers) for f in files)
    rows = keyword_rows(args.runs) + [r for rs in per_file for r in rs]
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(rows)} rows from {len(files)} files)")


if __name__ == "__main__":
    main()
