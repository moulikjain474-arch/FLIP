"""Statistics for the order-flip tests (pure numpy)."""
import math

import numpy as np


def flip_rate(correct_a, correct_b):
    """Share of questions answered correctly in exactly one of two versions."""
    return float((np.asarray(correct_a, bool) != np.asarray(correct_b, bool)).mean())


def ensemble_correct(logps, y):
    """Per-question correctness of the answer that maximises the summed log-probabilities."""
    return (sum(logps).argmax(-1) == np.asarray(y))


def paired_bootstrap(x, y, n_boot=10000, seed=0):
    """Mean of x - y per question with a 95% bootstrap interval: (mean, low, high)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    means = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(mid - half), float(mid + half)


def fisher_exact(a, b, c, d):
    """Two-sided exact Fisher test on [[a, b], [c, d]]. Returns (odds_ratio, p). Pure python."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    den = math.comb(n, c1)
    prob = lambda x: math.comb(r1, x) * math.comb(r2, c1 - x) / den
    lo, hi = max(0, c1 - r2), min(r1, c1)
    p_obs = prob(a)
    p = sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs * (1 + 1e-9))
    odds = (a * d) / (b * c) if b * c else float("inf") if a * d else float("nan")
    return float(odds), float(min(1.0, p))


def mcnemar_exact(fixed, broke):
    """Exact two-sided McNemar: of fixed+broke discordant questions, is the split 50/50?"""
    n = fixed + broke
    if n == 0:
        return 1.0
    k = min(fixed, broke)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, p))


def order_stats(ca, cb):
    """Strict / lenient accuracy, three-way split, chance flip rate and Fisher test for one question set
    asked in two orders (ca, cb = per-question correctness). Descriptive, added after pre-registration."""
    ca, cb = np.asarray(ca, bool), np.asarray(cb, bool)
    n = len(ca)
    both_right = float((ca & cb).mean())
    both_wrong = float((~ca & ~cb).mean())
    flip = float((ca != cb).mean())
    p1, p2 = float(ca.mean()), float(cb.mean())
    chance = p1 * (1 - p2) + p2 * (1 - p1)
    odds, p = fisher_exact(int((ca & cb).sum()), int((ca & ~cb).sum()), int((~ca & cb).sum()), int((~ca & ~cb).sum()))
    return dict(strict=both_right, lenient=both_right + flip, both_right=both_right, both_wrong=both_wrong,
                flip_share=flip, flip_chance=float(chance), flip_minus_chance=flip - float(chance),
                fisher_odds=odds, fisher_p=p)


def pair_metrics(logp_a, logp_b, y):
    """Order-Flip summary for one question set asked in two orders (log-probs [N, K])."""
    y = np.asarray(y)
    ca, cb = logp_a.argmax(-1) == y, logp_b.argmax(-1) == y
    both = ensemble_correct([logp_a, logp_b], y)
    single = (ca.astype(float) + cb) / 2
    g = paired_bootstrap(both, single)
    fixed, broke = int((both & ~ca).sum()), int((~both & ca).sum())
    return dict(**order_stats(ca, cb), ens_fixed=fixed, ens_broke=broke, mcnemar_p=mcnemar_exact(fixed, broke),
                acc_orig=float(ca.mean()), acc_swapped=float(cb.mean()), acc_single_mean=float(single.mean()),
                acc_both=float(both.mean()), gain=g[0], gain_lo=g[1], gain_hi=g[2], flip=flip_rate(ca, cb),
                same_answer=float((logp_a.argmax(-1) == logp_b.argmax(-1)).mean()), n=int(len(y)))


def top2_margin(logp):
    """Per-question confidence: log-prob of the best candidate minus the second best ([N, K] -> [N])."""
    s = np.sort(np.asarray(logp, float), axis=-1)
    return s[:, -1] - s[:, -2]


def _ranks(x):
    """Average ranks (ties share the mean rank), pure numpy."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    """Spearman rank correlation; nan if either input is constant."""
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def confidence_gate(margin, flip, groups=3, n_boot=2000, seed=0):
    """Does the order effect concentrate where the model is unsure?

    Questions are sorted by confidence (margin, low to high; ties keep question order) and cut into
    `groups` near-equal groups. Reports the flip rate per group, the lowest- vs highest-confidence
    difference with a 95% bootstrap interval (groups recomputed in every resample), an exact Fisher
    test on the two extreme groups, the ratio low/high (None if the high group never flips) and the
    Spearman correlation between margin and flip (negative = flips concentrate at low confidence).
    Descriptive for any data; the registered claim C22 applies a fixed rule to it."""
    margin, flip = np.asarray(margin, float), np.asarray(flip, bool)
    assert len(margin) == len(flip) and len(flip) >= 2 * groups
    def rates(m, f):
        parts = np.array_split(np.argsort(m, kind="stable"), groups)
        return [float(f[p].mean()) for p in parts], [len(p) for p in parts], [int(f[p].sum()) for p in parts]
    r, n, k = rates(margin, flip)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(flip), (n_boot, len(flip)))
    diffs = np.array([(lambda rr: rr[0][0] - rr[0][-1])(rates(margin[i], flip[i])) for i in idx])
    odds, p = fisher_exact(k[0], n[0] - k[0], k[-1], n[-1] - k[-1])
    return dict(group_flip=r, group_n=n, low_flip=r[0], high_flip=r[-1], diff=r[0] - r[-1],
                diff_lo=float(np.percentile(diffs, 2.5)), diff_hi=float(np.percentile(diffs, 97.5)),
                ratio=(r[0] / r[-1]) if r[-1] > 0 else None, fisher_p=p,
                spearman_margin_flip=spearman(margin, flip.astype(float)))
