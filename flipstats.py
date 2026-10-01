"""Statistics for the order-flip tests (pure numpy)."""
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


def pair_metrics(logp_a, logp_b, y):
    """Order-Flip summary for one question set asked in two orders (log-probs [N, K])."""
    y = np.asarray(y)
    ca, cb = logp_a.argmax(-1) == y, logp_b.argmax(-1) == y
    both = ensemble_correct([logp_a, logp_b], y)
    single = (ca.astype(float) + cb) / 2
    g = paired_bootstrap(both, single)
    return dict(acc_orig=float(ca.mean()), acc_swapped=float(cb.mean()), acc_single_mean=float(single.mean()),
                acc_both=float(both.mean()), gain=g[0], gain_lo=g[1], gain_hi=g[2], flip=flip_rate(ca, cb),
                same_answer=float((logp_a.argmax(-1) == logp_b.argmax(-1)).mean()), n=int(len(y)))
