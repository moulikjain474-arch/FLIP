"""Node-side sanity check for claim C23: a different --seed must give a different question set,
and the same seed must give the same one. No model is loaded.

  python check_seed.py
"""
from tasks import build_all


def main():
    a, a2, b = build_all(60, 0, shots=2, layout="blocked"), build_all(60, 0, shots=2, layout="blocked"), \
        build_all(60, 1, shots=2, layout="blocked")
    for t in ("vartrack_d1", "vartrack_d2", "vartrack_d3"):
        pa, pa2, pb = ([it["prompt"] for it in x[t]] for x in (a, a2, b))
        assert pa == pa2, f"{t}: the same seed gave different questions"
        same = sum(x == y for x, y in zip(pa, pb)) / len(pa)
        assert same < 0.05, f"{t}: seeds 0 and 1 share {same:.0%} of their questions"
        print(f"{t}: seed 0 reproducible; seed 1 shares {same:.0%} of its questions with seed 0")
    print("SEED CHECK OK")


if __name__ == "__main__":
    main()
