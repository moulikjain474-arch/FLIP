"""Order-Flip Test on a real benchmark: the same multiple-choice questions with the options
in different orders (MMLU, 0-shot, base models). The correct option never changes, so any
answer that depends on the order is a flip.

  python benchmark_flip.py --model allenai/OLMo-2-1124-7B --out benchmark
  python benchmark_flip.py --selftest        # no model or data needed

Reports, per model: accuracy in each order, flip rate (correct in exactly one of original
and reversed order), and accuracy when answers from 2 orders (original + reversed) or all
4 cyclic orders are combined by summing log-probabilities.
"""
import argparse
import json
import os

import numpy as np

import flipstats as fs

LETTERS = "ABCD"
ORDERS = {"orig": [0, 1, 2, 3], "reversed": [3, 2, 1, 0], "shift1": [1, 2, 3, 0], "shift2": [2, 3, 0, 1]}


def make_prompt(subject, question, choices, order):
    """order[i] = which original option is shown as letter i."""
    body = "\n".join(f"{LETTERS[i]}. {choices[j]}" for i, j in enumerate(order))
    return (f"The following is a multiple choice question about {subject.replace('_', ' ')}.\n\n"
            f"{question.strip()}\n{body}\nAnswer:")


def to_original(logp_shown, order):
    """Log-probs over the letters as shown -> log-probs over the original options."""
    out = np.empty_like(logp_shown)
    for i, j in enumerate(order):
        out[:, j] = logp_shown[:, i]
    return out


def letter_ids(tok):
    """Token id of each answer letter as it follows 'Answer:' (must be exactly one token)."""
    base = tok("Answer:", add_special_tokens=False)["input_ids"]
    ids = []
    for L in LETTERS:
        full = tok(f"Answer: {L}", add_special_tokens=False)["input_ids"]
        if full[:len(base)] != base or len(full) != len(base) + 1:
            raise ValueError(f"letter {L!r} is not a single token after 'Answer:': {full} vs {base}")
        ids.append(full[-1])
    return ids


def arc_rows(raw):
    """ARC-Challenge rows -> the MMLU row shape (subject, question, choices, answer index). Keeps only questions
    with exactly four options whose answer key is A-D or 1-4 (about 97% of the test split)."""
    out = []
    for r in raw:
        labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
        key = str(r["answerKey"]).strip()
        if len(texts) != 4 or key not in labels:
            continue
        out.append(dict(subject="science", question=r["question"], choices=texts, answer=labels.index(key)))
    return out


def selftest():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 50)
    for name, order in ORDERS.items():
        assert sorted(order) == [0, 1, 2, 3]
        shown = np.full((50, 4), -5.0)
        for n in range(50):                       # a perfect model puts its mass on the right option
            shown[n, order.index(y[n])] = -0.1
        assert (to_original(shown, order).argmax(-1) == y).all(), name
    p = make_prompt("high_school_physics", " Q? ", ["w", "x", "y", "z"], ORDERS["reversed"])
    assert "A. z\nB. y\nC. x\nD. w\nAnswer:" in p
    raw = [dict(question="q1", choices=dict(label=["A", "B", "C", "D"], text=["a", "b", "c", "d"]), answerKey="C"),
           dict(question="q2", choices=dict(label=["1", "2", "3", "4"], text=["a", "b", "c", "d"]), answerKey="2"),
           dict(question="q3", choices=dict(label=["A", "B", "C"], text=["a", "b", "c"]), answerKey="A"),
           dict(question="q4", choices=dict(label=["A", "B", "C", "D", "E"], text=list("abcde")), answerKey="E")]
    got = arc_rows(raw)
    assert [g["question"] for g in got] == ["q1", "q2"] and [g["answer"] for g in got] == [2, 1]
    assert all(g["subject"] == "science" and len(g["choices"]) == 4 for g in got)
    print("benchmark selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--bos", action="store_true", help="prepend the BOS token (needed for Mistral)")
    ap.add_argument("--dataset", choices=("mmlu", "arc"), default="mmlu",
                    help="mmlu = cais/mmlu test (default); arc = ARC-Challenge test, four-option questions only")
    ap.add_argument("--out", default="benchmark_results")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.model:
        raise SystemExit("--model is required")

    import torch
    from datasets import load_dataset
    import model_utils as mu

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = mu.load(args.model, args.revision, dev, torch.bfloat16 if dev == "cuda" else torch.float32)
    ids = letter_ids(tok)
    if args.dataset == "arc":
        pool = arc_rows(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"))
    else:
        pool = load_dataset("cais/mmlu", "all", split="test")
    n = min(args.n, len(pool))
    pick = np.random.default_rng(args.seed).choice(len(pool), n, replace=False)
    rows = [pool[int(i)] for i in pick]
    y = np.array([r["answer"] for r in rows])
    prefix = (tok.bos_token or "") if args.bos else ""

    logp, correct = {}, {}
    for name, order in ORDERS.items():
        prompts = [prefix + make_prompt(r["subject"], r["question"], r["choices"], order) for r in rows]
        if name == "orig":
            print(prompts[0], flush=True)
        logits, _ = mu.run(tok, model, prompts, ids, batch_size=args.batch_size, device=dev)
        shown = torch.log_softmax(logits.float(), -1).numpy()
        logp[name] = to_original(shown, order)
        correct[name] = logp[name].argmax(-1) == y
        print(name, "accuracy", round(float(correct[name].mean()), 3), flush=True)

    ens2 = fs.ensemble_correct([logp["orig"], logp["reversed"]], y)
    ens4 = fs.ensemble_correct(list(logp.values()), y)
    single2 = (correct["orig"].astype(float) + correct["reversed"]) / 2
    summary = {f"acc_{k}": float(v.mean()) for k, v in correct.items()}
    summary.update(flip_orig_reversed=fs.flip_rate(correct["orig"], correct["reversed"]),
                   acc_single_mean2=float(single2.mean()), acc_ens2=float(ens2.mean()), acc_ens4=float(ens4.mean()),
                   acc_mean4=float(np.mean([c.mean() for c in correct.values()])), n=int(len(y)))
    gain = fs.paired_bootstrap(ens2, single2)
    summary.update(gain2=gain[0], gain2_lo=gain[1], gain2_hi=gain[2])
    print(json.dumps(summary, indent=2))
    os.makedirs(args.out, exist_ok=True)
    name = args.model.split("/")[-1] + (f"__{args.revision}" if args.revision else "") + ("__arc" if args.dataset == "arc" else "")
    with open(os.path.join(args.out, f"{name}.json"), "w") as f:
        json.dump(dict(model=args.model, revision=args.revision, dataset=args.dataset, summary=summary,
                       correct={k: v.astype(int).tolist() for k, v in correct.items()},
                       ens2=ens2.astype(int).tolist(), ens4=ens4.astype(int).tolist()), f)


if __name__ == "__main__":
    main()
