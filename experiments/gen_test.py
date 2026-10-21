"""Order-Flip Test with GENERATED answers (claim C28): does writing out reasoning remove the order effect?

  python gen_test.py --model Qwen/Qwen2.5-7B-Instruct --out gen
  python gen_test.py --selftest

Uses the medication-record wording (story_task.py, theme="med"), zero-shot, through the model's chat template when
it has one. Two modes, greedy decoding:
  direct  "Answer with only the number of mg."                         (max 16 new tokens)
  cot     "Think step by step, then end with 'Answer: N mg'."          (max --cot-tokens new tokens)
Each question is asked in both record orders. The final dose is parsed from the text (see parse_dose); raw outputs
are saved so the parsing can be checked by hand. Reports per mode and depth: accuracy in each order, flip rate
(correct in exactly one order), unparsed share, and how often the two answers differ.
"""
import argparse
import json
import os
import re

import numpy as np

DIRECT = "Answer with only the number of mg."
COT = "Think step by step, then end with a final line of the form 'Answer: N mg'."


def question_block(item, key):
    """The last block of a story prompt (the two records and the question), without the few-shot examples."""
    block = item[key].rsplit("\n\n", 1)[1]
    return block.rsplit("\nA:", 1)[0].strip()


def parse_dose(text):
    """Final dose in a generated answer, as a string, or None. Order of preference: the text after a closing
    </think>; then 'Answer: N'; then the last 'N mg'; then the last standalone number."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    for pat in (r"[Aa]nswer\s*[:=]?\s*\**\s*(\d+)", r"(\d+)\s*mg", r"\b(\d+)\b"):
        m = re.findall(pat, text)
        if m:
            return m[-1]
    return None


def score(answers_a, answers_b, gold):
    a, b, g = np.array(answers_a, dtype=object), np.array(answers_b, dtype=object), np.array(gold, dtype=object)
    ca, cb = a == g, b == g
    return dict(acc_orig=float(ca.mean()), acc_swapped=float(cb.mean()), flip=float((ca != cb).mean()),
                different_answer=float((a != b).mean()),
                unparsed=float(((a == None) | (b == None)).mean()), n=int(len(g)))  # noqa: E711


def selftest():
    assert parse_dose("4") == "4" and parse_dose(" 7 mg") == "7"
    assert parse_dose("Ravi takes 3 mg, so Mia takes 3 mg.\nAnswer: 3 mg") == "3"
    assert parse_dose("<think>maybe 5 mg... no, 6</think>\nThe dose is 8 mg.") == "8"
    assert parse_dose("**Answer: 9 mg**") == "9" and parse_dose("I am not sure.") is None
    s = score(["1", "2", None, "4"], ["1", "3", "3", "4"], ["1", "2", "3", "5"])
    assert s["acc_orig"] == 0.5 and s["acc_swapped"] == 0.5 and s["flip"] == 0.5 and s["unparsed"] == 0.25
    from story_task import make_story
    it = make_story(3, 3, theme="med")[0]
    qa, qb = question_block(it, "prompt"), question_block(it, "prompt_swapped")
    assert qa != qb and qa.endswith("take?") and qa.count("Patient") == 8 and "\n\n" not in qa
    print("gen selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--depths", type=int, nargs="+", default=[3, 6])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--modes", nargs="+", default=["direct", "cot"])
    ap.add_argument("--cot-tokens", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="gen")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.model:
        raise SystemExit("--model is required")

    import torch
    import model_utils as mu
    from story_task import make_story

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = mu.load(args.model, args.revision, dev, torch.bfloat16 if dev == "cuda" else torch.float32)
    chat = getattr(tok, "chat_template", None) is not None

    def render(q, mode):
        msg = f"{q}\n{DIRECT if mode == 'direct' else COT}"
        if chat:
            return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)
        return msg + "\n"

    @torch.no_grad()
    def generate(prompts, max_new):
        outs = []
        for i in range(0, len(prompts), args.batch_size):
            enc = tok(prompts[i:i + args.batch_size], return_tensors="pt", padding=True,
                      add_special_tokens=not chat).to(dev)
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"  {i + len(prompts[i:i + args.batch_size])}/{len(prompts)}", flush=True)
        return outs

    res = dict(model=args.model, revision=args.revision, chat_template=chat, tasks={})
    for d in args.depths:
        items = make_story(args.n, d, seed=7, theme="med")          # seed 7: not the log-prob test's items
        gold = [it["meta"]["answer"] for it in items]
        for mode in args.modes:
            max_new = 16 if mode == "direct" else args.cot_tokens
            ta = generate([render(question_block(it, "prompt"), mode) for it in items], max_new)
            tb = generate([render(question_block(it, "prompt_swapped"), mode) for it in items], max_new)
            pa, pb = [parse_dose(t) for t in ta], [parse_dose(t) for t in tb]
            r = score(pa, pb, gold)
            r["examples"] = [dict(gold=g, orig=x, swapped=y) for g, x, y in list(zip(gold, ta, tb))[:5]]
            r["raw"] = dict(orig=ta, swapped=tb, gold=gold)
            res["tasks"][f"{mode}_d{d}"] = r
            print(f"{mode}_d{d}", {k: v for k, v in r.items() if k not in ("examples", "raw")}, flush=True)
            print("  example:", repr(ta[0][-300:]), flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.model.split('/')[-1]}.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
