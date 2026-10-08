"""Order-Flip Test on a frontier model through its API, medication-record wording (claim C29). No GPU needed.

  export GEMINI_API_KEY=...        (or OPENAI_API_KEY=... / ANTHROPIC_API_KEY=...)
  python3 frontier_med.py --provider gemini --model <model-name>
  python3 frontier_med.py --selftest

Same items, prompts, modes (direct / cot) and answer parser as gen_test.py (depths 3 and 6, 100 questions each,
both record orders). Every reply is saved to frontier_med/<model>.json; re-running resumes from that file.
"""
import argparse
import json
import os

import gen_test as gt
from frontier_flip import _post, is_reasoning
from story_task import make_story


def ask(provider, model, prompt, mode):
    big = mode == "cot"
    if provider == "openai":
        body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        body.update({"max_completion_tokens": 8000} if is_reasoning(model) else {"temperature": 0, "max_tokens": 600 if big else 16})
        r = _post("https://api.openai.com/v1/chat/completions",
                  {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"}, body)
        return r["choices"][0]["message"]["content"] or ""
    if provider == "anthropic":
        r = _post("https://api.anthropic.com/v1/messages",
                  {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"},
                  {"model": model, "max_tokens": 600 if big else 16, "temperature": 0,
                   "messages": [{"role": "user", "content": prompt}]})
        return "".join(b.get("text", "") for b in r["content"])
    if provider == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={os.environ['GEMINI_API_KEY']}"
        r = _post(url, {"Content-Type": "application/json"},
                  {"contents": [{"parts": [{"text": prompt}]}],
                   "generationConfig": {"temperature": 0, "maxOutputTokens": 4096 if big else 1024}})
        c = r.get("candidates") or [{}]
        return "".join(p.get("text", "") for p in c[0].get("content", {}).get("parts", []))
    raise SystemExit(f"unknown provider {provider}")


def prompt_for(item, key, mode):
    return f"{gt.question_block(item, key)}\n{gt.DIRECT if mode == 'direct' else gt.COT}"


def run(ask_fn, n, depths, modes, path):
    res = json.load(open(path)) if os.path.exists(path) else {"tasks": {}}
    for d in depths:
        items = make_story(n, d, seed=7, theme="med")              # the same items as gen_test.py
        gold = [it["meta"]["answer"] for it in items]
        for mode in modes:
            key = f"{mode}_d{d}"
            raw = res["tasks"].get(key, {}).get("raw", {"orig": [], "swapped": []})
            for i, it in enumerate(items):
                if i < len(raw["orig"]) and i < len(raw["swapped"]):
                    continue
                raw["orig"] = raw["orig"][:i] + [ask_fn(prompt_for(it, "prompt", mode), mode)]
                raw["swapped"] = raw["swapped"][:i] + [ask_fn(prompt_for(it, "prompt_swapped", mode), mode)]
                res["tasks"][key] = dict(raw=dict(raw, gold=gold))
                if i % 10 == 9:
                    json.dump(res, open(path, "w"))
                    print(f"  {key} {i + 1}/{n}", flush=True)
            r = gt.score([gt.parse_dose(t) for t in raw["orig"]], [gt.parse_dose(t) for t in raw["swapped"]], gold)
            r["raw"] = dict(raw, gold=gold)
            res["tasks"][key] = r
            json.dump(res, open(path, "w"))
            print(key, {k: v for k, v in r.items() if k != "raw"}, flush=True)
    return res


def selftest():
    import tempfile
    calls = []

    def fake(prompt, mode):                       # always answers the dose of the LAST patient record shown
        calls.append(mode)
        last = prompt.split("\nQ:")[0].split("\n")[-1]
        return f"Answer: {last.split(' takes ')[1].split(' mg')[0]} mg" if " mg of " in last else "Answer: 1 mg"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.json")
        r = run(fake, 20, [3], ["direct"], p)
        t = r["tasks"]["direct_d3"]
        assert t["unparsed"] == 0 and t["n"] == 20 and len(calls) == 40
        r = run(fake, 20, [3], ["direct"], p)             # resume: nothing re-asked
        assert len(calls) == 40
    print("frontier med selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=("gemini", "openai", "anthropic"))
    ap.add_argument("--model")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--depths", type=int, nargs="+", default=[3, 6])
    ap.add_argument("--modes", nargs="+", default=["direct", "cot"])
    ap.add_argument("--out", default="frontier_med")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.provider and a.model):
        raise SystemExit("--provider and --model are required")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, a.model.replace("/", "_") + ".json")
    res = run(lambda p, m: ask(a.provider, a.model, p, m), a.n, a.depths, a.modes, path)
    res.update(provider=a.provider, model=a.model)
    json.dump(res, open(path, "w"))


if __name__ == "__main__":
    main()
