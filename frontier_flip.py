"""Order-Flip Test on closed frontier models through their APIs (exploratory, behavioural).

Runs on any laptop (no GPU). The model sees K chains of variable assignments, written in one
order and then in the reverse order, and must reply with the value of one variable. Harder
than the main task (more chains, deeper) so that strong models are not at 100%.

  export OPENAI_API_KEY=...      (or ANTHROPIC_API_KEY=...)
  python3 frontier_flip.py --provider openai --model <model-name> --n 200
  python3 frontier_flip.py --selftest
Reports accuracy in each order, flip rate (correct in exactly one order), and how often the
two orders give the same answer. Every reply is saved, so the scoring can be re-checked.
"""
import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request

NAMES = list("abcdefghijklmnopqrstuvwxyz")
INSTRUCTION = ("Below is a short Python session. Reply with only the single digit that the last "
               "line prints, nothing else.\n\n")


def make_items(n, chains, depth, seed=0):
    """Each item: list of chains (each a list of lines), the query variable, the answer."""
    if chains * (depth + 1) > len(NAMES):
        raise SystemExit(f"{chains}x{depth} needs {chains * (depth + 1)} variable names; only {len(NAMES)} exist")
    rng = random.Random(seed * 1009 + chains * 31 + depth)
    items = []
    for _ in range(n):
        names = rng.sample(NAMES, chains * (depth + 1))
        values = rng.sample([str(d) for d in range(10)], chains)
        cs = []
        for c in range(chains):
            ns = names[c * (depth + 1):(c + 1) * (depth + 1)]
            cs.append([f">>> {ns[0]} = {values[c]}"] + [f">>> {ns[i]} = {ns[i - 1]}" for i in range(1, depth + 1)])
        rng.shuffle(cs)
        target = rng.randrange(chains)
        q = cs[target][-1].split(" = ")[0][4:]
        items.append(dict(chains=cs, query=q, answer=cs[target][0].split(" = ")[1], target=target))
    return items


def render(item, reverse):
    cs = item["chains"][::-1] if reverse else item["chains"]
    return INSTRUCTION + "\n".join(l for c in cs for l in c) + f"\n>>> {item['query']}"


CODE_INSTRUCTION = "Reply with only the single digit that this program prints, nothing else.\n\n"


def code_prompts(n, level):
    from code_task import make_code
    items = make_code(n, level)
    strip = lambda p: p.rsplit("\n\n", 1)[1].replace("\nOutput:\n", "")   # the program alone, no examples
    return [dict(answer=it["meta"]["answer"], fwd=CODE_INSTRUCTION + strip(it["prompt"]),
                 rev=CODE_INSTRUCTION + strip(it["prompt_swapped"])) for it in items]


def parse(reply):
    m = re.search(r"\d", reply or "")
    return m.group(0) if m else None


def _post(url, headers, body, tries=6):
    data = json.dumps(body).encode()
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and k < tries - 1:
                time.sleep(2 ** k); continue
            raise RuntimeError(f"API error {e.code}: {e.read()[:300]!r}")
        except (urllib.error.URLError, TimeoutError):
            if k < tries - 1:
                time.sleep(2 ** k); continue
            raise RuntimeError("network error after retries")


def is_reasoning(model):
    """OpenAI reasoning models reject temperature/max_tokens and need room to think."""
    return model.startswith(("o1", "o3", "o4", "gpt-5"))


def ask(provider, model, prompt):
    if provider == "openai":
        body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        body.update({"max_completion_tokens": 4000} if is_reasoning(model) else {"temperature": 0, "max_tokens": 5})
        r = _post("https://api.openai.com/v1/chat/completions",
                  {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"}, body)
        return r["choices"][0]["message"]["content"] or ""
    if provider == "anthropic":
        r = _post("https://api.anthropic.com/v1/messages",
                  {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                   "content-type": "application/json"},
                  {"model": model, "max_tokens": 5, "temperature": 0, "messages": [{"role": "user", "content": prompt}]})
        return "".join(b.get("text", "") for b in r["content"])
    raise SystemExit(f"unknown provider {provider}")


def score(items, replies_fwd, replies_rev):
    ok_f = [parse(r) == it["answer"] for it, r in zip(items, replies_fwd)]
    ok_r = [parse(r) == it["answer"] for it, r in zip(items, replies_rev)]
    n = len(items)
    return dict(acc_forward=sum(ok_f) / n, acc_reversed=sum(ok_r) / n,
                flip=sum(a != b for a, b in zip(ok_f, ok_r)) / n,
                same_answer=sum(parse(a) == parse(b) for a, b in zip(replies_fwd, replies_rev)) / n,
                unparsed=sum(parse(r) is None for r in replies_fwd + replies_rev) / (2 * n), n=n)


def selftest():
    items = make_items(300, 4, 3)
    for it in items:                       # the answer really is what the session prints
        env = {}
        for line in (l for c in it["chains"] for l in c):
            lhs, rhs = line[4:].split(" = ")
            env[lhs] = rhs if rhs.isdigit() else env[rhs]
        assert env[it["query"]] == it["answer"]
        assert render(it, False) != render(it, True)
    perfect = [it["answer"] for it in items]
    s = score(items, perfect, perfect)
    assert s["acc_forward"] == 1 and s["flip"] == 0 and s["same_answer"] == 1
    last = lambda it, rev: (it["chains"][::-1] if rev else it["chains"])[-1][0].split(" = ")[1]
    s = score(items, [last(it, False) for it in items], [last(it, True) for it in items])
    assert s["flip"] > 0.2 and s["same_answer"] < 0.5    # a "take the last chain" model flips a lot
    assert parse("The answer is 7.") == "7" and parse("") is None
    from code_task import run
    assert is_reasoning("o3-mini") and not is_reasoning("gpt-4o-mini")
    for setting in ("2x3", "4x3", "6x3"):
        c, d = (int(x) for x in setting.split("x")); make_items(5, c, d)
    for it in code_prompts(50, 3):
        assert "Output" not in it["fwd"] and "```python" in it["fwd"]
        prog = lambda p: p.split("\n\n", 1)[1].replace("```python\n", "").replace("\n```", "")
        assert run(prog(it["fwd"])) == run(prog(it["rev"])) == it["answer"]
    print("frontier selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=("openai", "anthropic"))
    ap.add_argument("--model")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--settings", default="2x3,4x3,6x3,code2,code3",
                    help="chains x depth (e.g. 4x3) or codeL (program level L), comma-separated")
    ap.add_argument("--out", default="frontier_results")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not (args.provider and args.model):
        raise SystemExit("give --provider and --model")
    out = dict(provider=args.provider, model=args.model, settings={})
    for setting in args.settings.split(","):
        if setting.startswith("code"):
            items = code_prompts(args.n, int(setting[4:]))
            prompts = [(it["fwd"], it["rev"]) for it in items]
        else:
            chains, depth = (int(x) for x in setting.split("x"))
            items = make_items(args.n, chains, depth)
            prompts = [(render(it, False), render(it, True)) for it in items]
        fwd, rev = [], []
        try:
            for i, (pf, pr) in enumerate(prompts):
                fwd.append(ask(args.provider, args.model, pf))
                rev.append(ask(args.provider, args.model, pr))
                if i % 50 == 49:
                    print(f"  {setting}: {i + 1}/{len(items)}", flush=True)
        except RuntimeError as e:          # keep everything collected so far, then stop
            items, fwd, rev = items[:len(rev)], fwd[:len(rev)], rev
            print(f"STOPPED in {setting} after {len(rev)} questions: {e}")
            out["settings"][setting] = dict(summary=score(items, fwd, rev) if rev else None, replies_forward=fwd,
                                            replies_reversed=rev, stopped=str(e))
            save(out, args)
            raise SystemExit(1)
        s = score(items, fwd, rev)
        out["settings"][setting] = dict(summary=s, replies_forward=fwd, replies_reversed=rev)
        print(setting, {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()}, flush=True)
        save(out, args)                    # after every setting, so nothing paid for is lost


def save(out, args):
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.provider}__{args.model.replace('/', '_')}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
