"""Order-Flip Test on closed frontier models through their APIs (exploratory, behavioural).

Runs on any laptop (no GPU). The model sees K chains of variable assignments, written in one
order and then in the reverse order, and must reply with the value of one variable. Harder
than the main task (more chains, deeper) so that strong models are not at 100%.

  export OPENAI_API_KEY=...      (or ANTHROPIC_API_KEY=...)
  python3 frontier_flip.py --provider openai --model <model-name> --n 200
  python3 frontier_flip.py --selftest
In hard mode the "reversed" fields hold the SHUFFLED version. Re-running the same model adds
to (never overwrites) its saved file. Reports accuracy in each order, flip rate (correct in exactly one order), and how often the
two orders give the same answer. Every reply is saved, so the scoring can be re-checked.
"""
import argparse
import http.client
import json
import os
import random
import re
import time
import urllib.error
import urllib.request

_L = "abcdefghijklmnopqrstuvwxyz"
NAMES = list(_L) + [a + b for a in _L for b in _L if a + b not in ("as", "if", "in", "is", "or")]   # 2-letter names for the big settings
INSTRUCTION = ("Below is a short Python session. Reply with only the single digit that the last "
               "line prints, nothing else.\n\n")


def make_items(n, chains, depth, seed=0):
    """Each item: list of chains (each a list of lines), the query variable, the answer."""
    if chains > 10:
        raise SystemExit(f"{chains} chains need {chains} different digits; at most 10 exist")
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


# ---- hard mode: facts that depend on each other, given in dependency order vs shuffled ------
# Built from published failure modes of frontier models: premise order (Chen et al., ICML 2024:
# GPT-4-turbo 80.5% -> 57.2% when dependent rules are shuffled), more hops and more distractor
# chains (same paper; RULER, COLM 2024), and long irrelevant filler text (RULER).
_SYL = ["ka", "ri", "mo", "lu", "sa", "ne", "ti", "po", "da", "vi", "ra", "no", "be", "zu", "fe", "ga"]
PEOPLE = [(a + b).capitalize() for a in _SYL for b in _SYL if a != b]          # 240 two-syllable names
FILLER_SENTENCES = ["The weather was mild that afternoon.", "Someone left a window open in the hall.",
                    "A bus went past the old library.", "The garden smelled of fresh rain.",
                    "Nobody had cleaned the kitchen yet.", "There was music coming from next door.",
                    "The cat slept on the warm stairs.", "A delivery van parked across the street."]
HARD_LEVELS = {1: (4, 4, 0), 2: (6, 6, 800), 3: (8, 8, 2500), 4: (10, 10, 6000)}   # chains, hops, filler words
HARD_INSTRUCTION = ("Read the facts below. Each person has some coins, either a stated number or the same "
                    "number as another person. Reply with only the number, nothing else.\n\n")


def make_hard(n, level, seed=0):
    """Items with the same facts in dependency order ('forward') and randomly shuffled ('shuffled')."""
    chains, hops, filler = HARD_LEVELS[level]
    rng = random.Random(7000 + 13 * level + seed)
    items = []
    for _ in range(n):
        names = rng.sample(PEOPLE, chains * (hops + 1))
        values = rng.sample([str(d) for d in range(10)], chains)
        facts = []                                   # per chain, in dependency order
        for c in range(chains):
            ns = names[c * (hops + 1):(c + 1) * (hops + 1)]
            ch = [f"{ns[0]} has {values[c]} coins."] + [f"{ns[i]} has as many coins as {ns[i - 1]}." for i in range(1, hops + 1)]
            facts.append(ch)
        target = rng.randrange(chains)
        query = names[target * (hops + 1) + hops]
        # forward: chains interleaved but each chain's facts keep their dependency order
        order = [c for c in range(chains) for _ in range(hops + 1)]
        rng.shuffle(order)
        pos, forward = [0] * chains, []
        for c in order:
            forward.append(facts[c][pos[c]]); pos[c] += 1
        shuffled = forward[:]
        while shuffled == forward:
            rng.shuffle(shuffled)
        fill = [rng.choice(FILLER_SENTENCES) for _ in range(filler // 7)]
        def with_filler(lines):
            out, slots = list(lines), sorted(rng.sample(range(len(lines) + len(fill)), len(fill))) if fill else []
            for s, sentence in zip(slots, fill):
                out.insert(s, sentence)
            return out
        fwd_lines, shf_lines = with_filler(forward), with_filler(shuffled)
        q = f"\n\nHow many coins does {query} have?"
        items.append(dict(answer=values[target], fwd=HARD_INSTRUCTION + " ".join(fwd_lines) + q,
                          rev=HARD_INSTRUCTION + " ".join(shf_lines) + q))
    return items


def resolve_story(text):
    """Answer computed from the facts (used to check every hard item)."""
    env, links = {}, {}
    for s in re.findall(r"([A-Z][a-z]+) has (?:as many coins as ([A-Z][a-z]+)|(\d) coins)\.", text):
        who, src, num = s
        (env.__setitem__(who, num) if num else links.__setitem__(who, src))
    q = re.search(r"How many coins does ([A-Z][a-z]+) have", text).group(1)
    while q not in env:
        q = links[q]
    return env[q]


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
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and k < tries - 1:
                time.sleep(2 ** k); continue
            raise RuntimeError(f"API error {e.code}: {e.read()[:300]!r}")
        except (urllib.error.URLError, http.client.HTTPException, OSError):   # incl. timeouts and dropped connections
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
    for level in HARD_LEVELS:
        hs = make_hard(30, level)
        assert hs == make_hard(30, level)
        for h in hs:
            assert resolve_story(h["fwd"]) == resolve_story(h["rev"]) == h["answer"] and h["fwd"] != h["rev"]
    for setting in ("2x3", "4x3", "6x3", "8x4", "10x5"):
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
    ap.add_argument("--settings", default="2x3,4x3,code2,code3,hard1,hard2,hard3,hard4",
                    help="chains x depth (e.g. 4x3), codeL (program level L) or hardL (hard mode level L)")
    ap.add_argument("--score", metavar="DIR", help="score C21 from saved results in DIR, then exit")
    ap.add_argument("--out", default="frontier_results")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.score:
        return score_c21(args.score)
    if not (args.provider and args.model):
        raise SystemExit("give --provider and --model")
    path = os.path.join(args.out, f"{args.provider}__{args.model.replace('/', '_')}.json")
    out = json.load(open(path)) if os.path.exists(path) else dict(provider=args.provider, model=args.model, settings={})
    for setting in args.settings.split(","):
        if setting.startswith("hard"):
            items = make_hard(args.n, int(setting[4:]))
            prompts = [(it["fwd"], it["rev"]) for it in items]
        elif setting.startswith("code"):
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
        except (RuntimeError, KeyError, IndexError, TypeError, KeyboardInterrupt) as e:   # keep what was collected, stop
            items, fwd, rev = items[:len(rev)], fwd[:len(rev)], rev
            print(f"STOPPED in {setting} after {len(rev)} questions: {e}")
            out["settings"][setting] = dict(summary=score(items, fwd, rev) if rev else None, replies_forward=fwd,
                                            replies_reversed=rev, answers=[it["answer"] for it in items], stopped=repr(e))
            save(out, args)
            raise SystemExit(1)
        s = score(items, fwd, rev)
        out["settings"][setting] = dict(summary=s, replies_forward=fwd, replies_reversed=rev,
                                        answers=[it["answer"] for it in items])
        print(setting, {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()}, flush=True)
        save(out, args)                    # after every setting, so nothing paid for is lost


def score_c21(folder):
    """C21: some frontier model, at some hard level, is >= 0.05 less accurate with shuffled facts,
    with a paired-bootstrap 95% interval of the difference above 0."""
    import glob
    import numpy as np
    import flipstats as fs
    hit = False
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    print("== C21: shuffled facts hurt frontier models (hard mode; >= 0.05 drop, 95% interval above 0)")
    for f in files:
        s = json.load(open(f))
        if "settings" not in s:
            continue
        for level in sorted(k for k in s["settings"] if k.startswith("hard")):
            r = s["settings"][level]
            if not r.get("replies_reversed"):
                continue
            n = len(r["replies_reversed"])
            ans = r.get("answers") or [it["answer"] for it in make_hard(n, int(level[4:]))]
            ans = ans[:n]
            cf = np.array([parse(x) == a for x, a in zip(r["replies_forward"][:n], ans)], float)
            cs = np.array([parse(x) == a for x, a in zip(r["replies_reversed"], ans)], float)
            d, lo, hi = fs.paired_bootstrap(cf, cs)
            ok = d >= 0.05 - 1e-9 and lo > 0
            hit |= ok
            unparsed = sum(parse(x) is None for x in r["replies_forward"][:n] + r["replies_reversed"]) / (2 * n)
            print(f"  {s['provider']} {s['model']} {level}: forward {cf.mean():.3f}  shuffled {cs.mean():.3f}  "
                  f"drop {d:+.3f} [{lo:+.3f},{hi:+.3f}]  n={n}  unparsed {unparsed:.2f}  {'yes' if ok else 'no'}")
    print("  C21:", "HIT" if hit else ("MISS" if files else "NOT RUN"))


def save(out, args):
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.provider}__{args.model.replace('/', '_')}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
