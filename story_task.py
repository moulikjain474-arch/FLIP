"""Second binding task in plain English (pure Python): the same two-chain structure as
vartrack, told as a short story, with BOTH chain orders built for every question.

  Ravi has 4 coins. Mia has as many coins as Ravi.      <- one chain (depth 1)
  Ana has 7 coins. Tom has as many coins as Ana.        <- the other chain
  Q: How many coins does Mia have?
  A:                                                    -> "4" on the next line

Items are generated with their own random seed, so the vartrack items in tasks.py are
untouched. The answer follows "A:" and a newline with bare-digit candidates "0".."9", as in
tasks.py, so it is one token for every tokenizer (Pythia, OLMo-2, Qwen).
"""
import random

NAMES = ["Ravi", "Mia", "Ana", "Tom", "Lea", "Sam", "Ivy", "Ben", "Zoe", "Max", "Eva", "Leo",
         "Ada", "Kai", "Nia", "Raj", "Uma", "Dev", "Ali", "Joy", "Ken", "Liz", "Ned", "Ola"]
DIGITS = [str(d) for d in range(10)]
CANDIDATES = list(DIGITS)


def _chains(depth, rng):
    names = rng.sample(NAMES, 2 * (depth + 1))
    tgt, dis = names[: depth + 1], names[depth + 1:]
    v_t, v_d = rng.sample(DIGITS, 2)
    out = []
    for chain, v in ((tgt, v_t), (dis, v_d)):
        s = [f"{chain[0]} has {v} {'coin' if v == '1' else 'coins'}."] + [f"{chain[i]} has as many coins as {chain[i - 1]}." for i in range(1, depth + 1)]
        out.append(" ".join(s))
    return out[0], out[1], tgt[-1], v_t, v_d


def make_story(n, depth, seed=0, shots=2):
    """Items with prompt (target chain first or second at random) and prompt_swapped."""
    rng = random.Random(1000 + 17 * depth + seed)
    items = []
    for _ in range(n):
        blocks = []
        for _ in range(shots):
            a, b, q, ans, _ = _chains(depth, rng)
            first, second = (a, b) if rng.random() < 0.5 else (b, a)
            blocks.append(f"{first}\n{second}\nQ: How many coins does {q} have?\nA:\n{ans}")
        a, b, q, ans, dis = _chains(depth, rng)
        target_first = rng.random() < 0.5
        orders = [(a, b), (b, a)] if target_first else [(b, a), (a, b)]
        head = "\n\n".join(blocks) + "\n\n"
        prompts = [head + f"{x}\n{y}\nQ: How many coins does {q} have?\nA:\n" for x, y in orders]
        items.append(dict(task=f"story_d{depth}", prompt=prompts[0], prompt_swapped=prompts[1],
                          candidates=list(CANDIDATES), label=DIGITS.index(ans),
                          meta=dict(depth=depth, answer=ans, distractor=dis, target_first=target_first)))
    return items
