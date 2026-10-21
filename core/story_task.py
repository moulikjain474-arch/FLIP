"""Second binding task in plain English (pure Python): the same two-chain structure as
vartrack, told as a short story, with BOTH chain orders built for every question.

  Ravi has 4 coins. Mia has as many coins as Ravi.      <- one chain (depth 1)
  Ana has 7 coins. Tom has as many coins as Ana.        <- the other chain
  Q: How many coins does Mia have?
  A:                                                    -> "4" on the next line

Items are generated with their own random seed, so the vartrack items in tasks.py are
untouched. The answer follows "A:" and a newline with bare-digit candidates "0".."9", as in
tasks.py, so it is one token for every tokenizer (Pythia, OLMo-2, Qwen).

theme="med" (claim C26) tells the same structure as a medication record; only the wording changes:
  Patient Ravi takes 4 mg of warfarin. Patient Mia takes the same warfarin dose as patient Ravi.
  Q: How many mg of warfarin does patient Mia take?
Both chains in one question use the same drug, so only the binding tells them apart. The med items use
their own seed offset; theme="coins" (the default) produces exactly the same items as before.
"""
import random

NAMES = ["Ravi", "Mia", "Ana", "Tom", "Lea", "Sam", "Ivy", "Ben", "Zoe", "Max", "Eva", "Leo",
         "Ada", "Kai", "Nia", "Raj", "Uma", "Dev", "Ali", "Joy", "Ken", "Liz", "Ned", "Ola"]
DIGITS = [str(d) for d in range(10)]
CANDIDATES = list(DIGITS)
DRUGS = ["warfarin", "lorazepam", "morphine", "haloperidol", "diazepam", "methadone", "insulin glargine", "prednisone"]


def _chains(depth, rng, drug=None):
    names = rng.sample(NAMES, 2 * (depth + 1))
    tgt, dis = names[: depth + 1], names[depth + 1:]
    v_t, v_d = rng.sample(DIGITS if drug is None else DIGITS[1:], 2)     # no "0 mg" doses in the med wording
    out = []
    for chain, v in ((tgt, v_t), (dis, v_d)):
        if drug is None:
            s = [f"{chain[0]} has {v} {'coin' if v == '1' else 'coins'}."] + [f"{chain[i]} has as many coins as {chain[i - 1]}." for i in range(1, depth + 1)]
        else:
            s = [f"Patient {chain[0]} takes {v} mg of {drug}."] + [f"Patient {chain[i]} takes the same {drug} dose as patient {chain[i - 1]}." for i in range(1, depth + 1)]
        out.append(" ".join(s))
    return out[0], out[1], tgt[-1], v_t, v_d


def _question(q, drug):
    return f"Q: How many coins does {q} have?" if drug is None else f"Q: How many mg of {drug} does patient {q} take?"


def make_story(n, depth, seed=0, shots=2, theme="coins"):
    """Items with prompt (target chain first or second at random) and prompt_swapped."""
    assert theme in ("coins", "med")
    med = theme == "med"
    rng = random.Random(1000 + 17 * depth + seed + (5000 if med else 0))
    items = []
    for _ in range(n):
        blocks = []
        for _ in range(shots):
            drug = rng.choice(DRUGS) if med else None
            a, b, q, ans, _ = _chains(depth, rng, drug)
            first, second = (a, b) if rng.random() < 0.5 else (b, a)
            blocks.append(f"{first}\n{second}\n{_question(q, drug)}\nA:\n{ans}")
        drug = rng.choice(DRUGS) if med else None
        a, b, q, ans, dis = _chains(depth, rng, drug)
        target_first = rng.random() < 0.5
        orders = [(a, b), (b, a)] if target_first else [(b, a), (a, b)]
        head = "\n\n".join(blocks) + "\n\n"
        prompts = [head + f"{x}\n{y}\n{_question(q, drug)}\nA:\n" for x, y in orders]
        items.append(dict(task=f"{'med' if med else 'story'}_d{depth}", prompt=prompts[0], prompt_swapped=prompts[1],
                          candidates=list(CANDIDATES), label=DIGITS.index(ans),
                          meta=dict(depth=depth, answer=ans, distractor=dis, target_first=target_first, theme=theme)))
    return items
