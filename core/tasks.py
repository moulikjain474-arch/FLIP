"""EarlyMind task generators.

Every task is a K-way choice at the NEXT token, so behavior, probes and the
model's own-logit decoder are all scored on the same K candidates.

Item fields:
  task        e.g. "copy", "vartrack_d2", "agreement"
  prompt      text fed to the model (answer is the next token)
  candidates  list of K candidate strings (each must be ONE token)
  label       index of the correct candidate
  meta        dict with generation details (for the data book / debugging)
"""
import random

LETTERS = list("ABCDEFGHJK")          # 10 classes for the copy task
DIGITS = [str(d) for d in range(10)]  # 10 classes for variable tracking
VAR_NAMES = list("abcdefghmnpqrstuvwxyz")

SG_PL = [
    ("key", "keys"), ("dog", "dogs"), ("teacher", "teachers"), ("car", "cars"),
    ("book", "books"), ("girl", "girls"), ("boy", "boys"), ("window", "windows"),
    ("picture", "pictures"), ("student", "students"), ("doctor", "doctors"),
    ("table", "tables"), ("painting", "paintings"), ("farmer", "farmers"),
    ("letter", "letters"), ("song", "songs"), ("friend", "friends"),
    ("house", "houses"), ("bird", "birds"), ("tree", "trees"),
    ("computer", "computers"), ("player", "players"), ("river", "rivers"),
    ("flower", "flowers"), ("lamp", "lamps"), ("chair", "chairs"),
]
PREPS = ["near", "behind", "next to", "beside", "above", "under"]


def make_copy(n, rng, seq_len=12):
    """In-context copying ('learning to learn'): a random letter sequence is
    shown, then repeated; the model must continue the repeat."""
    items = []
    for _ in range(n):
        seq = [rng.choice(LETTERS) for _ in range(seq_len)]
        k = rng.randint(3, seq_len - 1)          # how much of the repeat is shown
        prompt = " ".join(seq) + " |" + "".join(" " + c for c in seq[:k])
        items.append(dict(task="copy", prompt=prompt,
                          candidates=[" " + c for c in LETTERS],
                          label=LETTERS.index(seq[k]),
                          meta=dict(seq="".join(seq), k=k)))
    return items


def _vartrack_example(depth, rng, layout="interleaved"):
    """Two assignment chains (target + distractor) of equal depth with different
    start values. layout="interleaved": the chains' lines are shuffled together
    (order kept within each chain); "blocked": each chain's lines stay together,
    chains in random order. Returns (lines, query_var, answer, distractor)."""
    names = rng.sample(VAR_NAMES, 2 * (depth + 1))
    tgt, dis = names[: depth + 1], names[depth + 1:]
    v_t, v_d = rng.sample(DIGITS, 2)
    chains = []
    for chain, v in ((tgt, v_t), (dis, v_d)):
        stmts = [f">>> {chain[0]} = {v}"]
        stmts += [f">>> {chain[i]} = {chain[i - 1]}" for i in range(1, depth + 1)]
        chains.append(stmts)
    if layout == "blocked":
        rng.shuffle(chains)
        return chains[0] + chains[1], tgt[-1], v_t, v_d
    if layout != "interleaved":
        raise ValueError(f"unknown layout {layout!r}")
    order = [0] * len(chains[0]) + [1] * len(chains[1])
    rng.shuffle(order)
    idx, lines = [0, 0], []
    for c in order:
        lines.append(chains[c][idx[c]])
        idx[c] += 1
    return lines, tgt[-1], v_t, v_d


def make_vartrack(n, rng, depth, n_shots=2, layout="interleaved"):
    """Multi-step reasoning with a built-in difficulty knob: to answer, the
    model must follow `depth` assignment hops, ignoring a distractor chain whose
    value also appears in the prompt (so 'pick a digit you saw' gets ~50%)."""
    items = []
    for _ in range(n):
        blocks = []
        for _ in range(n_shots):
            lines, q, ans, _ = _vartrack_example(depth, rng, layout)
            blocks.append("\n".join(lines + [f">>> {q}", ans]))
        lines, q, ans, dis = _vartrack_example(depth, rng, layout)
        blocks.append("\n".join(lines + [f">>> {q}"]) + "\n")
        items.append(dict(task=f"vartrack_d{depth}", prompt="\n\n".join(blocks),
                          candidates=list(DIGITS), label=DIGITS.index(ans),
                          meta=dict(depth=depth, answer=ans, distractor=dis,
                                    shots=n_shots, layout=layout)))
    return items


def make_agreement(n, rng):
    """Subject-verb agreement with an attractor noun of the opposite number.
    Nouns appear as subject and as attractor equally often, so a bag-of-words
    model cannot tell which noun is the subject."""
    items, seen = [], set()
    while len(items) < n:  # no duplicate prompts (they would leak across CV folds)
        subj_pair, attr_pair = rng.sample(SG_PL, 2)
        plural = rng.random() < 0.5
        subj = subj_pair[1] if plural else subj_pair[0]
        attr = attr_pair[0] if plural else attr_pair[1]
        prompt = f"The {subj} {rng.choice(PREPS)} the {attr}"
        if prompt in seen:
            continue
        seen.add(prompt)
        items.append(dict(task="agreement", prompt=prompt,
                          candidates=[" is", " are"], label=int(plural),
                          meta=dict(subj=subj, attr=attr)))
    return items


DEFAULT_SHOTS, DEFAULT_LAYOUT = 2, "interleaved"


def build_all(n_per_task=600, seed=0, depths=(1, 2, 3), shots=DEFAULT_SHOTS, layout=DEFAULT_LAYOUT):
    """Fixed item set used for EVERY checkpoint (same arguments -> same items)."""
    rng = random.Random(seed)
    tasks = {"copy": make_copy(n_per_task, rng),
             "agreement": make_agreement(n_per_task, rng)}
    for d in depths:
        tasks[f"vartrack_d{d}"] = make_vartrack(n_per_task, rng, d, shots, layout)
    return tasks


if __name__ == "__main__":
    t = build_all(n_per_task=2)
    for name, items in t.items():
        print("=" * 20, name)
        print(items[0]["prompt"], "->", repr(items[0]["candidates"][items[0]["label"]]))
