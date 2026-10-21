"""Meaning-preserving edits of a vartrack prompt (pure Python, no torch).

The final problem block (after the last blank line) is two self-contained chains of
assignments followed by a query line. Two edits leave the correct answer unchanged:

  swap_chains    write the second chain first        (changes the ORDER of the chains)
  rename_chains  give every variable a new name      (changes only the NAMES)

Comparing how often a model changes its answer under each edit separates "order matters"
from "any small edit makes the model change its mind".
"""
import keyword
import random
import re
import string

NAME_POOL = "abcdefghmnpqrstuvwxyz"   # the letters tasks.py draws variable names from
ASSIGN = re.compile(r"^>>> (\w+) = (\w+)\s*$")
VARIANTS = ("orig", "swap", "rename", "swap_rename")
INSTRUCT_VARIANTS = ("instruct", "instruct_swap")
# A plain prompt-based mitigation, as tried in earlier work on serial-position effects
# (Guo & Vosoughi 2025): tell the model what to attend to, then ask in both orders as usual.
INSTRUCTION = ("# Careful: answer with the value of the variable that is asked about on the last line,\n"
               "# not simply the first or the last value written.\n\n")


def split_problem(prompt):
    """(head, chain_a, chain_b, tail) of the final problem block; asserts the expected format."""
    cut = prompt.rstrip("\n").rfind("\n\n") + 2
    head, block = prompt[:cut], prompt[cut:]
    lines = block.split("\n")
    queries = [i for i, l in enumerate(lines) if l.startswith(">>> ") and not ASSIGN.match(l)]
    assert queries, "no query line in the final problem block"
    q = queries[-1]
    assigns, tail = lines[:q], lines[q:]
    assert all(ASSIGN.match(l) for l in assigns), f"unexpected line in problem block: {assigns}"
    assert len(assigns) % 2 == 0, f"odd number of assignment lines: {assigns}"
    h = len(assigns) // 2
    assert h > 0, "no assignment lines in the final problem block"
    a, b = assigns[:h], assigns[h:]
    for chain in (a, b):  # each chain starts from a constant and then refers only to itself
        assert ASSIGN.match(chain[0]).group(2).isdigit(), f"chain does not start with a number: {chain}"
        names = {ASSIGN.match(chain[0]).group(1)}
        for line in chain[1:]:
            lhs, rhs = ASSIGN.match(line).groups()
            assert rhs in names, f"chain is not self-contained: {chain}"
            names.add(lhs)
    return head, a, b, tail


def swap_chains(prompt):
    """The prompt with the two chains of the final problem swapped (answer unchanged)."""
    head, a, b, tail = split_problem(prompt)
    return head + "\n".join(b + a + tail)


def rename_chains(prompt, seed):
    """The prompt with every variable of the final problem renamed (answer unchanged).

    New names are random, have the same length as the old ones, come from the same letters
    tasks.py uses, and never reuse a name that appears in the problem, so the edit changes the
    names and nothing else.
    """
    head, a, b, tail = split_problem(prompt)
    old = [ASSIGN.match(line).group(1) for line in a + b]
    rng = random.Random(seed)
    taken, mapping = set(old), {}
    for name in old:
        while True:
            pool = NAME_POOL if len(name) == 1 else string.ascii_lowercase
            new = "".join(rng.choice(pool) for _ in name)
            if new not in taken and not keyword.iskeyword(new):
                break
        mapping[name] = new
        taken.add(new)

    def sub(text):
        return re.sub(r"\b\w+\b", lambda m: mapping.get(m.group(0), m.group(0)), text)

    def line(l):
        return ">>> " + sub(l[4:]) if l.startswith(">>> ") else l

    return head + "\n".join([line(l) for l in a + b] + [line(l) for l in tail])


def variants(prompt, seed, instruct=False):
    """All versions of a prompt, keyed by VARIANTS (plus INSTRUCT_VARIANTS if instruct)."""
    v = dict(orig=prompt, swap=swap_chains(prompt), rename=rename_chains(prompt, seed),
             swap_rename=rename_chains(swap_chains(prompt), seed))
    if instruct:
        v.update(instruct=INSTRUCTION + v["orig"], instruct_swap=INSTRUCTION + v["swap"])
    return v


def chain_char_spans(prompt):
    """[(start, end), (start, end)]: character spans of the first and second chain of the final block."""
    head, a, b, _ = split_problem(prompt)
    ta, tb = "\n".join(a), "\n".join(b)
    s1 = len(head)
    s2 = s1 + len(ta) + 1
    assert prompt[s1:s1 + len(ta)] == ta and prompt[s2:s2 + len(tb)] == tb
    return [(s1, s1 + len(ta)), (s2, s2 + len(tb))]


def chain_values(prompt):
    """Start value of the first and second chain, and which chain (0/1) the query is about."""
    _, a, b, tail = split_problem(prompt)
    q = re.match(r"^>>> (\w+)", tail[0]).group(1)
    vals = [ASSIGN.match(c[0]).group(2) for c in (a, b)]
    names = [{ASSIGN.match(l).group(1) for l in c} for c in (a, b)]
    return vals, (0 if q in names[0] else 1)


FILLER = ">>> pass"


def add_filler(prompt, where, k):
    """Insert k no-op lines ('>>> pass') into the final block (answer unchanged).

    where="between":      after the first chain, so the first chain moves AWAY from the question.
    where="before_query": after the second chain, so BOTH chains move away from the question.
    """
    head, a, b, tail = split_problem(prompt)
    fill = [FILLER] * k
    body = a + fill + b if where == "between" else a + b + fill if where == "before_query" else None
    assert body is not None, f"unknown filler position {where!r}"
    return head + "\n".join(body + tail)


def strip_filler(prompt):
    return prompt.replace("\n" + FILLER, "")


def final_answer(prompt):
    """Value of the queried variable, computed from the final block; None if the format differs."""
    try:
        _, a, b, tail = split_problem(prompt)
        env = {}
        for l in a + b:
            lhs, rhs = ASSIGN.match(l).groups()
            env[lhs] = rhs if rhs.isdigit() else env[rhs]
    except (AssertionError, KeyError):
        return None
    q = re.match(r"^>>> (\w+)\s*$", tail[0])
    return env.get(q.group(1)) if q else None


def single_chain_control(prompt, seed):
    """Control pair for the order-patching test: the final problem with ONLY the queried chain.

    Returns (src_prompt, tgt_prompt, v_src, v_tgt). The two prompts are identical except for the
    chain's start digit (v_src vs a different v_tgt), so there is no second chain to choose
    between: patching the source's last-token state into the target at a layer shows where a VALUE
    is read out. Compare with the two-chain order patch, which shows where the CHOICE of chain is made.
    """
    head, a, b, tail = split_problem(prompt)
    _, which = chain_values(prompt)
    chain = [a, b][which]
    v_src = ASSIGN.match(chain[0]).group(2)
    rng = random.Random(seed)
    v_tgt = rng.choice([d for d in "0123456789" if d != v_src])
    mk = lambda v: head + "\n".join([chain[0].rsplit(" = ", 1)[0] + " = " + v] + chain[1:] + tail)
    return mk(v_src), mk(v_tgt), v_src, v_tgt
