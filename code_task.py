"""A harder, more realistic binding test: short Python programs (pure Python).

Each program defines several small functions and values, then prints one of them. The
definitions do not depend on each other's ORDER (every function is defined before anything
calls it, and every value line only uses functions and earlier-defined values), so two
orderings of the same program print the same thing. Every value prints a different digit,
so the other values act as distractors, as in the variable-chain task. Every answer is checked by running
the program. Answers are single digits (no negative numbers occur), and the answer follows a newline
with bare-digit candidates "0".."9", as in tasks.py, so it is one token for every tokenizer.

  def scale(x):
      return x * 3
  def shift(x):
      return x + 4
  p = scale(2)
  q = shift(1)
  print(q % 10)
  Output:
  5
"""
import contextlib
import io
import random

FUNCS = [("scale", "x * {k}"), ("shift", "x + {k}"), ("square", "x * x + {k}"), ("halve", "x // 2 + {k}"),
         ("twice", "2 * x + {k}"), ("bump", "(x + {k}) * 2")]
VARS = list("pqrstuvwmn")


def _program(rng, n_funcs, n_vals):
    """Functions, value lines and the queried variable. Every value line prints a DIFFERENT
    digit, so answering needs the right variable (the others act as distractors)."""
    while True:
        funcs = rng.sample(FUNCS, n_funcs)
        fdefs = [f"def {name}(x):\n    return {body.format(k=rng.randint(1, 5))}" for name, body in funcs]
        names = rng.sample(VARS, n_vals)
        vals = [f"{v} = {rng.choice(funcs)[0]}({rng.randint(0, 9)})" for v in names]
        digits = [run("\n".join(fdefs + vals + [f"print({v} % 10)"])) for v in names]
        if len(set(digits)) == n_vals:
            return fdefs, vals, rng.choice(names)


def run(code):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, {})
    return buf.getvalue().strip()


def render(fdefs, vals, query):
    return "\n".join(fdefs + vals + [f"print({query} % 10)"])


def make_code(n, level, seed=0, shots=2):
    """level 1: 2 functions, 2 values; level 2: 3 and 3; level 3: 4 and 4."""
    n_funcs, n_vals = {1: (2, 2), 2: (3, 3), 3: (4, 4)}[level]
    rng = random.Random(5000 + 101 * level + seed)
    items = []
    for _ in range(n):
        shots_txt = []
        for _ in range(shots):
            f, v, q = _program(rng, n_funcs, n_vals)
            code = render(f, v, q)
            shots_txt.append(f"```python\n{code}\n```\nOutput:\n{run(code)}")
        f, v, q = _program(rng, n_funcs, n_vals)
        code_a = render(f, v, q)
        code_b = render(f[::-1], v[::-1], q)          # same program, definitions in reverse order
        ans = run(code_a)
        assert ans == run(code_b) and len(ans) == 1, (code_a, code_b)
        head = "\n\n".join(shots_txt) + "\n\n"
        items.append(dict(task=f"code_l{level}",
                          prompt=head + f"```python\n{code_a}\n```\nOutput:\n",
                          prompt_swapped=head + f"```python\n{code_b}\n```\nOutput:\n",
                          candidates=[str(d) for d in range(10)], label=int(ans),
                          meta=dict(level=level, answer=ans, query=q)))
    return items
