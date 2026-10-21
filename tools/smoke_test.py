"""End-to-end check of the whole pipeline on two tiny random models (no downloads).
The numbers are meaningless; this only proves every script runs and writes its outputs.

  source env.sh && python tools/smoke_test.py
"""
import os
import subprocess
import sys
import tempfile

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, PreTrainedTokenizerFast

from tasks import build_all

HERE = os.path.dirname(os.path.abspath(__file__))
N = 150


def build_tiny(path, hidden):
    """A random GPT-NeoX with a word-level tokenizer covering every task word."""
    words = set()
    for items in build_all(N).values():
        for it in items:
            words.update(it["prompt"].split())
            words.update(c.strip() for c in it["candidates"])
    vocab = {w: i for i, w in enumerate(["<unk>", "<eos>"] + sorted(words))}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<eos>", unk_token="<unk>",
                            pad_token="<eos>").save_pretrained(path)
    cfg = GPTNeoXConfig(vocab_size=len(vocab), hidden_size=hidden, num_hidden_layers=4,
                        num_attention_heads=4, intermediate_size=2 * hidden,
                        max_position_embeddings=512, eos_token_id=1, pad_token_id=1)
    GPTNeoXForCausalLM(cfg).save_pretrained(path)


ROOT = os.path.dirname(HERE)


def sh(script, *args):
    path = next(os.path.join(ROOT, d, script) for d in ("core", "experiments", "analysis", "claims", "tools")
                if os.path.exists(os.path.join(ROOT, d, script)))
    print("$", script, *args, flush=True)
    subprocess.run([sys.executable, path, *args], check=True, cwd=ROOT)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        runs, figs, steer = (os.path.join(tmp, d) for d in ("runs", "figs", "steer"))
        tiny = [os.path.join(tmp, "tiny-10m"), os.path.join(tmp, "tiny-20m")]  # sizes in the name
        for path, hidden in zip(tiny, (32, 64)):
            build_tiny(path, hidden)
            # a local folder ignores 'revision', so both "checkpoints" have the same weights
            sh("sweep.py", "--model", path, "--steps", "0", "1000", "--n", str(N), "--out", runs,
               "--device", "cpu")

        sh("probes.py", "--runs", runs, "--out", os.path.join(tmp, "results.csv"),
           "--all-layers", "--jobs", "2")
        sh("analyze.py", "--results", os.path.join(tmp, "results.csv"), "--figdir", figs)
        sh("errors.py", "--runs", runs, "--out", os.path.join(figs, "error_types.csv"))
        for task in ("vartrack_d1", "vartrack_d2"):
            sh("steer.py", "--model", tiny[0], "--step", "0", "--task", task, "--n", str(N),
               "--out", steer, "--device", "cpu")
        sh("steer_summary.py", "--dir", steer, "--out", os.path.join(figs, "steer_summary.csv"))

        sys.path.insert(0, HERE)  # demo engine (the web UI itself needs gradio)
        from demo import Engine
        engine = Engine(runs, tiny[0], "vartrack_d1")
        prompt, answer = engine.example(seed=1)
        said, inside = engine.predict(engine.steps[-1], prompt)
        assert len(said) == len(inside) == 10 and abs(sum(said.values()) - 1) < 1e-4, "demo output"

        expected = {"depth_leads.png", "layer_depth.png", "scale_onsets.png", "onsets_and_leads.csv",
                    "seed_spread.csv", "h1_sign_test.csv", "error_types.csv", "steer_summary.csv"}
        missing = expected - set(os.listdir(figs))
        assert not missing, f"missing outputs: {missing}"
        print("outputs:", sorted(os.listdir(figs)))
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
