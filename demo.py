"""Booth demo: what the model SAYS vs what is READABLE INSIDE it, at any checkpoint.

Pick a training checkpoint and a variable-tracking prompt. The left panel shows
the model's own answer; the right panel shows the answer read from its middle
layer by the probe. At an early checkpoint the model often answers wrong while
its insides already point to the right answer; that is the project in one view.

Probes are fit on the activations sweep.py already saved (no extra model runs),
with exactly the settings used in probes.py. Runs on CPU for 160M/410M models.

  pip install gradio
  python demo.py --runs runs/ --model EleutherAI/pythia-410m --task vartrack_d1
"""
import argparse
import glob
import os
import random
from collections import OrderedDict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import io_utils as iou
import model_utils as mu
from probes import C
from tasks import make_vartrack


class Engine:
    """Fits one probe per checkpoint from saved activations; answers prompts live."""

    def __init__(self, runs, model, task, device="cpu", max_loaded=2):
        self.model_name, self.task, self.device = model, task, device
        self.max_loaded = max_loaded
        self.files = {}
        for path in filter(iou.is_result, glob.glob(os.path.join(runs, "*.npz"))):
            m, rev, item_tag, t = iou.parse_result_name(path)
            if m == model and t == task:
                self.files[int(np.load(path)["step"])] = (path, rev, item_tag)
        if not self.files:
            raise SystemExit(f"no saved results for {model} / {task} in {runs!r}; run sweep.py first")
        self.steps = sorted(self.files)
        _, _, item_tag = self.files[self.steps[0]]
        self.candidates = iou.load_items(runs, item_tag)[task][0]["candidates"]
        self.depth = int(task.rsplit("_d", 1)[1]) if "_d" in task else 1
        self._probes, self._loaded = {}, OrderedDict()

    def probe(self, step):
        """Probe for one checkpoint (fit once from saved activations, then cached)."""
        if step not in self._probes:
            d = np.load(self.files[step][0])
            layer = int(max(d["layers"])) // 2
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
            clf.fit(d[f"acts_L{layer}"].astype(np.float32), d["labels"])
            self._probes[step] = (layer, clf)
        return self._probes[step]

    def model(self, step):
        """Model at one checkpoint (least-recently-used cache, to bound memory)."""
        if step in self._loaded:
            self._loaded.move_to_end(step)
        else:
            if len(self._loaded) >= self.max_loaded:
                self._loaded.popitem(last=False)
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._loaded[step] = mu.load(self.model_name, self.files[step][1], self.device, dtype)
        return self._loaded[step]

    def predict(self, step, prompt):
        """-> ({answer: prob} from the model's output, {answer: prob} from the probe)."""
        step = int(step)
        layer, clf = self.probe(step)
        tok, model = self.model(step)
        cids = mu.candidate_ids(tok, self.candidates)
        logits, acts = mu.run(tok, model, [prompt], cids, read_layers=[layer],
                              batch_size=1, device=self.device)
        said = torch.softmax(logits[0], -1).numpy()
        inside = np.zeros(len(self.candidates))
        inside[clf.classes_] = clf.predict_proba(acts[layer].numpy())[0]
        as_dict = lambda p: {c.strip(): float(v) for c, v in zip(self.candidates, p)}
        return as_dict(said), as_dict(inside)

    def example(self, seed=None):
        """A fresh variable-tracking prompt (never seen by the probe) and its answer."""
        item = make_vartrack(1, random.Random(seed), self.depth)[0]
        return item["prompt"], item["meta"]["answer"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--task", default="vartrack_d1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--share", action="store_true", help="public link (useful on Colab)")
    args = ap.parse_args()
    try:
        import gradio as gr
    except ImportError:
        raise SystemExit("the demo needs gradio: pip install gradio")

    engine = Engine(args.runs, args.model, args.task, args.device)
    first_prompt, first_answer = engine.example(seed=0)

    with gr.Blocks(title="EarlyMind demo") as ui:
        gr.Markdown(f"## EarlyMind: what the model says vs what it already knows inside\n"
                    f"`{args.model}` · task `{args.task}`. Pick a checkpoint to move through training.")
        step = gr.Dropdown([str(s) for s in engine.steps], value=str(engine.steps[len(engine.steps) // 2]),
                           label="Training checkpoint (step)")
        prompt = gr.Textbox(first_prompt, lines=10, label="Prompt")
        answer = gr.Textbox(first_answer, label="Correct answer", interactive=False)
        with gr.Row():
            new = gr.Button("New example")
            run = gr.Button("Run", variant="primary")
        with gr.Row():
            said = gr.Label(num_top_classes=3, label="What the model SAYS")
            inside = gr.Label(num_top_classes=3, label="What is READABLE INSIDE (middle layer)")
        run.click(engine.predict, [step, prompt], [said, inside])
        step.change(engine.predict, [step, prompt], [said, inside])
        new.click(lambda: engine.example(), None, [prompt, answer])
    ui.launch(share=args.share)


if __name__ == "__main__":
    main()
