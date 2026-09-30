"""Loading checkpoints, running prompts, reading and editing activations.

Convention: "layer l" (1..n_layers) is the residual stream AFTER transformer
block l, read at the LAST prompt token. Reading and steering use the same hook
point, so probes and interventions refer to exactly the same vector.
"""
import contextlib

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

TOKENS_PER_STEP = 2_097_152  # Pythia: 1024 sequences x 2048 tokens per optimizer step


def load(model_name, revision=None, device="cuda", dtype=torch.float16):
    """Tokenizer (left-padded) + model at one training checkpoint (revision)."""
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    tok.padding_side = "left"            # the last prompt token is always at position -1
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # transformers >= 4.56 calls this argument `dtype`; older versions `torch_dtype`
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dtype_kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, **{dtype_kw: dtype})
    model.to(device).eval()
    return tok, model


def get_layers(model):
    """The list of transformer blocks (GPT-NeoX / Pythia, Llama-style incl. OLMo, GPT-2)."""
    for path in ("gpt_neox.layers", "model.layers", "transformer.h"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise ValueError("unknown architecture: cannot find transformer blocks")


def n_layers(model):
    return len(get_layers(model))


def default_layers(model):
    """Layers recorded by sweep.py: quarter, middle (the pre-registered main layer), 3/4, last."""
    n = n_layers(model)
    return sorted({max(1, n // 4), n // 2, (3 * n) // 4, n})


def candidate_ids(tok, candidates):
    """Token id of each answer option; every option must be exactly one token."""
    ids = []
    for c in candidates:
        t = tok(c, add_special_tokens=False)["input_ids"]
        if len(t) != 1:
            raise ValueError(f"candidate {c!r} is {len(t)} tokens {t}; must be exactly 1")
        ids.append(t[0])
    return ids


def encode(tok, prompts, device):
    """Left-padded batch plus rotary-safe position ids (padding does not shift positions)."""
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    last = enc["input_ids"][:, -1]
    if tok.eos_token_id is not None and (last == tok.eos_token_id).any():
        raise ValueError("a prompt ends in EOS: the last-token convention would read the wrong token")
    pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    return enc, pos


def last_logits(model, enc, pos):
    """Next-token logits at the last position only (avoids a B x T x vocab tensor)."""
    kw = dict(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], position_ids=pos)
    try:
        out = model(**kw, logits_to_keep=1)
    except TypeError:  # older transformers without logits_to_keep
        out = model(**kw)
    return out.logits[:, -1, :]


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _replace(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


@contextlib.contextmanager
def hooks(model, read_layers=(), edit=None):
    """Record and/or edit the last-token residual stream.

    read_layers: layer numbers whose last-token vector is recorded (after any edit).
    edit:        optional (layers, fn). fn(h[B, d] float32, layer) -> new h, applied at
                 the last token after every block in `layers` (an int or a collection).
    """
    blocks = get_layers(model)
    edit_layers = set()
    if edit is not None:
        edit_layers = {edit[0]} if isinstance(edit[0], int) else set(edit[0])
    store, handles = {}, []

    def make(l):
        def hook(_module, _inputs, out):
            h = _hidden(out)
            if l in edit_layers:
                h = h.clone()
                h[:, -1, :] = edit[1](h[:, -1, :].float(), l).to(h.dtype)
                out = _replace(out, h)
            if l in read_layers:
                store[l] = h[:, -1, :].float().cpu()
            return out
        return hook

    for l in sorted(set(read_layers) | edit_layers):
        handles.append(blocks[l - 1].register_forward_hook(make(l)))
    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def run(tok, model, prompts, cand_ids, read_layers=(), edit=None, batch_size=32, device="cuda"):
    """Returns (candidate logits [N, K] float32, {layer: last-token activations [N, d]})."""
    all_logits, acts = [], {l: [] for l in read_layers}
    cand = torch.tensor(cand_ids, device=device)
    for i in range(0, len(prompts), batch_size):
        enc, pos = encode(tok, prompts[i:i + batch_size], device)
        with hooks(model, read_layers, edit) as store:
            logits = last_logits(model, enc, pos)
        all_logits.append(logits.index_select(-1, cand).float().cpu())
        for l in read_layers:
            acts[l].append(store[l])
    return torch.cat(all_logits), {l: torch.cat(v) for l, v in acts.items()}
