"""One-time edit of model_utils.load: put the model straight onto the GPU while loading.

Without it, from_pretrained first builds the whole model in host RAM, which for Qwen2.5-72B
is ~145 GB and may not fit. Run once in the repo:  python3 patch_model_utils.py
"""
src = open("model_utils.py").read()
old = """    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, **{dtype_kw: dtype})
    model.to(device).eval()"""
new = """    kw = {dtype_kw: dtype}
    if str(device).startswith("cuda"):     # load straight onto the GPU (a 72B model may not fit in host RAM)
        kw.update(device_map=device, low_cpu_mem_usage=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, **kw)
    if "device_map" not in kw:
        model.to(device)
    model.eval()"""
if new in src:
    print("model_utils.py already patched")
else:
    assert src.count(old) == 1, "model_utils.load does not look as expected: send it to Claude, nothing changed"
    open("model_utils.py", "w").write(src.replace(old, new))
    print("model_utils.py patched")
