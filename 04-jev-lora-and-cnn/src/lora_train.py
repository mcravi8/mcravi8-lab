"""Run ON THE POD. LoRA fine-tune of JEV-27B-VL System 1 for weekly charts.

The model card documents no fine-tuning recipe, so this follows the System 1
format exactly: the decision LoRA (adapter_vllm, r=32) is merged into the base
weights, a new small LoRA (r=16) is added to the language model, and training
minimizes cross-entropy over the three choice-token logits (A/B/C = Up/Flat/
Down) after the decision head's bias and calibration temperature, i.e. the same
readout /v1/decide uses. Prompt is identical to Experiments 02/03.
Early stopping on validation log loss (last ~20% of train weeks).
"""
import argparse, json, math, random, re, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model

MD = Path("/workspace/models/JEV-27B-VL")
LABELS = ["Up", "Flat", "Down"]
QUESTION = ("This is a 5-day daily stock chart covering one trading week. Over the next week, "
            "will the price go Up, stay Flat, or go Down?")
PROMPT = ("[kind] choice\n[state] Chart: <|vision_start|><|image_pad|><|vision_end|>"
          f"\n[question] {QUESTION}\n[options]\nA) Up\nB) Flat\nC) Down\n[decision]:")

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/root/d04")
ap.add_argument("--out", default="/workspace/exp04")
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--evals_per_epoch", type=int, default=2)
ap.add_argument("--patience", type=int, default=2)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--diag", action="store_true")
ap.add_argument("--skip_merge", default="", help="comma list of module-name substrings not to merge")
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0); random.seed(0)

dh = json.load(open(MD / "adapter_vllm/decision_head.json"))
T = json.load(open(MD / "calibration.json"))["per_kind"]["choice"]
s = dh["slots"]["ranges"]["choice"][0]
IDS = dh["verbalizer_ids"][s:s + 3]
BIAS = torch.tensor(dh["bias"][s:s + 3], dtype=torch.float32, device="cuda")
print("choice ids", IDS, "T", T, flush=True)

proc = AutoProcessor.from_pretrained(MD)
model = AutoModelForImageTextToText.from_pretrained(MD, dtype=torch.bfloat16, device_map="cuda")
cfg = model.config
tied = getattr(cfg, "tie_word_embeddings", False) or getattr(getattr(cfg, "text_config", cfg), "tie_word_embeddings", False)
print("tied embeddings:", tied, flush=True)
if tied:
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())

# merge the decision LoRA
ad = load_file(MD / "adapter_vllm/adapter_model.safetensors")
acfg = json.load(open(MD / "adapter_vllm/adapter_config.json"))
scale = acfg["lora_alpha"] / acfg["r"]
mods = dict(model.named_modules())
merged = 0
for k in ad:
    if ".lora_A." not in k:
        continue
    path = k.replace("base_model.model.", "", 1).split(".lora_A.")[0]
    cands = [path, path.replace("model.layers.", "model.language_model.layers.", 1),
             path.replace("model.", "model.language_model.", 1)]
    target = next((c for c in cands if c in mods and hasattr(mods[c], "weight")), None)
    if target is None:
        raise SystemExit(f"cannot map adapter key {k}")
    if args.skip_merge and any(x in target for x in args.skip_merge.split(",")):
        continue
    A = ad[k].to("cuda", torch.float32); Bm = ad[k.replace("lora_A", "lora_B")].to("cuda", torch.float32)
    W = mods[target].weight
    W.data = (W.data.float() + scale * (Bm @ A)).to(W.dtype)
    merged += 1
print("merged decision LoRA modules:", merged, flush=True)
del ad

idx = pd.read_csv(Path(args.data) / "index.csv")
img_cache = {}
def image(f, mirror=False):
    im = Image.open(Path(args.data) / f).convert("RGB")
    return ImageOps.mirror(im) if mirror else im

def logits3(f, mirror=False):
    inp = proc(text=[PROMPT], images=[image(f, mirror)], return_tensors="pt", add_special_tokens=False).to("cuda")
    o = model(**inp, logits_to_keep=1)
    return (o.logits[0, -1, IDS].float() + BIAS) / T

def score(rows, mirror=False):
    model.eval(); P = []
    with torch.no_grad():
        for f in rows.file:
            P.append(torch.softmax(logits3(f, mirror), 0).cpu().numpy())
    return np.array(P)

def logloss(P, labels):
    y = labels.map(LABELS.index).to_numpy()
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)).mean()), float((P.argmax(1) == y).mean())

fit = idx[idx.fit_split == "fit"].reset_index(drop=True)
val = idx[idx.fit_split == "val"].reset_index(drop=True)
test = idx[idx.split == "test"].reset_index(drop=True)
if args.smoke:
    fit, val, test = fit.head(16), val.head(8), test.head(12)

# sanity: merged model zero-shot vs vLLM 03 predictions
t0 = time.time()
P0 = score(test.head(12))
ref = pd.read_csv(Path(args.data) / "zs_preds.csv").set_index("file").loc[test.head(12).file, ["p_up", "p_flat", "p_down"]].to_numpy()
if args.diag:
    print(np.round(np.c_[P0, ref], 3)); raise SystemExit
print("zero-shot sanity max abs diff vs vLLM:", float(np.abs(P0 - ref).max()), f"({time.time()-t0:.1f}s for 12)", flush=True)

model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
pat = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)$"
model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules=pat))
model.print_trainable_parameters()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
steps_per_epoch = math.ceil(len(fit) / args.accum)
total = steps_per_epoch * args.epochs
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, (st + 1) / 10) * max(0.0, 1 - st / total))
eval_every = max(1, len(fit) // args.evals_per_epoch)

Pv = score(val); vl, va = logloss(Pv, val.label)
hist = [{"examples_seen": 0, "epoch": 0.0, "train_loss": None, "val_logloss": vl, "val_acc": va}]
print(f"step0 val logloss {vl:.4f} acc {va:.3f}", flush=True)
best, best_state, bad = vl, None, 0
lossf = torch.nn.CrossEntropyLoss()
seen, run, nrun, stop = 0, 0.0, 0, False
for ep in range(args.epochs):
    order = list(range(len(fit))); random.Random(ep).shuffle(order)
    model.train()
    for j, i in enumerate(order):
        r = fit.iloc[i]
        z = logits3(r.file)
        loss = lossf(z[None], torch.tensor([LABELS.index(r.label)], device="cuda"))
        (loss / args.accum).backward()
        run += loss.item(); nrun += 1; seen += 1
        if seen % args.accum == 0 or j == len(order) - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad()
        if seen % eval_every == 0:
            Pv = score(val); vl, va = logloss(Pv, val.label); model.train()
            hist.append({"examples_seen": seen, "epoch": seen / len(fit), "train_loss": run / nrun, "val_logloss": vl, "val_acc": va})
            print(json.dumps(hist[-1]), f"{time.time()-t0:.0f}s", flush=True)
            run, nrun = 0.0, 0
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
            else:
                bad += 1
                if bad >= args.patience:
                    stop = True; break
    if stop:
        break
if best_state is not None:
    model.load_state_dict(best_state, strict=False)
    best_at = min((h for h in hist), key=lambda h: h["val_logloss"])
else:
    best_at = hist[0]
    with torch.no_grad():  # no improvement over zero-shot: zero the new LoRA
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.zero_()
print("best checkpoint:", best_at, flush=True)
if best_state is not None:
    model.save_pretrained(str(out / f"lora_r{args.rank}"))
res = {}
for name, rows, mirror in (("val", val, False), ("test", test, False), ("test_placebo", test, True)):
    P = score(rows, mirror)
    res[name] = pd.DataFrame({"file": rows.file, "p_up": P[:, 0], "p_flat": P[:, 1], "p_down": P[:, 2]})
t = res["test"].merge(res["test_placebo"].rename(columns={"p_up": "p_up_pl", "p_flat": "p_flat_pl", "p_down": "p_down_pl"}), on="file")
t.to_csv(out / "lora_test_preds.csv", index=False, float_format="%.6g")
res["val"].to_csv(out / "lora_val_preds.csv", index=False, float_format="%.6g")
json.dump({"args": vars(args), "choice_ids": IDS, "temperature": T, "merged_decision_lora_modules": merged,
           "zero_shot_sanity_max_abs_diff": float(np.abs(P0 - ref).max()), "best": best_at,
           "history": hist, "seconds": time.time() - t0}, open(out / "lora_training.json", "w"), indent=1)
print("DONE", time.time() - t0, flush=True)
