"""Train a centreline model on chips from the "Export Training Chips" tool.

Meant for Google Colab (free GPU); also runs on a CPU, slowly. The notebook
dl/Train_Centerline_Colab.ipynb is built from this file (scripts/build_notebook.py).

Model: U-Net with an ImageNet-pretrained ResNet encoder (segmentation_models_pytorch),
taking R, G, B, NIR and drawing a ridge on the road centreline (the "heat" target).
The loss only counts pixels near hand-checked roads (the chip "mask").

Score: in validation blocks (ground never trained on), cross-sections are taken across
each checked line; the distance from the checked line to the crest of the
predicted ridge is the error. Compare it with baseline.json, the local roads'
error on the same roads.
"""
from __future__ import annotations

import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

DEFAULTS = dict(
    encoder="resnet34",
    encoder_weights="imagenet",   # None to train from scratch
    epochs=40,
    batch_size=16,
    lr=1e-3,
    weight_decay=1e-4,
    pos_weight=10.0,              # extra weight on the ridge, which is a thin target
    search_ft=15.0,               # look this far each side of the line for the crest
    min_peak=0.3,                 # a crest weaker than this counts as "not found"
    eval_every=5,                 # score the validation chips every this many epochs (and last)
    seed=0,
)


# --------------------------------------------------------------------------- data

def load_chips(folder: str):
    """{'train': dict, 'val': dict} with image (N,C,H,W) uint8, heat, mask, lines.
    Arrays are filled shard by shard, so memory holds one copy of the chips."""
    with open(os.path.join(folder, "manifest.json")) as f:
        manifest = json.load(f)
    out = {}
    for split in ("train", "val"):
        files = sorted(glob.glob(os.path.join(folder, f"shard_{split}_*.npz")))
        if not files:
            raise FileNotFoundError(f"No {split} shards in {folder}")
        sizes, lines = [], []
        for p in files:
            with open(p[:-4] + ".json") as f:
                lines += json.load(f)
            with np.load(p) as q:
                sizes.append(len(q["road"]))
                if len(sizes) == 1:
                    shapes = {k: q[k].shape[1:] for k in ("image", "heat", "mask")}
        n = sum(sizes)
        d = {k: np.empty((n,) + shapes[k], np.uint8) for k in shapes}
        d["road"], start = [], 0
        for p, size in zip(files, sizes):
            with np.load(p) as q:
                for k in shapes:
                    d[k][start:start + size] = q[k]
                d["road"] += q["road"].tolist()
            start += size
        d["road"] = np.array(d["road"])
        d["lines"] = lines
        out[split] = d
    return manifest, out


def channel_stats(images: np.ndarray, sample: int = 500):
    pick = np.linspace(0, len(images) - 1, min(sample, len(images))).astype(int)
    x = images[pick].astype(np.float32) / 255.0
    return x.mean(axis=(0, 2, 3)).tolist(), (x.std(axis=(0, 2, 3)) + 1e-6).tolist()


def to_device(data, device):
    """Chips as uint8 tensors on the GPU (about 0.4 MB each), so batches need no CPU
    work: free Colab has 2 CPU cores, which would otherwise leave the GPU idle."""
    keep = device if data["image"].nbytes < 6e9 else "cpu"
    return tuple(torch.from_numpy(data[k]).to(keep) for k in ("image", "heat", "mask"))


def make_batch(tensors, idx, mean, std, device, augment):
    """Normalised images, targets and masks for chips ``idx``; with ``augment``, a
    random rotation / mirror (the ridge turns with the image, so labels stay right)
    and a small brightness / contrast change per chip."""
    images, heat, mask = tensors
    idx = idx.to(images.device)
    x = images[idx].to(device).float() / 255.0
    y = heat[idx].to(device).float()[:, None] / 255.0
    m = mask[idx].to(device).float()[:, None]
    if augment:
        b = len(idx)
        k = torch.randint(0, 4, (b,), device=device)
        flip = torch.rand(b, device=device) < 0.5
        for t in (x, y, m):
            for kk in range(1, 4):
                sel = k == kk
                if sel.any():
                    t[sel] = torch.rot90(t[sel], kk, (2, 3))
            if flip.any():
                t[flip] = torch.flip(t[flip], (3,))
        x = x * (0.9 + 0.2 * torch.rand(b, 1, 1, 1, device=device)) \
            + 0.05 * (torch.rand(b, 1, 1, 1, device=device) - 0.5)
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)
    return (x - mean_t) / std_t, y, m


# --------------------------------------------------------------------------- model

def build_model(in_channels, encoder="resnet34", encoder_weights="imagenet"):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights,
                    in_channels=in_channels, classes=1)


def masked_loss(logits, target, mask, pos_weight):
    w = mask * (1.0 + pos_weight * target)
    loss = F.binary_cross_entropy_with_logits(logits, target, weight=w, reduction="sum")
    return loss / w.sum().clamp(min=1.0)


# --------------------------------------------------------------------------- scoring

def crest_errors(prob, lines, res, search_ft=15.0, min_peak=0.3, border=8, mask=None):
    """Distance (map units) from each checked line to the predicted ridge crest,
    measured across the line; NaN where no crest is found. With ``mask``, only
    points on mask pixels count (validation ground, away from line ends)."""
    h, w = prob.shape
    reach = search_ft / res
    offs = np.arange(-reach, reach + 0.25, 0.25)
    errs = []
    for line in lines:
        c = np.asarray(line, float)
        if len(c) < 3:
            continue
        t = np.gradient(c, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        n = np.column_stack([-t[:, 1], t[:, 0]])
        ok = ((c[:, 0] > border) & (c[:, 0] < w - 1 - border)
              & (c[:, 1] > border) & (c[:, 1] < h - 1 - border))
        if mask is not None:
            r = np.clip(np.round(c[:, 1]).astype(int), 0, h - 1)
            q = np.clip(np.round(c[:, 0]).astype(int), 0, w - 1)
            ok &= mask[r, q] > 0
        c, n = c[ok][::2], n[ok][::2]  # every ~4 px (2 ft)
        if not len(c):
            continue
        cols = c[:, :1] + offs[None] * n[:, :1]
        rows = c[:, 1:] + offs[None] * n[:, 1:]
        v = ndimage.map_coordinates(prob, [rows.ravel(), cols.ravel()], order=1,
                                    mode="constant").reshape(rows.shape)
        i = np.argmax(v, axis=1)
        k = np.arange(len(v))
        top = v[k, i]
        inner = (i > 0) & (i < len(offs) - 1)
        lo, hi = v[k, np.maximum(i - 1, 0)], v[k, np.minimum(i + 1, len(offs) - 1)]
        den = lo - 2 * top + hi
        use = inner & (den < 0)  # parabola through the top three samples
        sub = np.where(use, 0.5 * (lo - hi) / np.where(use, den, -1.0), 0.0)
        e = np.abs(offs[i] + sub * 0.25) * res
        e[top < min_peak] = np.nan
        errs.append(e)
    return np.concatenate(errs) if errs else np.zeros(0)


def summarize(errs):
    found = errs[~np.isnan(errs)]
    if not len(found):
        return dict(points=int(len(errs)), found=0.0)
    return dict(points=int(len(errs)), found=float(len(found) / len(errs)),
                median=float(np.median(found)), mean=float(found.mean()),
                p90=float(np.percentile(found, 90)),
                within05=float((found <= 0.5).mean()), within1=float((found <= 1).mean()),
                within2=float((found <= 2).mean()))


@torch.no_grad()
def predict_probs(model, images, mean, std, device, batch=16):
    model.eval()
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)
    out = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(images[i:i + batch]).to(device).float() / 255.0
        out.append(torch.sigmoid(model((x - mean_t) / std_t)).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def evaluate(model, val, mean, std, res, device, cfg, batch=64):
    errs = []
    for i in range(0, len(val["image"]), batch):  # in parts: 448+ chips of floats is big
        probs = predict_probs(model, val["image"][i:i + batch], mean, std, device)
        errs += [crest_errors(p, ln, res, cfg["search_ft"], cfg["min_peak"], mask=m)
                 for p, ln, m in zip(probs, val["lines"][i:i + batch], val["mask"][i:i + batch])]
    return summarize(np.concatenate(errs) if errs else np.zeros(0))


# --------------------------------------------------------------------------- training

def train(chips_folder, out_folder, **overrides):
    cfg = dict(DEFAULTS, **overrides)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    os.makedirs(out_folder, exist_ok=True)
    manifest, data = load_chips(chips_folder)
    res = float(manifest["res"])
    mean, std = channel_stats(data["train"]["image"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(data['train']['image'])} training / {len(data['val']['image'])} validation "
          f"chips, {len(mean)} channels, device {device}")
    baseline = {}
    bpath = os.path.join(chips_folder, "baseline.json")
    if os.path.exists(bpath):
        with open(bpath) as f:
            baseline = json.load(f).get("val", {})
        print(f"Score to beat (local roads in the validation blocks): median "
              f"{baseline.get('median', float('nan')):.2f}, within 1: "
              f"{baseline.get('within1', float('nan')):.0%}")

    model = build_model(len(mean), cfg["encoder"], cfg["encoder_weights"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * math.ceil(
            len(data["train"]["image"]) / cfg["batch_size"]))
    tensors = to_device(data["train"], device)
    n_train = len(data["train"]["image"])
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    ckpt = os.path.join(out_folder, "checkpoint.pt")
    start, best, history = 0, None, []
    if os.path.exists(ckpt):  # Colab disconnected: carry on where it stopped
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start, best, history = state["epoch"] + 1, state.get("best"), state.get("history", [])
        print(f"Resuming after epoch {start}")
    for epoch in range(start, cfg["epochs"]):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        order = torch.randperm(n_train)
        for i in range(0, n_train, cfg["batch_size"]):
            x, y, m = make_batch(tensors, order[i:i + cfg["batch_size"]], mean, std, device,
                                 augment=True)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                logits = model(x)
            loss = masked_loss(logits.float(), y, m, cfg["pos_weight"])
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total, n = total + loss.item() * len(x), n + len(x)
        line = f"epoch {epoch + 1}/{cfg['epochs']}: loss {total / max(n, 1):.4f}"
        last = epoch == cfg["epochs"] - 1
        if last or (epoch + 1) % cfg["eval_every"] == 0:
            score = evaluate(model, data["val"], mean, std, res, device, cfg)
            line += (f", validation median {score.get('median', float('nan')):.2f} ft, within "
                     f"1 ft {score.get('within1', 0):.0%}, crest found "
                     f"{score.get('found', 0):.0%}")
            if best is None or score.get("median", 1e9) < best.get("median", 1e9):
                best = dict(epoch=epoch, **score)
                torch.save(model.state_dict(), os.path.join(out_folder, "best_model.pt"))
        else:
            score = {}
        history.append(dict(epoch=epoch, loss=total / max(n, 1), **score))
        print(line + f" ({time.time() - t0:.0f}s)")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), epoch=epoch, best=best,
                        history=history), ckpt)
    info = dict(config=cfg, mean=mean, std=std, res=res, channels=manifest["channels"],
                chip_px=manifest["chip_px"], best=best, baseline=baseline, history=history)
    with open(os.path.join(out_folder, "model_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    report(best, baseline)
    return info


def report(best, baseline):
    if not best:
        return
    print("\nBest validation score (roads the model never saw):")
    print(f"  model:       median {best.get('median', float('nan')):.2f} ft, worst 10% "
          f"{best.get('p90', float('nan')):.2f} ft, within 0.5 ft {best.get('within05', 0):.0%},"
          f" within 1 ft {best.get('within1', 0):.0%}, crest found {best.get('found', 0):.0%}")
    if baseline:
        print(f"  local roads: median {baseline['median']:.2f} ft, worst 10% "
              f"{baseline['p90']:.2f} ft, within 1 ft {baseline['within1']:.0%}")
        better = best.get("median", 1e9) < baseline["median"]
        print("  -> the model is " + ("MORE" if better else "LESS")
              + " accurate than the local roads on these validation roads.")
