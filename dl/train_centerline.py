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
    num_workers=2,
    seed=0,
)


# --------------------------------------------------------------------------- data

def load_chips(folder: str):
    """{'train': dict, 'val': dict} with image (N,C,H,W) uint8, heat, mask, lines."""
    with open(os.path.join(folder, "manifest.json")) as f:
        manifest = json.load(f)
    out = {}
    for split in ("train", "val"):
        files = sorted(glob.glob(os.path.join(folder, f"shard_{split}_*.npz")))
        if not files:
            raise FileNotFoundError(f"No {split} shards in {folder}")
        parts = [np.load(p) for p in files]
        lines = []
        for p in files:
            with open(p[:-4] + ".json") as f:
                lines += json.load(f)
        out[split] = dict(image=np.concatenate([q["image"] for q in parts]),
                          heat=np.concatenate([q["heat"] for q in parts]),
                          mask=np.concatenate([q["mask"] for q in parts]),
                          road=np.concatenate([q["road"] for q in parts]),
                          lines=lines)
    return manifest, out


def channel_stats(images: np.ndarray):
    x = images.astype(np.float32) / 255.0
    return x.mean(axis=(0, 2, 3)).tolist(), (x.std(axis=(0, 2, 3)) + 1e-6).tolist()


class ChipDataset(torch.utils.data.Dataset):
    def __init__(self, data, mean, std, augment=False):
        self.d, self.augment = data, augment
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __len__(self):
        return len(self.d["image"])

    def __getitem__(self, i):
        x = torch.from_numpy(self.d["image"][i]).float() / 255.0
        y = torch.from_numpy(self.d["heat"][i]).float()[None] / 255.0
        m = torch.from_numpy(self.d["mask"][i]).float()[None]
        if self.augment:  # the ridge rotates with the image, so these are free labels
            k = int(torch.randint(0, 4, (1,)))
            x, y, m = (torch.rot90(t, k, (1, 2)) for t in (x, y, m))
            if torch.rand(1) < 0.5:
                x, y, m = (torch.flip(t, (2,)) for t in (x, y, m))
            x = x * (0.9 + 0.2 * torch.rand(1)) + 0.05 * (torch.rand(1) - 0.5)  # light/contrast
        return (x - self.mean) / self.std, y, m


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
        for (cx, cy), (nx, ny) in zip(c[ok][::2], n[ok][::2]):  # every ~4 px (2 ft)
            cols, rows = cx + offs * nx, cy + offs * ny
            v = ndimage.map_coordinates(prob, [rows, cols], order=1, mode="constant")
            i = int(np.argmax(v))
            if v[i] < min_peak:
                errs.append(np.nan)
                continue
            sub = 0.0
            if 0 < i < len(v) - 1:  # parabola through the top three samples
                den = v[i - 1] - 2 * v[i] + v[i + 1]
                if den < 0:
                    sub = 0.5 * (v[i - 1] - v[i + 1]) / den
            errs.append(abs(offs[i] + sub * 0.25) * res)
    return np.array(errs)


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


def evaluate(model, val, mean, std, res, device, cfg):
    probs = predict_probs(model, val["image"], mean, std, device)
    errs = [crest_errors(p, ln, res, cfg["search_ft"], cfg["min_peak"], mask=m)
            for p, ln, m in zip(probs, val["lines"], val["mask"])]
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
    loader = torch.utils.data.DataLoader(
        ChipDataset(data["train"], mean, std, augment=True), batch_size=cfg["batch_size"],
        shuffle=True, num_workers=cfg["num_workers"], drop_last=False)
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
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                logits = model(x)
            loss = masked_loss(logits.float(), y, m, cfg["pos_weight"])
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total, n = total + loss.item() * len(x), n + len(x)
        score = evaluate(model, data["val"], mean, std, res, device, cfg)
        history.append(dict(epoch=epoch, loss=total / max(n, 1), **score))
        print(f"epoch {epoch + 1}/{cfg['epochs']}: loss {total / max(n, 1):.4f}, validation "
              f"median {score.get('median', float('nan')):.2f} ft, within 1 ft "
              f"{score.get('within1', 0):.0%}, crest found {score.get('found', 0):.0%} "
              f"({time.time() - t0:.0f}s)")
        is_best = best is None or score.get("median", 1e9) < best.get("median", 1e9)
        if is_best:
            best = dict(epoch=epoch, **score)
            torch.save(model.state_dict(), os.path.join(out_folder, "best_model.pt"))
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
