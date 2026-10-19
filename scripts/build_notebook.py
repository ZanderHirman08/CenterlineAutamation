"""Build the Colab notebook dl/Train_Centerline_Colab.ipynb from dl/train_centerline.py.

    python scripts/build_notebook.py          # rebuild after changing the training code
    python scripts/build_notebook.py --check  # fail if the committed notebook is out of date
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "dl" / "train_centerline.py"
OUT = ROOT / "dl" / "Train_Centerline_Colab.ipynb"


def md(text):
    return dict(cell_type="markdown", metadata={}, source=text.strip("\n").splitlines(True))


def code(text):
    return dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                source=text.strip("\n").splitlines(True))


def build():
    cells = [
        md("""
# Train the centerline model

1. **Runtime > Change runtime type > T4 GPU**, then **Runtime > Run all**.
2. Allow Google Drive access when asked.
3. Training prints one line per epoch. If Colab disconnects, run all again: it
   resumes from the last finished epoch.

Expects `chips.zip` (from the toolbox tool **Export Training Chips**) in the Drive
folder set below. Results go to the `model` folder next to it.
"""),
        code("""
# Settings: where chips.zip is on your Google Drive
DRIVE_FOLDER = "/content/drive/MyDrive/Centerline"
CHIPS_ZIP = DRIVE_FOLDER + "/chips.zip"
MODEL_FOLDER = DRIVE_FOLDER + "/model"
EPOCHS = 40
"""),
        code("""
from google.colab import drive
drive.mount("/content/drive")
!pip -q install segmentation-models-pytorch==0.3.3
import os, shutil, zipfile
if os.path.exists("/content/chips"):
    shutil.rmtree("/content/chips")
zipfile.ZipFile(CHIPS_ZIP).extractall("/content/chips")
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else "NONE - set Runtime > Change runtime type > T4 GPU")
"""),
        md("The training code (same as `dl/train_centerline.py` in the repository):"),
        code("%%writefile /content/train_centerline.py\n" + SRC.read_text()),
        code("""
import sys
sys.path.insert(0, "/content")
import train_centerline as tc
info = tc.train("/content/chips", MODEL_FOLDER, epochs=EPOCHS)
"""),
        code("""
import matplotlib.pyplot as plt
h = info["history"]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot([r["loss"] for r in h]); ax[0].set_title("training loss"); ax[0].set_xlabel("epoch")
ax[1].plot([r.get("median", float("nan")) for r in h], label="model")
if info["baseline"]:
    ax[1].axhline(info["baseline"]["median"], color="r", ls="--", label="local roads")
ax[1].set_title("validation median error (ft)"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.show()
"""),
        md("""
## Look at validation chips

Left: imagery with your checked line (red). Middle: the model's prediction
(bright = centreline). Right: your line on the prediction. First a table of checked roads (validation
and training) where the model disagrees with the line, worth a look in Pro:
sometimes the checked line is the one that's off. Then 4 chips spread through
the validation set, then the 4 where the model is furthest off.
"""),
        code("""
import numpy as np
_, data = tc.load_chips("/content/chips")
val = data["val"]
model = tc.build_model(len(info["mean"]), info["config"]["encoder"], None)
model.load_state_dict(torch.load(MODEL_FOLDER + "/best_model.pt", map_location="cpu"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.to(dev)
res = info["res"]
import pandas as pd

def chip_errors(d):
    # median distance (ft) from each chip's checked line to the model's centreline
    out = []
    for i in range(0, len(d["image"]), 64):
        probs = tc.predict_probs(model, d["image"][i:i + 64], info["mean"], info["std"], dev)
        for k, p in enumerate(probs):
            e = tc.crest_errors(p, d["lines"][i + k], res, mask=d["mask"][i + k])
            e = e[~np.isnan(e)]
            out.append(np.median(e) if len(e) else np.nan)
    return np.array(out)

def by_road(d, err, name):
    half = d["image"].shape[-1] * res / 2
    df = pd.DataFrame(dict(road_oid=d["road"], err=err, x=d["x0"] + half, y=d["y0"] - half))
    t = (df.dropna().groupby("road_oid")
         .agg(chips=("err", "size"), median_error_ft=("err", "median"),
              x=("x", "median"), y=("y", "median")))
    t.insert(0, "set", name)
    return t

chip_err = chip_errors(val)
train_err = chip_errors(data["train"])
has = np.flatnonzero(~np.isnan(chip_err))
spread = has[np.linspace(0, len(has) - 1, 4).astype(int)]
worst = has[np.argsort(chip_err[has])[-4:][::-1]]

# Checked roads the model disagrees with: either the model is wrong there, or the
# checked line is. Zoom to the coordinates in Pro (same coordinate system as the
# imagery) and look. 15.00 means "15 ft or more" (the edge of the search).
# Training roads were learned from, so a bad line there shows a smaller gap:
# their threshold is lower.
roads = pd.concat([by_road(val, chip_err, "validation"),
                   by_road(data["train"], train_err, "training")])
suspect = roads[((roads.set == "validation") & (roads.median_error_ft > 3))
                | ((roads.set == "training") & (roads.median_error_ft > 2))]
suspect = suspect.sort_values("median_error_ft", ascending=False)
print(f"{len(suspect)} of {len(roads)} checked roads where the model and the line disagree "
      "(over 3 ft on validation roads, over 2 ft on training roads). Check these in Pro:")
print(suspect.round(2).to_string() if len(suspect) else "  none")
oids = ", ".join(dict.fromkeys(str(o) for o in suspect.index))
if oids:
    print()
    print(f"Select By Attributes on Roads_working:  OBJECTID IN ({oids})")

def show(i):
    p = tc.predict_probs(model, val["image"][i:i + 1], info["mean"], info["std"], dev)[0]
    h, w = p.shape
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
    ax[0].imshow(np.moveaxis(val["image"][i][:3], 0, -1))
    ax[1].imshow(p, vmin=0, vmax=1)
    ax[2].imshow(p, vmin=0, vmax=1, cmap="gray")
    for ln in val["lines"][i]:
        ln = np.asarray(ln)
        for a in (ax[0], ax[2]):
            a.plot(ln[:, 0], ln[:, 1], "r-", lw=1)
    for a in ax:
        a.set_xlim(-0.5, w - 0.5)
        a.set_ylim(h - 0.5, -0.5)
        a.axis("off")
    fig.suptitle(f"chip {i}: median error {chip_err[i]:.2f} ft")
    plt.show()

print("Spread through the validation set:")
for i in spread:
    show(i)
print("Where the model is furthest from your line:")
for i in worst:
    show(i)
"""),
    ]
    nb = dict(cells=cells, nbformat=4, nbformat_minor=5,
              metadata=dict(accelerator="GPU", colab=dict(provenance=[]),
                            kernelspec=dict(name="python3", display_name="Python 3"),
                            language_info=dict(name="python")))
    return json.dumps(nb, indent=1) + "\n"


if __name__ == "__main__":
    text = build()
    if "--check" in sys.argv:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit(f"{OUT} is out of date: run python scripts/build_notebook.py")
        print("up to date")
    else:
        OUT.write_text(text)
        print(f"wrote {OUT}")
