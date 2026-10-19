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
(bright = centreline). Right: your line on the prediction. First 4 spread
through the validation set, then the 4 where the model is furthest off.
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
chip_err = []
for i in range(0, len(val["image"]), 64):
    probs = tc.predict_probs(model, val["image"][i:i + 64], info["mean"], info["std"], dev)
    for k, p in enumerate(probs):
        e = tc.crest_errors(p, val["lines"][i + k], res, mask=val["mask"][i + k])
        e = e[~np.isnan(e)]
        chip_err.append(np.median(e) if len(e) else np.nan)
chip_err = np.array(chip_err)
has = np.flatnonzero(~np.isnan(chip_err))
spread = has[np.linspace(0, len(has) - 1, 4).astype(int)]
worst = has[np.argsort(chip_err[has])[-4:][::-1]]

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
