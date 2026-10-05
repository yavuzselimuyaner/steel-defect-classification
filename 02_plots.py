"""Figures for step 2: model comparison, confusion matrices and learning curves."""
import json

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from defect.data import CLASS_NAMES, CLASSES, FIGURES, RESULTS
from defect.plots import BLUE, GRID, INK, INK_2, MUTED, ORANGE, SURFACE, plt, save

FIG = FIGURES
FIG.mkdir(parents=True, exist_ok=True)
# Sequential blue ramp, light -> dark
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

runs = pd.read_csv(RESULTS / "02_runs.csv")
test = runs[runs["split"] == "test"]
order = list(dict.fromkeys(test["model"]))

# Figure: test accuracy per model, mean +- std over seeds
fig, ax = plt.subplots(figsize=(7.5, 2.8))
for i, model in enumerate(order):
    acc = test.loc[test["model"] == model, "accuracy"].to_numpy()
    m, s = acc.mean(), acc.std(ddof=1) if len(acc) > 1 else 0.0
    ax.errorbar(m, i, xerr=s, fmt="o", color=BLUE, ms=8, capsize=4, lw=2, mec=SURFACE, mew=2)
    label = f"{m:.1%}" + (f" ± {s:.1%}" if len(acc) > 1 else "")
    ax.text(m + max(s, 0.01) + 0.015, i, label, va="center", color=INK, fontsize=9)
ax.axvline(1 / 6, color=MUTED, lw=1, ls="--")
ax.text(1 / 6 + 0.01, len(order) - 1.5, "chance (16.7%)", color=MUTED, fontsize=8, va="center")
ax.set_yticks(range(len(order)), order)
ax.invert_yaxis()
ax.set_xlim(0, 1.12)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}" if x <= 1 else ""))
ax.grid(axis="y", visible=False)
ax.set_title("Test accuracy (360 images; neural models: mean ± std over 3 seeds)", loc="left")
save(fig, FIG / "02_model_comparison.png")

# Figure: row-normalized confusion matrices (summed over seeds)
names = [CLASS_NAMES[c] for c in CLASSES]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (arch, title) in zip(axes, [("mlp", "MLP"), ("cnn", "Small CNN")]):
    cm = np.load(RESULTS / f"02_confusion_{arch}.npy")
    pct = cm / cm.sum(1, keepdims=True)
    ax.imshow(pct, cmap=BLUES, vmin=0, vmax=1)
    for r in range(6):
        for c in range(6):
            if cm[r, c]:
                ax.text(c, r, f"{pct[r, c]:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if pct[r, c] > 0.55 else INK)
    ax.set_xticks(range(6), names, rotation=35, ha="right")
    ax.set_yticks(range(6), names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.grid(False)
    ax.set_title(f"{title}: share of each true class (3 seeds pooled)", loc="left")
    ax.set_xticks(np.arange(-.5, 6), minor=True)
    ax.set_yticks(np.arange(-.5, 6), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="minor", length=0)
fig.tight_layout()
save(fig, FIG / "02_confusion.png")

# Figure: learning curves (seed 0)
hist = json.load(open(RESULTS / "02_histories.json"))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, (key, title) in zip(axes, [("mlp_seed0", "MLP"), ("cnn_seed0", "Small CNN")]):
    h = pd.DataFrame(hist[key])
    ends = {"train_loss": h["train_loss"].iloc[-1], "val_loss": h["val_loss"].iloc[-1]}
    # Push the two end labels apart when the curves finish close together
    gap = 0.09
    if abs(ends["val_loss"] - ends["train_loss"]) < gap:
        mid = (ends["val_loss"] + ends["train_loss"]) / 2
        ends = {"train_loss": mid - gap / 2, "val_loss": mid + gap / 2}
    for col, color, label in [("train_loss", BLUE, "train"), ("val_loss", ORANGE, "validation")]:
        ax.plot(h["epoch"], h[col], color=color, label=label)
        ax.text(h["epoch"].iloc[-1] + 0.8, ends[col], label, color=INK_2, va="center", fontsize=9)
    best = h["val_loss"].idxmin()
    ax.plot(h["epoch"][best], h["val_loss"][best], "o", color=ORANGE, ms=8, mec=SURFACE, mew=2)
    ax.set_title(f"{title}: cross-entropy loss per epoch (seed 0)", loc="left")
    ax.set_xlabel("Epoch")
    ax.set_xlim(0, h["epoch"].iloc[-1] + 9)
axes[0].set_ylabel("Loss")
axes[0].legend(loc="upper right")
fig.text(0.01, -0.03, "Dot = epoch with the lowest validation loss; those weights are kept (early stopping).",
         color=INK_2, fontsize=9)
fig.tight_layout()
save(fig, FIG / "02_learning_curves.png")
print("figures written")
