"""Step 1 - inspect the dataset, check it for problems and fix the train/val/test split."""
import numpy as np
from matplotlib.patches import Rectangle

from defect.data import CLASS_NAMES, CLASSES, DATA_DIR, ROOT, SPLIT, build_index, load_gray, write_splits
from defect.plots import BLUE, INK_2, ORANGE, plt, save

FIG = ROOT / "figures"

index = build_index()
print(f"{len(index)} images, {index['cls'].nunique()} classes")
print(index["cls"].value_counts().to_string(), "\n")

# Exact duplicate images would leak between train and test
dups = index[index.duplicated("md5", keep=False)].sort_values("md5")
print(f"exact duplicate images: {len(dups)}")
if len(dups):
    print(dups[["file", "cls", "md5"]].to_string(index=False))

# Defect boxes: how many per image and how much of the image they cover
index["n_boxes"] = index["boxes"].map(len)
index["coverage"] = index["boxes"].map(
    lambda bs: min(1.0, sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bs) / (200 * 200)))
images = {f: load_gray(DATA_DIR / "IMAGES" / f) for f in index["file"]}
index["mean_intensity"] = index["file"].map(lambda f: images[f].mean())
index["contrast"] = index["file"].map(lambda f: images[f].std())
stats = index.groupby("cls")[["n_boxes", "coverage", "mean_intensity", "contrast"]].mean().loc[CLASSES]
print("\nper-class means:\n", stats.round(2).to_string())

# Fixed split shared by every later experiment; one copy of each duplicate is dropped
# first so the same picture can never end up on both sides of the split
split = write_splits(index)
print(f"\n{SPLIT} split sizes:\n", split.groupby(["split", "cls"]).size().unstack().to_string())

# Figure 1: example images with their defect boxes
rng = np.random.default_rng(0)
n = 5
fig, axes = plt.subplots(len(CLASSES), n, figsize=(n * 1.6, len(CLASSES) * 1.75))
for r, cls in enumerate(CLASSES):
    rows = index[index["cls"] == cls].iloc[rng.choice(300, n, replace=False)]
    for c, (_, row) in enumerate(rows.iterrows()):
        ax = axes[r, c]
        ax.imshow(images[row["file"]], cmap="gray", vmin=0, vmax=255)
        for x1, y1, x2, y2 in row["boxes"]:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=ORANGE, lw=1.2))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[r, 0].set_ylabel(CLASS_NAMES[cls], rotation=0, ha="right", va="center", color=INK_2, fontsize=10)
fig.suptitle("NEU-DET: five random examples per class (orange = annotated defect box)", y=1.0)
fig.tight_layout()
save(fig, FIG / "01_examples.png")

# Figure 2: per-class distributions of box coverage and image statistics
cols = [("coverage", "Defect box coverage of image", "{:.0%}"),
        ("mean_intensity", "Mean pixel intensity (0-255)", "{:.0f}"),
        ("contrast", "Pixel std. dev. (contrast)", "{:.0f}")]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for ax, (col, title, fmt) in zip(axes, cols):
    for i, cls in enumerate(CLASSES):
        v = index.loc[index["cls"] == cls, col].to_numpy()
        jitter = rng.uniform(-0.18, 0.18, len(v))
        ax.scatter(v, i + jitter, s=6, color=BLUE, alpha=0.35, lw=0)
        ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=INK_2, lw=2)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    if col == "coverage":
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
axes[0].set_yticks(range(len(CLASSES)), [CLASS_NAMES[c] for c in CLASSES])
axes[0].invert_yaxis()
fig.text(0.01, -0.04, "Each dot is one image; dark tick = class median.", color=INK_2, fontsize=9)
fig.tight_layout()
save(fig, FIG / "01_class_statistics.png")
print("\nfigures written to", FIG)
