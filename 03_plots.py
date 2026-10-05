"""Figure for step 3: test accuracy against the number of training images per class."""
import pandas as pd

from defect.data import FIGURES, RESULTS, SPLIT
from defect.plots import INK_2, MUTED, SERIES, SURFACE, plt, save

runs = pd.read_csv(RESULTS / "03_runs.csv")
step2 = pd.read_csv(RESULTS / "02_runs.csv")
FULL = 180  # training images per class in the full split
runs["n"] = runs["per_class"].replace("all", FULL).astype(int)

fig, ax = plt.subplots(figsize=(8.5, 4.4))
curves = [("ResNet-18 linear probe (frozen)", "ImageNet ResNet-18, frozen + logistic regression", SERIES[0]),
          ("Small CNN (from scratch)", "Small CNN trained from scratch", SERIES[1])]
for model, label, color in curves:
    s = runs[runs["model"] == model].groupby("n")["accuracy"].agg(["mean", "min", "max"]).reset_index()
    ax.fill_between(s["n"], s["min"], s["max"], color=color, alpha=0.15, lw=0)
    ax.plot(s["n"], s["mean"], color=color, marker="o", ms=7, mec=SURFACE, mew=1.5, label=label)

# Full-data reference points from step 2 (early stopping on the validation set)
cnn_full = step2[(step2["model"] == "Small CNN") & (step2["split"] == "test")]["accuracy"].mean()
ft = runs[runs["model"] == "ResNet-18 fine-tuned"]["accuracy"].mean()
ax.plot(FULL, cnn_full, marker="s", ms=8, color=SERIES[1], mec=SURFACE, mew=1.5, ls="none",
        label="Small CNN, full data + early stopping (step 2)")
ax.plot(FULL, ft, marker="D", ms=8, color=SERIES[2], mec=SURFACE, mew=1.5, ls="none",
        label="ResNet-18 fine-tuned, full data")
# Keep the two end labels apart when the values are close
y_ft, y_cnn = ft, cnn_full
if abs(y_ft - y_cnn) < 0.012:
    mid = (y_ft + y_cnn) / 2
    y_ft, y_cnn = (mid + 0.006, mid - 0.006) if ft >= cnn_full else (mid - 0.006, mid + 0.006)
ax.text(FULL * 1.08, y_ft, f"{ft:.1%}", va="center", fontsize=9, color=INK_2)
ax.text(FULL * 1.08, y_cnn, f"{cnn_full:.1%}", va="center", fontsize=9, color=INK_2)

ax.set_xscale("log")
ticks = [5, 10, 25, 50, FULL]
ax.set_xticks(ticks, [str(t) for t in ticks])
ax.minorticks_off()
ax.set_xlim(4, FULL * 1.6)
ax.set_xlabel("Training images per class (log scale)")
ax.set_ylabel("Test accuracy")
ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_title(f"How much labelled data is needed? ({SPLIT} split)", loc="left")
ax.legend(loc="lower right", fontsize=8.5)
fig.text(0.01, -0.03, "Line = mean over 3 random subsets, band = min-max. Small-data runs use no validation set "
         "(fixed 300 training steps).", color=MUTED, fontsize=8.5)
FIGURES.mkdir(parents=True, exist_ok=True)
save(fig, FIGURES / "03_data_efficiency.png")
print("figure written")
