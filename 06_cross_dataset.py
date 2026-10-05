"""Step 6 - test on a dataset the models have never seen: X-SDD (Feng et al., 2021).

X-SDD comes from a different hot-rolled steel line and camera. Three of its seven
classes correspond to NEU classes; the other four are defect types the models were
never trained on and are used to check whether low confidence flags them.

    slag inclusion                     -> inclusion
    oxide scale of temperature system  -> rolled-in scale
    surface scratch                    -> scratches   (dark lines on bright steel in X-SDD,
                                                       bright lines on dark steel in NEU)
"""
import numpy as np
import pandas as pd
import torch

from defect.data import CLASS_NAMES, CLASSES, FIGURES, MODELS, RESULTS
from defect.explain import load_checkpoint
from defect.plots import INK, INK_2, MUTED, SERIES, SURFACE, plt, save
from defect.train import load_tensors, normalize, predict_logits
from defect.xsdd import MAPPING, UNSEEN, load_xsdd

torch.set_num_threads(12)
SEEDS = (0, 1, 2)
NETS = {"Small CNN": "cnn", "Small CNN + photometric aug.": "cnn_photometric", "ResNet-18 fine-tuned": "resnet18"}
LOW_CONFIDENCE = 0.6

# X-SDD images: grayscale, resized from 128x128 to the 200x200 the networks were trained on
X, xcls, y_mapped, files = load_xsdd()
xmean, xstd = X.float().mean().item(), X.float().std().item()
neu, neu_mean, neu_std = load_tensors()
print(f"X-SDD: {len(files)} images, pixel mean {xmean:.1f} (NEU train: {neu_mean:.1f})")

mapped = np.isin(xcls, list(MAPPING))
target_idx = [CLASSES.index(c) for c in MAPPING.values()]

rows, pred_tables, conf = [], {}, []
for name, prefix in NETS.items():
    for seed in SEEDS:
        path = MODELS / f"{prefix}_seed{seed}.pt"
        if not path.exists():
            continue
        model, _, mean, std = load_checkpoint(path)
        variants = {
            "as is": predict_logits(model, X, mean, std),
            "X-SDD pixel statistics": predict_logits(model, X, xmean, xstd),
        }
        inv = X.clone()
        inv[xcls == "surface scratch"] = 255 - inv[xcls == "surface scratch"]
        variants["scratches inverted"] = predict_logits(model, inv, mean, std)
        for variant, logits in variants.items():
            p = logits.softmax(1).numpy()
            pred = p.argmax(1)
            restricted = np.array(target_idx)[p[:, target_idx].argmax(1)]
            row = {"model": name, "seed": seed, "variant": variant,
                   "acc_mapped": (pred[mapped] == y_mapped[mapped]).mean(),
                   "acc_mapped_3way": (restricted[mapped] == y_mapped[mapped]).mean()}
            for c, t in MAPPING.items():
                row[f"acc_{t}"] = (pred[xcls == c] == CLASSES.index(t)).mean()
            rows.append(row)
            if variant == "as is":
                pred_tables[(name, seed)] = pred
                conf.append({"model": name, "seed": seed, "group": "X-SDD, matching classes",
                             "conf": p[mapped].max(1)})
                conf.append({"model": name, "seed": seed, "group": "X-SDD, unseen defect types",
                             "conf": p[~mapped].max(1)})
        neu_p = predict_logits(model, neu["test"][0], mean, std).softmax(1).numpy()
        conf.append({"model": name, "seed": seed, "group": "NEU test (same source)", "conf": neu_p.max(1)})
        print(f"{name} seed {seed}: mapped accuracy {rows[-3]['acc_mapped']:.1%}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(RESULTS / "06_cross_dataset.csv", index=False)
cols = ["acc_mapped", "acc_mapped_3way"] + [f"acc_{t}" for t in MAPPING.values()]
summary = res.groupby(["model", "variant"], sort=False)[cols].mean().round(3)
print("\nX-SDD accuracy on the three matching classes (mean over seeds):")
print(summary.to_string())

# How confident are the models in and out of their training distribution?
conf_rows = [{"model": c["model"], "group": c["group"], "mean_conf": c["conf"].mean(),
              "flagged": (c["conf"] < LOW_CONFIDENCE).mean()} for c in conf]
conf_df = pd.DataFrame(conf_rows).groupby(["model", "group"], sort=False).mean().round(3)
conf_df.to_csv(RESULTS / "06_confidence.csv")
print(f"\nconfidence and share flagged for manual inspection (< {LOW_CONFIDENCE:.0%}):")
print(conf_df.to_string())

# Figure: where each X-SDD class ends up (best model by mapped accuracy, all seeds pooled)
best = res[res["variant"] == "as is"].groupby("model")["acc_mapped"].mean().idxmax()
order = list(MAPPING) + UNSEEN
counts = np.zeros((len(order), len(CLASSES)))
for (name, seed), pred in pred_tables.items():
    if name == best:
        for r, c in enumerate(order):
            counts[r] += np.bincount(pred[xcls == c], minlength=len(CLASSES))
share = counts / counts.sum(1, keepdims=True)
from matplotlib.colors import LinearSegmentedColormap
blues = LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                                "#256abf", "#184f95", "#0d366b"])
fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.imshow(share, cmap=blues, vmin=0, vmax=1, aspect="auto")
for r in range(len(order)):
    for c in range(len(CLASSES)):
        if share[r, c] >= 0.005:
            ax.text(c, r, f"{share[r, c]:.0%}", ha="center", va="center", fontsize=8.5,
                    color="white" if share[r, c] > 0.55 else INK)
for r, c in enumerate(MAPPING.values()):
    ax.add_patch(plt.Rectangle((CLASSES.index(c) - 0.5, r - 0.5), 1, 1, fill=False, ec=SERIES[1], lw=2))
ax.axhline(len(MAPPING) - 0.5, color=INK_2, lw=1.2)
ax.set_xticks(range(len(CLASSES)), [CLASS_NAMES[c] for c in CLASSES], rotation=30, ha="right")
ax.set_yticks(range(len(order)), [f"{c} (n={int((xcls == c).sum())})" for c in order])
ax.set_xlabel("Predicted NEU class")
ax.grid(False)
ax.set_title(f"X-SDD images through the NEU-trained {best} (3 seeds pooled)", loc="left")
fig.text(0.01, -0.03, "Orange box = the correct NEU class. Rows below the line are defect types that "
         "do not exist in NEU.", color=MUTED, fontsize=8.5)
FIGURES.mkdir(parents=True, exist_ok=True)
save(fig, FIGURES / "06_cross_dataset_confusion.png")

# Figure: confidence distributions, in-distribution vs X-SDD
present = [n for n in NETS if any(c["model"] == n for c in conf)]
fig, axes = plt.subplots(1, len(present), figsize=(4.4 * len(present), 3.4), sharey=True, squeeze=False)
axes = axes[0]
groups = ["NEU test (same source)", "X-SDD, matching classes", "X-SDD, unseen defect types"]
for ax, name in zip(axes, present):
    data = [np.concatenate([c["conf"] for c in conf if c["model"] == name and c["group"] == g]) for g in groups]
    bins = np.linspace(1 / 6, 1, 21)
    for d, g, color in zip(data, groups, SERIES):
        ax.hist(d, bins=bins, histtype="step", lw=2, color=color, density=True, label=g)
    ax.axvline(LOW_CONFIDENCE, color=MUTED, ls="--", lw=1)
    ax.set_title(name, loc="left")
    ax.set_xlabel("Confidence (max class probability)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
axes[0].set_ylabel("Density")
axes[0].legend(loc="upper left", fontsize=8)
fig.text(0.01, -0.04, f"Dashed line = {LOW_CONFIDENCE:.0%} threshold used by the demo to send a part to manual inspection.",
         color=MUTED, fontsize=8.5)
fig.tight_layout()
save(fig, FIGURES / "06_confidence.png")
print("figures written")
