"""Step 5 - where does the network look? Grad-CAM, checked against the annotated defect boxes.

Pointing game: a heat map "hits" when its hottest pixel lies inside an annotated defect
box. Chance level for an image is the fraction of its area covered by boxes.
Also collects every misclassified test image for the failure-case section.
"""
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle

from defect.data import CLASS_NAMES, CLASSES, DATA_DIR, FIGURES, MODELS, RESULTS, read_boxes
from defect.explain import HEAT, grad_cam, load_checkpoint
from defect.plots import INK, INK_2, ORANGE, SERIES, SURFACE, plt, save
from defect.train import load_tensors, normalize

torch.set_num_threads(6)


def box_mask(boxes, size=200):
    m = np.zeros((size, size), bool)
    for x1, y1, x2, y2 in boxes:
        m[y1:y2 + 1, x1:x2 + 1] = True
    return m


data, _, _ = load_tensors()
X, y, df = data["test"]
masks = np.stack([box_mask(read_boxes(DATA_DIR / "ANNOTATIONS" / f.replace(".jpg", ".xml"))) for f in df["file"]])

nets = {"Small CNN": MODELS / "cnn_seed0.pt", "ResNet-18 fine-tuned": MODELS / "resnet18_seed0.pt"}
cams, probs, rows = {}, {}, []
for name, path in nets.items():
    model, layer, mean, std = load_checkpoint(path)
    c, p = zip(*[grad_cam(model, layer, normalize(X[i:i + 32], mean, std)) for i in range(0, len(X), 32)])
    cams[name], probs[name] = np.concatenate(c), np.concatenate(p)
    peak = cams[name].reshape(len(X), -1).argmax(1)
    hit = masks.reshape(len(X), -1)[np.arange(len(X)), peak]
    energy = (cams[name] * masks).sum((1, 2)) / cams[name].sum((1, 2))
    rows += [{"model": name, "cls": df["cls"][i], "hit": hit[i], "chance": masks[i].mean(),
              "energy_in_box": energy[i], "correct": probs[name][i].argmax() == y[i].item()} for i in range(len(X))]

res = pd.DataFrame(rows)
res.to_csv(RESULTS / "05_gradcam.csv", index=False)
summary = res.groupby(["model", "cls"])[["hit", "chance", "energy_in_box"]].mean().round(3)
print("pointing game per class (hit = hottest pixel inside a defect box; chance = box area share):")
print(summary.to_string())
print("\noverall:\n", res.groupby("model")[["hit", "chance", "energy_in_box", "correct"]].mean().round(3).to_string())


def show(ax, i, model_name, title):
    ax.imshow(X[i, 0], cmap="gray", vmin=0, vmax=255)
    ax.imshow(cams[model_name][i], cmap=HEAT, vmin=0, vmax=1)
    for x1, y1, x2, y2 in read_boxes(DATA_DIR / "ANNOTATIONS" / df["file"][i].replace(".jpg", ".xml")):
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=SERIES[0], lw=1.2))
    ax.set_title(title, fontsize=8, color=INK_2)
    ax.axis("off")


# Figure: two correctly classified examples per class, both models
rng = np.random.default_rng(1)
fig, axes = plt.subplots(len(CLASSES), 4, figsize=(4 * 2.1, len(CLASSES) * 2.2))
for r, cls in enumerate(CLASSES):
    ok = np.flatnonzero((df["cls"] == cls).to_numpy() &
                        (probs["Small CNN"].argmax(1) == y.numpy()) & (probs["ResNet-18 fine-tuned"].argmax(1) == y.numpy()))
    for k, i in enumerate(rng.choice(ok, 2, replace=False)):
        show(axes[r, 2 * k], i, "Small CNN", f"Small CNN · {df['file'][i]}")
        show(axes[r, 2 * k + 1], i, "ResNet-18 fine-tuned", "ResNet-18")
    axes[r, 0].text(-0.08, 0.5, CLASS_NAMES[cls], transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=10)
fig.suptitle("Grad-CAM for the predicted class (orange = where the network looks, blue box = annotated defect)", y=1.0)
fig.tight_layout()
FIGURES.mkdir(parents=True, exist_ok=True)
save(fig, FIGURES / "05_gradcam_examples.png")

# Figure: every misclassified test image (Small CNN, all three seeds pooled is in step 2;
# here seed 0 of each model, so the heat map belongs to the model that made the mistake)
wrong = [(name, i) for name in nets for i in np.flatnonzero(probs[name].argmax(1) != y.numpy())]
print(f"\nmisclassified test images (seed 0): {len(wrong)}")
if wrong:
    cols = min(5, len(wrong))
    nrows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 2.4, nrows * 2.7), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (name, i) in zip(axes.flat, wrong):
        p = probs[name][i]
        show(ax, i, name, "")
        ax.set_title(f"{name}\n{df['file'][i]}\ntrue {CLASS_NAMES[df['cls'][i]]}\n"
                     f"pred {CLASS_NAMES[CLASSES[p.argmax()]]} ({p.max():.0%})", fontsize=7.5, color=INK)
        print(f"  {name:22s} {df['file'][i]:24s} -> {CLASSES[p.argmax()]} ({p.max():.0%})")
    fig.suptitle("Failure cases: misclassified test images with the Grad-CAM of the wrong prediction", y=1.0)
    fig.tight_layout()
    save(fig, FIGURES / "05_failure_cases.png")
