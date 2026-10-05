"""Overfitting / leakage check: are test images near-copies of training images?

Exact duplicates were removed in step 1. Here every test image is compared with every
training image in the feature space of an ImageNet ResNet-18 (cosine similarity), and
the nearest pairs are shown so they can be inspected by eye.
"""
import numpy as np
import torch

from defect.data import FIGURES, RESULTS
from defect.models import ResNet18Gray
from defect.plots import INK_2, plt, save
from defect.train import load_tensors, normalize

torch.set_num_threads(6)
data, mean, std = load_tensors()
model = ResNet18Gray().eval()
feats = {}
with torch.no_grad():
    for split in ("train", "test"):
        x = data[split][0]
        f = torch.cat([model.features(normalize(x[i:i + 64], mean, std)) for i in range(0, len(x), 64)])
        feats[split] = torch.nn.functional.normalize(f, dim=1).numpy()

sim_test = feats["test"] @ feats["train"].T            # test vs train
sim_train = feats["train"] @ feats["train"].T          # train vs train, as a reference
np.fill_diagonal(sim_train, -1)
nn_test, nn_train = sim_test.max(1), sim_train.max(1)
print(f"nearest training neighbour, cosine similarity (1 = identical features)")
print(f"  test  -> train: median {np.median(nn_test):.3f}, 99th pct {np.percentile(nn_test, 99):.3f}, max {nn_test.max():.3f}")
print(f"  train -> train: median {np.median(nn_train):.3f}, 99th pct {np.percentile(nn_train, 99):.3f}, max {nn_train.max():.3f}")
for t in (0.99, 0.98, 0.97):
    print(f"  test images with a training neighbour above {t}: {(nn_test > t).sum()} / {len(nn_test)}")

# Show the 6 most similar test/train pairs
test_df, train_df = data["test"][2], data["train"][2]
order = np.argsort(-nn_test)[:6]
fig, axes = plt.subplots(2, 6, figsize=(12, 4.6))
for c, i in enumerate(order):
    j = sim_test[i].argmax()
    for r, (img, name) in enumerate([(data["test"][0][i, 0], "test: " + test_df["file"][i]),
                                     (data["train"][0][j, 0], "train: " + train_df["file"][j])]):
        ax = axes[r, c]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name, fontsize=7.5, color=INK_2)
        ax.axis("off")
    axes[1, c].text(0.5, -0.12, f"similarity {nn_test[i]:.3f}", transform=axes[1, c].transAxes,
                    ha="center", fontsize=8.5)
fig.suptitle("Six test images most similar to a training image (top: test, bottom: its nearest training image)", y=1.0)
fig.tight_layout()
FIGURES.mkdir(parents=True, exist_ok=True)
save(fig, FIGURES / "02b_nearest_pairs.png")

# Does accuracy hold up on the test images that are least similar to the training set?
import pandas as pd
pred = pd.read_csv(RESULTS / "02_test_predictions.csv")
correct = np.mean([(pred[f"cnn_seed{s}_pred"] == pred["label"]).to_numpy() for s in range(3)], axis=0)
q = pd.qcut(nn_test, 4, labels=["least similar 25%", "2nd quarter", "3rd quarter", "most similar 25%"])
print("\nSmall CNN test accuracy by similarity to the training set:")
print(pd.Series(correct).groupby(q, observed=True).mean().round(4).to_string())
