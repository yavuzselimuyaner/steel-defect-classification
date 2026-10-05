"""Step 4 - does the network recognize the defect, or the lighting?

Test images are degraded the way a real inspection camera might degrade them
(exposure, contrast, blur, sensor noise) and every model is re-evaluated.
A Small CNN trained with photometric augmentation is added to see whether the
sensitivity can be trained away.
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from defect.data import FIGURES, MODELS, RESULTS
from defect.explain import perturb
from defect.models import ResNet18Gray, SmallCNN
from defect.plots import INK_2, MUTED, SERIES, SURFACE, plt, save
from defect.train import fit, load_tensors, photometric, predict_logits

FIG, LOGS = FIGURES, RESULTS / "logs"
SEEDS = (0, 1, 2)
PERTURBATIONS = {
    "exposure": ("Exposure (brightness gain)", [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5], 1.0),
    "contrast": ("Contrast factor", [0.4, 0.6, 0.8, 1.0, 1.25, 1.5], 1.0),
    "blur": ("Gaussian blur sigma (px)", [0, 0.5, 1, 1.5, 2, 3], 0),
    "noise": ("Gaussian noise sigma (gray levels)", [0, 5, 10, 15, 20, 30], 0),
}
MODEL_ORDER = ["Brightness + contrast (2 features)", "Small CNN", "Small CNN + photometric aug.",
               "ResNet-18 fine-tuned"]


def train_augmented(seed):
    torch.set_num_threads(4)
    data, mean, std = load_tensors()
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / f"04_cnn_photometric_seed{seed}.log", "w", encoding="utf-8") as f:
        def log(line):
            f.write(line + "\n")
            f.flush()
        model, _ = fit(SmallCNN(), data["train"][:2], data["val"][:2], mean, std,
                       epochs=40, patience=10, augment=photometric, seed=seed, log=log)
    torch.save({"state": model.state_dict(), "mean": mean, "std": std}, MODELS / f"cnn_photometric_seed{seed}.pt")
    return seed


def load(path, model):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["state"])
    return model.eval(), ckpt["mean"], ckpt["std"]


def evaluate():
    torch.set_num_threads(12)
    data, _, _ = load_tensors()
    X_tr, y_tr = data["train"][:2]
    X_te, y_te = data["test"][:2]
    y_te = y_te.numpy()

    def stats(x):
        x = x.float().flatten(1)
        return torch.stack([x.mean(1), x.std(1)], 1).numpy()
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(stats(X_tr), y_tr.numpy())

    nets = {"Small CNN": [load(MODELS / f"cnn_seed{s}.pt", SmallCNN()) for s in SEEDS],
            "Small CNN + photometric aug.": [load(MODELS / f"cnn_photometric_seed{s}.pt", SmallCNN()) for s in SEEDS],
            "ResNet-18 fine-tuned": [load(MODELS / f"resnet18_seed{s}.pt", ResNet18Gray(pretrained=False)) for s in SEEDS]}
    rows = []
    for kind, (_, levels, _) in PERTURBATIONS.items():
        for level in levels:
            x = perturb(X_te, kind, level)
            rows.append({"perturbation": kind, "level": level, "model": MODEL_ORDER[0], "seed": 0,
                         "accuracy": (baseline.predict(stats(x)) == y_te).mean()})
            for name, models in nets.items():
                for seed, (model, mean, std) in zip(SEEDS, models):
                    pred = predict_logits(model, x, mean, std).argmax(1).numpy()
                    rows.append({"perturbation": kind, "level": level, "model": name, "seed": seed,
                                 "accuracy": (pred == y_te).mean()})
            print(f"{kind} {level}: done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "04_robustness.csv", index=False)
    return out


def plot(df):
    summary = df.groupby(["perturbation", "level", "model"])["accuracy"].agg(["mean", "min", "max"]).reset_index()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True)
    for ax, (kind, (title, levels, clean)) in zip(axes, PERTURBATIONS.items()):
        ax.axvline(clean, color=MUTED, lw=1, ls="--")
        for color, name in zip(SERIES, MODEL_ORDER):
            s = summary[(summary["perturbation"] == kind) & (summary["model"] == name)].sort_values("level")
            ax.fill_between(s["level"], s["min"], s["max"], color=color, alpha=0.15, lw=0)
            ax.plot(s["level"], s["mean"], color=color, marker="o", ms=5, mec=SURFACE, mew=1.5, label=name)
        ax.set_title(title, loc="left")
        ax.set_xticks(levels, [f"{v:g}" for v in levels], fontsize=8)
        ax.set_ylim(0, 1.03)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[0].set_ylabel("Test accuracy")
    axes[0].text(1.0, 0.04, " original", color=MUTED, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08), fontsize=9)
    fig.text(0.01, -0.04, "Line = mean over 3 seeds, band = min-max. Dashed line = undistorted test images. "
             "Augmentation ranges seen in training: exposure 0.6-1.4, contrast 0.7-1.3, noise sigma <= 10, blur sigma <= 1.",
             color=INK_2, fontsize=8.5)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    save(fig, FIG / "04_robustness.png")


def main():
    missing = [s for s in SEEDS if not (MODELS / f"cnn_photometric_seed{s}.pt").exists()]
    if missing:
        with ProcessPoolExecutor(max_workers=3) as pool:
            for seed in pool.map(train_augmented, missing):
                print(f"trained augmented CNN seed {seed}", flush=True)
    df = evaluate()
    plot(df)
    clean = df[((df["perturbation"] == "exposure") & (df["level"] == 1.0))]
    print("\nclean test accuracy:\n", clean.groupby("model")["accuracy"].agg(["mean", "std"]).round(4).to_string())
    worst = df.groupby(["model", "perturbation"])["accuracy"].min().unstack().round(3)
    print("\nworst accuracy per perturbation family (any level, any seed):\n", worst.to_string())


if __name__ == "__main__":
    main()
