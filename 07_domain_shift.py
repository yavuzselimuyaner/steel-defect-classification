"""Step 7 - closing the gap to X-SDD.

A. Train the Small CNN on NEU with photometric augmentation *plus random intensity
   inversion*, then test on NEU and on the three X-SDD classes that have a NEU counterpart.
B. Few-shot adaptation: show the network k labelled X-SDD images per matching class
   (k = 5, 10, 25), fine-tune briefly on NEU + those images, and test on the remaining
   X-SDD images. NEU test accuracy is tracked to see whether NEU is forgotten.
   The same is done with frozen ImageNet ResNet-18 features + logistic regression.

X-SDD support/query images are drawn with three different seeds; support images are
never used for testing.
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from defect.data import CLASSES, FIGURES, MODELS, RESULTS
from defect.explain import load_checkpoint
from defect.models import ResNet18Gray, SmallCNN
from defect.plots import INK_2, MUTED, SERIES, SURFACE, plt, save
from defect.train import fit, load_tensors, normalize, photometric, photometric_invert, predict_logits
from defect.xsdd import MAPPING, load_xsdd

LOGS = RESULTS / "logs"
SEEDS = (0, 1, 2)
SHOTS = (5, 10, 25)
TARGET = [CLASSES.index(c) for c in MAPPING.values()]


def logger(name):
    LOGS.mkdir(parents=True, exist_ok=True)
    f = open(LOGS / f"{name}.log", "w", encoding="utf-8")

    def log(line):
        f.write(line + "\n")
        f.flush()
    return log


def support_query(labels, k, seed):
    """k random X-SDD images per matching class for training; the other matching images for testing."""
    rng = np.random.default_rng(100 + seed)
    support = np.concatenate([rng.choice(np.flatnonzero(labels == c), k, replace=False) for c in TARGET])
    query = np.setdiff1d(np.flatnonzero(labels >= 0), support)
    return support, query


def xsdd_scores(pred, labels, idx):
    """Overall and per-class accuracy on the given X-SDD images (6-way prediction)."""
    out = {"xsdd_acc": (pred[idx] == labels[idx]).mean()}
    for c in TARGET:
        sel = idx[labels[idx] == c]
        out[f"xsdd_{CLASSES[c]}"] = (pred[sel] == c).mean()
    out["xsdd_balanced"] = np.mean([out[f"xsdd_{CLASSES[c]}"] for c in TARGET])
    return out


def train_invert(seed):
    torch.set_num_threads(4)
    data, mean, std = load_tensors()
    model, _ = fit(SmallCNN(), data["train"][:2], data["val"][:2], mean, std, epochs=40, patience=10,
                   augment=photometric_invert, seed=seed, log=logger(f"07_cnn_invert_seed{seed}"))
    torch.save({"state": model.state_dict(), "mean": mean, "std": std}, MODELS / f"cnn_invert_seed{seed}.pt")
    return seed


def adapt(k, seed, base):
    """Fine-tune a NEU model on NEU train + k X-SDD images per matching class."""
    torch.set_num_threads(4)
    data, _, _ = load_tensors()
    X, _, labels, _ = load_xsdd()
    support, query = support_query(labels, k, seed)
    model, _, mean, std = load_checkpoint(MODELS / f"{base}_seed{seed}.pt")
    # Repeat the few X-SDD images so they make up roughly a fifth of every epoch
    reps = max(1, round(0.25 * len(data["train"][0]) / len(support)))
    Xtr = torch.cat([data["train"][0], X[support].repeat(reps, 1, 1, 1)])
    ytr = torch.cat([data["train"][1], torch.from_numpy(labels[support]).repeat(reps)])
    augment = photometric_invert if base == "cnn_invert" else photometric
    model, _ = fit(model, (Xtr, ytr), None, mean, std, epochs=6, lr=3e-4, augment=augment, seed=seed,
                   log=logger(f"07_adapt_{base}_{k}shot_seed{seed}"))
    pred_x = predict_logits(model, X, mean, std).argmax(1).numpy()
    pred_n = predict_logits(model, data["test"][0], mean, std).argmax(1).numpy()
    return {"method": f"fine-tune {base}", "k": k, "seed": seed,
            "neu_test_acc": (pred_n == data["test"][1].numpy()).mean(), **xsdd_scores(pred_x, labels, query)}


def run(job):
    kind, *args = job
    return train_invert(*args) if kind == "train" else adapt(*args)


def evaluate_unadapted(data, X, labels):
    """Every NEU-only model on NEU test and on all matching X-SDD images (k = 0)."""
    torch.set_num_threads(12)
    rows = []
    all_matching = np.flatnonzero(labels >= 0)
    for base in ("cnn", "cnn_photometric", "cnn_invert"):
        for seed in SEEDS:
            model, _, mean, std = load_checkpoint(MODELS / f"{base}_seed{seed}.pt")
            pred_x = predict_logits(model, X, mean, std).argmax(1).numpy()
            pred_n = predict_logits(model, data["test"][0], mean, std).argmax(1).numpy()
            # Same query sets as the adapted runs, so k = 0 is directly comparable
            for k in (0,) + SHOTS:
                idx = all_matching if k == 0 else support_query(labels, k, seed)[1]
                rows.append({"method": f"no adaptation {base}", "k": k, "seed": seed,
                             "neu_test_acc": (pred_n == data["test"][1].numpy()).mean(),
                             **xsdd_scores(pred_x, labels, idx)})
    return rows


def linear_probe(data, X, labels, mean, std):
    """Frozen ImageNet ResNet-18 features + logistic regression on NEU (+ k X-SDD images)."""
    torch.set_num_threads(12)
    net = ResNet18Gray().eval()
    with torch.no_grad():
        def feats(x):
            return torch.cat([net.features(normalize(x[i:i + 64], mean, std)) for i in range(0, len(x), 64)]).numpy()
        f_tr, f_te, f_x = feats(data["train"][0]), feats(data["test"][0]), feats(X)
    y_tr = data["train"][1].numpy()
    rows = []
    for k in (0,) + SHOTS:
        for seed in SEEDS:
            support, query = support_query(labels, k, seed) if k else (np.array([], int), np.flatnonzero(labels >= 0))
            reps = max(1, round(0.25 * len(f_tr) / max(1, len(support)))) if k else 0
            F = np.concatenate([f_tr] + [f_x[support]] * reps)
            Y = np.concatenate([y_tr] + [labels[support]] * reps)
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(F, Y)
            rows.append({"method": "linear probe (frozen ResNet-18)", "k": k, "seed": seed,
                         "neu_test_acc": (clf.predict(f_te) == data["test"][1].numpy()).mean(),
                         **xsdd_scores(clf.predict(f_x), labels, query)})
    return rows


def plot(df):
    methods = [("no adaptation cnn", "Small CNN, no adaptation", SERIES[0], "--"),
               ("fine-tune cnn_photometric", "Small CNN + photometric aug., fine-tuned", SERIES[1], "-"),
               ("fine-tune cnn_invert", "Small CNN + photometric + inversion aug., fine-tuned", SERIES[2], "-"),
               ("linear probe (frozen ResNet-18)", "Frozen ResNet-18 + logistic regression", SERIES[3], "-")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for ax, col, title in [(axes[0], "xsdd_balanced", "X-SDD accuracy (3 matching classes, held-out images)"),
                           (axes[1], "neu_test_acc", "NEU test accuracy (does NEU get forgotten?)")]:
        for key, label, color, ls in methods:
            s = df[df["method"] == key]
            if key.startswith("fine-tune"):
                base = key.split(" ", 1)[1]
                s = pd.concat([df[(df["method"] == f"no adaptation {base}") & (df["k"] == 0)], s])
            g = s.groupby("k")[col].agg(["mean", "min", "max"]).reset_index()
            ax.fill_between(g["k"], g["min"], g["max"], color=color, alpha=0.12, lw=0)
            ax.plot(g["k"], g["mean"], color=color, ls=ls, marker="o", ms=6, mec=SURFACE, mew=1.5, label=label)
        ax.set_xticks((0,) + SHOTS)
        ax.set_xlabel("Labelled X-SDD images per matching class used for adaptation")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(title, loc="left")
    axes[0].axhline(1 / 3, color=MUTED, lw=1, ls=":")
    axes[0].text(25, 1 / 3 + 0.01, "chance among the 3 classes", color=MUTED, fontsize=8, ha="right", va="bottom")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Mean per-class accuracy")
    axes[1].set_ylim(0.9, 1.005)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.1), fontsize=9)
    fig.text(0.01, -0.04, "Line = mean over 3 seeds (different X-SDD support images), band = min-max. "
             "Predictions are 6-way: confusing a matching X-SDD image with any NEU class counts as wrong.",
             color=INK_2, fontsize=8.5)
    fig.tight_layout()
    save(fig, FIGURES / "07_domain_adaptation.png")


def main():
    missing = [s for s in SEEDS if not (MODELS / f"cnn_invert_seed{s}.pt").exists()]
    jobs = [("train", s) for s in missing]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for s in pool.map(run, jobs):
            print(f"trained inversion-augmented CNN seed {s}", flush=True)

    data, mean, std = load_tensors()
    X, _, labels, _ = load_xsdd()
    rows = evaluate_unadapted(data, X, labels)
    rows += linear_probe(data, X, labels, mean, std)
    print("unadapted models and linear probe evaluated", flush=True)

    jobs = [("adapt", k, s, base) for base in ("cnn_invert", "cnn_photometric") for k in SHOTS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for job, row in zip(jobs, pool.map(run, jobs)):
            rows.append(row)
            print(f"done {job[1:]}: X-SDD {row['xsdd_balanced']:.1%}, NEU {row['neu_test_acc']:.1%}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "07_domain_shift.csv", index=False)
    cols = ["neu_test_acc", "xsdd_balanced"] + [f"xsdd_{CLASSES[c]}" for c in TARGET]
    print("\n", df.groupby(["method", "k"], sort=False)[cols].mean().round(3).to_string())
    plot(df)
    print("figure written")


if __name__ == "__main__":
    main()
