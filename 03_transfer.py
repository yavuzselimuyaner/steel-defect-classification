"""Step 3 - transfer learning and the small-data question.

A. Linear probe: frozen ImageNet ResNet-18 features + logistic regression, for 5..all images per class.
B. Fine-tuning: the whole pretrained ResNet-18 on the full training set (3 seeds).
C. Small CNN from scratch on 5/10/25/50 images per class (3 random subsets each).

The small-data runs (A and C) never touch the validation set: with only a handful of
labelled images you would not have 60 spare ones per class for early stopping either.
C trains for a fixed budget of gradient steps and keeps the final weights.
"""
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from defect.data import MODELS, RESULTS
from defect.models import ResNet18Gray, SmallCNN
from defect.train import fit, load_tensors, normalize, predict_logits, scores

LOGS = RESULTS / "logs"
SHOTS = (5, 10, 25, 50)
SEEDS = (0, 1, 2)
FEWSHOT_STEPS = 300


def subset(y, n, seed):
    """Indices of n random training images per class."""
    rng = np.random.default_rng(seed)
    y = y.numpy()
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), n, replace=False) for c in range(6)]))


def logger(name):
    LOGS.mkdir(parents=True, exist_ok=True)
    f = open(LOGS / f"{name}.log", "w", encoding="utf-8")

    def log(line):
        f.write(line + "\n")
        f.flush()
    return log


def linear_probe(data, mean, std):
    torch.set_num_threads(12)
    model = ResNet18Gray().eval()
    feats = {}
    with torch.no_grad():
        for split in ("train", "test"):
            x = data[split][0]
            feats[split] = torch.cat([model.features(normalize(x[i:i + 64], mean, std))
                                      for i in range(0, len(x), 64)]).numpy()
    rows = []
    y_tr, y_te = data["train"][1], data["test"][1].numpy()
    for n in SHOTS + (None,):
        for seed in (SEEDS if n else (0,)):
            idx = subset(y_tr, n, seed) if n else np.arange(len(y_tr))
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
            clf.fit(feats["train"][idx], y_tr.numpy()[idx])
            rows.append({"model": "ResNet-18 linear probe (frozen)", "per_class": n or "all", "seed": seed,
                         **scores(y_te, clf.predict(feats["test"]))})
    return rows


def finetune(seed):
    torch.set_num_threads(4)
    data, mean, std = load_tensors()
    model, history = fit(ResNet18Gray(), data["train"][:2], data["val"][:2], mean, std,
                         epochs=10, patience=4, lr=3e-4, seed=seed, log=logger(f"03_resnet18_seed{seed}"))
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "mean": mean, "std": std}, MODELS / f"resnet18_seed{seed}.pt")
    pred = predict_logits(model, data["test"][0], mean, std).argmax(1).numpy()
    return {"model": "ResNet-18 fine-tuned", "per_class": "all", "seed": seed, "epochs": len(history),
            **scores(data["test"][1].numpy(), pred)}


def fewshot_cnn(n, seed):
    torch.set_num_threads(4)
    data, mean, std = load_tensors()
    idx = torch.from_numpy(subset(data["train"][1], n, seed))
    steps_per_epoch = math.ceil(len(idx) / 32)
    epochs = math.ceil(FEWSHOT_STEPS / steps_per_epoch)
    model, _ = fit(SmallCNN(), (data["train"][0][idx], data["train"][1][idx]), None, mean, std,
                   epochs=epochs, seed=seed, log=logger(f"03_cnn_{n}shot_seed{seed}"))
    pred = predict_logits(model, data["test"][0], mean, std).argmax(1).numpy()
    return {"model": "Small CNN (from scratch)", "per_class": n, "seed": seed, "epochs": epochs,
            **scores(data["test"][1].numpy(), pred)}


def run(job):
    kind, *args = job
    return finetune(*args) if kind == "finetune" else fewshot_cnn(*args)


def main():
    data, mean, std = load_tensors()
    rows = linear_probe(data, mean, std)
    print(pd.DataFrame(rows).groupby("per_class", sort=False)["accuracy"].agg(["mean", "std"]).round(4), flush=True)

    jobs = [("finetune", s) for s in SEEDS] + [("fewshot", n, s) for n in SHOTS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for job, row in zip(jobs, pool.map(run, jobs)):
            rows.append(row)
            print(f"done {job}: test accuracy {row['accuracy']:.1%}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "03_runs.csv", index=False)
    print("\nTEST ACCURACY (mean / std over seeds)")
    print(out.groupby(["model", "per_class"], sort=False)["accuracy"].agg(["mean", "std", "count"]).round(4).to_string())


if __name__ == "__main__":
    main()
