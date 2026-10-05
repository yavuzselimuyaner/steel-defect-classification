"""Step 2 - baselines: intensity statistics, MLP and a small CNN on the same fixed split.

Neural models are trained with three seeds; we report mean +- std on the test set.
"""
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from defect.data import MODELS, RESULTS
from defect.models import MLP, SmallCNN
from defect.train import confusion, fit, load_tensors, predict_logits, scores

SEEDS = (0, 1, 2)
ARCHS = {"mlp": MLP, "cnn": SmallCNN}


def stat_features(x, kind):
    x = x.squeeze(1).float().numpy().reshape(len(x), -1)
    if kind == "mean_std":
        return np.stack([x.mean(1), x.std(1)], 1)
    return np.stack([np.histogram(r, bins=32, range=(0, 255), density=True)[0] for r in x])


def run_stat_baselines(data):
    rows = []
    for kind, label in [("mean_std", "Brightness + contrast (2 features)"),
                        ("hist", "Intensity histogram (32 bins)")]:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        clf.fit(stat_features(data["train"][0], kind), data["train"][1].numpy())
        for split in ("val", "test"):
            pred = clf.predict(stat_features(data[split][0], kind))
            rows.append({"model": label, "seed": 0, "split": split, **scores(data[split][1].numpy(), pred)})
    return rows


def train_one(arch, seed):
    torch.set_num_threads(4)
    data, mean, std = load_tensors()
    (RESULTS / "logs").mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "logs" / f"02_{arch}_seed{seed}.log", "w", encoding="utf-8") as f:
        def log(line):
            f.write(line + "\n")
            f.flush()
        model, history = fit(ARCHS[arch](), data["train"][:2], data["val"][:2], mean, std,
                             epochs=40, patience=10, seed=seed, log=log)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "mean": mean, "std": std}, MODELS / f"{arch}_seed{seed}.pt")
    out = {"history": history, "pred": {}}
    for split in ("val", "test"):
        logits = predict_logits(model, data[split][0], mean, std)
        out["pred"][split] = logits.softmax(1).numpy().tolist()
    return arch, seed, out


def main():
    data, _, _ = load_tensors()
    rows = run_stat_baselines(data)
    histories, probs = {}, {}

    jobs = [(a, s) for a in ARCHS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for arch, seed, out in pool.map(train_one, *zip(*jobs)):
            histories[(arch, seed)] = out["history"]
            for split in ("val", "test"):
                p = np.array(out["pred"][split])
                probs[(arch, seed, split)] = p
                rows.append({"model": {"mlp": "MLP (64x64 pixels)", "cnn": "Small CNN"}[arch], "seed": seed,
                             "split": split, "epochs": len(out["history"]),
                             **scores(data[split][1].numpy(), p.argmax(1))})
            print(f"done {arch} seed {seed}", flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(RESULTS / "02_runs.csv", index=False)
    summary = (runs[runs["split"] == "test"].groupby("model", sort=False)[["accuracy", "macro_f1"]]
               .agg(["mean", "std", "count"]))
    summary.to_csv(RESULTS / "02_summary.csv")
    print("\nTEST SET (mean / std over seeds)\n", summary.round(4).to_string())

    # Keep everything later steps and plots need
    with open(RESULTS / "02_histories.json", "w") as f:
        json.dump({f"{a}_seed{s}": h for (a, s), h in histories.items()}, f)
    test_df = data["test"][2][["file", "cls", "label"]].copy()
    for (arch, seed, split), p in probs.items():
        if split == "test":
            test_df[f"{arch}_seed{seed}_pred"] = p.argmax(1)
            test_df[f"{arch}_seed{seed}_conf"] = p.max(1)
    test_df.to_csv(RESULTS / "02_test_predictions.csv", index=False)

    for arch in ARCHS:
        cms = [confusion(data["test"][1].numpy(), probs[(arch, s, "test")].argmax(1)) for s in SEEDS]
        np.save(RESULTS / f"02_confusion_{arch}.npy", np.sum(cms, axis=0))


if __name__ == "__main__":
    main()
