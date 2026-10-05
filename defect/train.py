"""Tensor loading, augmentation, the training loop and evaluation metrics."""
import copy
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torchvision.transforms.functional import gaussian_blur

from .data import DATA_DIR, load_gray, load_splits


def load_tensors():
    """All images as uint8 tensors per split, plus the train-set pixel mean/std."""
    splits = load_splits()
    out = {}
    for name, df in splits.groupby("split"):
        X = np.stack([load_gray(DATA_DIR / "IMAGES" / f) for f in df["file"]])
        out[name] = (torch.from_numpy(X).unsqueeze(1), torch.tensor(df["label"].to_numpy()), df.reset_index(drop=True))
    X_train = out["train"][0].float()
    return out, X_train.mean().item(), X_train.std().item()


def normalize(x_uint8, mean, std):
    return (x_uint8.float() - mean) / std


def random_flips(x):
    """Horizontal and vertical flips: a defect's class does not depend on its orientation."""
    h = torch.rand(len(x)) < 0.5
    v = torch.rand(len(x)) < 0.5
    x = x.clone()
    x[h] = x[h].flip(-1)
    x[v] = x[v].flip(-2)
    return x


@torch.no_grad()
def predict_logits(model, x, mean, std, batch_size=256):
    model.eval()
    return torch.cat([model(normalize(x[i:i + batch_size], mean, std)) for i in range(0, len(x), batch_size)])


def fit(model, train, val, mean, std, *, epochs=60, batch_size=32, lr=1e-3, weight_decay=1e-4,
        patience=12, augment=random_flips, seed=0, log=None):
    """Adam + cosine schedule; keeps the weights of the epoch with the lowest validation loss.

    `log` receives one progress line per epoch (with a time-left estimate), e.g. print.
    With `val=None` there is no early stopping and the final weights are kept.
    """
    torch.manual_seed(seed)
    X_tr, y_tr = train
    X_va, y_va = val if val is not None else (None, None)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)

    best, best_state, bad, history = np.inf, None, 0, []
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(X_tr), generator=g)
        total, correct = 0.0, 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb = X_tr[idx]
            if augment is not None:
                xb = augment(xb)
            logits = model(normalize(xb, mean, std))
            loss = loss_fn(logits, y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            correct += (logits.argmax(1) == y_tr[idx]).sum().item()
        sched.step()

        h = {"epoch": epoch + 1, "train_loss": total / len(X_tr), "train_acc": correct / len(X_tr)}
        if X_va is not None:
            val_logits = predict_logits(model, X_va, mean, std)
            h["val_loss"] = loss_fn(val_logits, y_va).item()
            h["val_acc"] = (val_logits.argmax(1) == y_va).float().mean().item()
            if h["val_loss"] < best - 1e-4:
                best, best_state, bad = h["val_loss"], copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
        h["seconds"] = time.time() - t0
        history.append(h)
        if log is not None:
            left = (epochs - epoch - 1) * np.mean([e["seconds"] for e in history])
            val_txt = f"val loss {h['val_loss']:.3f} acc {h['val_acc']:.1%}  " if X_va is not None else ""
            log(f"epoch {epoch + 1:2d}/{epochs}  train loss {h['train_loss']:.3f} acc {h['train_acc']:.1%}  "
                f"{val_txt}{h['seconds']:.0f}s/epoch  max ~{left / 60:.1f} min left")
        if X_va is not None and bad >= patience:
            if log is not None:
                log(f"early stopping: no val improvement for {patience} epochs")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def scores(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro")}


def confusion(y_true, y_pred, n=6):
    return confusion_matrix(y_true, y_pred, labels=list(range(n)))


def photometric(x):
    """Flips plus random exposure, contrast, sensor noise and slight blur.

    Simulates a camera whose lighting and focus drift; used to test whether the
    network can be made independent of global brightness.
    """
    x = random_flips(x).float()
    n = len(x)
    m = x.mean((2, 3), keepdim=True)
    contrast = torch.empty(n, 1, 1, 1).uniform_(0.7, 1.3)
    gain = torch.empty(n, 1, 1, 1).uniform_(0.6, 1.4)
    x = ((x - m) * contrast + m) * gain
    noisy = torch.rand(n) < 0.3
    x[noisy] += torch.randn_like(x[noisy]) * torch.empty(int(noisy.sum()), 1, 1, 1).uniform_(0, 10)
    if torch.rand(1).item() < 0.3:
        x = gaussian_blur(x, 5, torch.empty(1).uniform_(0.3, 1.0).item())
    return x.clamp(0, 255)


def photometric_invert(x):
    """Photometric augmentation plus a random intensity inversion (negative image).

    A defect stays the same defect whether it appears bright on dark or dark on bright
    steel; NEU only shows one polarity per class, other lines show the other one.
    """
    x = photometric(x)
    inv = torch.rand(len(x)) < 0.5
    x[inv] = 255 - x[inv]
    return x
