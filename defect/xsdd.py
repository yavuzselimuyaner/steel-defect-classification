"""The independent X-SDD dataset (Feng et al., 2021), used only for out-of-domain tests."""
import glob
import os

import numpy as np
import torch
from PIL import Image

from .data import CLASSES, ROOT

XSDD_DIR = ROOT / "external" / "X-SDD"
# X-SDD classes with a NEU counterpart; the remaining four do not exist in NEU
MAPPING = {"slag inclusion": "inclusion", "oxide scale of temperature system": "rolled-in_scale",
           "surface scratch": "scratches"}
UNSEEN = ["red iron", "iron sheet ash", "oxide scale of plate system", "finishing roll printing"]


def load_xsdd():
    """(images as uint8 (n, 1, 200, 200), X-SDD class names, NEU label or -1, file paths).

    X-SDD images are 128x128 RGB; they are converted to grayscale and resized to the
    200x200 the networks were trained on.
    """
    files = sorted(glob.glob(str(XSDD_DIR / "*" / "*")))
    names = np.array([os.path.basename(os.path.dirname(f)) for f in files])
    X = torch.from_numpy(np.stack([np.asarray(Image.open(f).convert("L").resize((200, 200), Image.BILINEAR))
                                   for f in files])).unsqueeze(1)
    labels = np.array([CLASSES.index(MAPPING[c]) if c in MAPPING else -1 for c in names])
    return X, names, labels, files
