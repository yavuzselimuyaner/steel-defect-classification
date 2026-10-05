"""Interactive demo: steel surface defect classification with Grad-CAM.

    streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import torch
from matplotlib.patches import Rectangle
from PIL import Image

from defect.data import CLASS_NAMES, CLASSES, DATA_DIR, ROOT, load_gray, read_boxes
from defect.explain import HEAT, grad_cam, load_checkpoint, perturb
from defect.plots import BLUE, GRID, INK_2, SERIES, plt
from defect.train import normalize

st.set_page_config(page_title="Steel Defect Inspector", layout="wide")
torch.set_num_threads(4)

MODEL_FILES = {
    "Small CNN": "cnn_seed0.pt",
    "Small CNN + photometric augmentation": "cnn_photometric_seed0.pt",
    "ResNet-18 (ImageNet, fine-tuned)": "resnet18_seed0.pt",
}
LOW_CONFIDENCE = 0.6


def models_dir():
    """Prefer the leakage-free block split; fall back to the random split."""
    for split in ("block", "random"):
        d = ROOT / "models" / split
        if any((d / f).exists() for f in MODEL_FILES.values()):
            return d, split
    return None, None


@st.cache_resource
def get_model(path):
    return load_checkpoint(path)


@st.cache_data
def test_images(split):
    s = pd.read_csv(ROOT / "results" / split / "splits.csv")
    return s[s["split"] == "test"][["file", "cls"]].reset_index(drop=True)


st.title("Steel surface defect inspector")
st.caption("NEU-DET hot-rolled steel strip defects · COM4573 Artificial Neural Networks project")

mdir, split = models_dir()
if mdir is None:
    st.error("No trained models found. Run `python run_all.py` first.")
    st.stop()
available = {k: mdir / v for k, v in MODEL_FILES.items() if (mdir / v).exists()}

with st.sidebar:
    st.header("Model")
    model_name = st.selectbox("Network", list(available))
    st.caption(f"Weights from the **{split}** split (seed 0).")
    st.header("Image")
    source = st.radio("Source", ["Test-set image", "Upload your own"])
    st.header("Simulate the camera")
    st.caption("Degrade the image the way a real inspection camera might.")
    exposure = st.slider("Exposure (brightness gain)", 0.5, 1.5, 1.0, 0.05)
    contrast = st.slider("Contrast factor", 0.4, 1.5, 1.0, 0.05)
    blur = st.slider("Blur sigma (px)", 0.0, 3.0, 0.0, 0.25)
    noise = st.slider("Sensor noise sigma (gray levels)", 0, 30, 0, 1)

boxes, true_cls = [], None
if source == "Test-set image":
    tests = test_images(split)
    cls_filter = st.selectbox("Class", ["any"] + [CLASS_NAMES[c] for c in CLASSES])
    options = tests if cls_filter == "any" else tests[tests["cls"].map(CLASS_NAMES) == cls_filter]
    file = st.selectbox("Test image (never seen during training)", options["file"])
    img = load_gray(DATA_DIR / "IMAGES" / file)
    boxes = read_boxes(DATA_DIR / "ANNOTATIONS" / file.replace(".jpg", ".xml"))
    true_cls = file.rsplit("_", 1)[0]
else:
    up = st.file_uploader("Grayscale steel surface image (any size; resized to 200x200)",
                          type=["jpg", "jpeg", "png", "bmp"])
    if up is None:
        st.info("Upload an image to classify it.")
        st.stop()
    img = np.asarray(Image.open(up).convert("L").resize((200, 200)))

x = torch.from_numpy(np.ascontiguousarray(img))[None, None]
for kind, level, neutral in [("exposure", exposure, 1.0), ("contrast", contrast, 1.0),
                             ("blur", blur, 0.0), ("noise", noise, 0)]:
    if level != neutral:
        x = perturb(x, kind, level)
x = x.float()

model, layer, mean, std = get_model(available[model_name])
cam, probs = grad_cam(model, layer, normalize(x, mean, std))
cam, probs = cam[0], probs[0]
pred = CLASSES[probs.argmax()]

left, right = st.columns([1.1, 1])
with left:
    show_cam = st.toggle("Show Grad-CAM (where the network looks)", value=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.imshow(x[0, 0].numpy(), cmap="gray", vmin=0, vmax=255)
    if show_cam:
        ax.imshow(cam, cmap=HEAT, vmin=0, vmax=1)
    for x1, y1, x2, y2 in boxes:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=SERIES[0], lw=1.5))
    ax.axis("off")
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    if boxes:
        st.caption("Blue boxes: annotated defect regions. Orange: Grad-CAM for the predicted class.")

with right:
    st.subheader("Prediction")
    c1, c2 = st.columns(2)
    c1.metric("Predicted defect", CLASS_NAMES[pred])
    c2.metric("Confidence", f"{probs.max():.1%}")
    if true_cls is not None:
        if pred == true_cls:
            st.success(f"Correct: the true class is {CLASS_NAMES[true_cls]}.")
        else:
            st.error(f"Wrong: the true class is {CLASS_NAMES[true_cls]}.")
    if probs.max() < LOW_CONFIDENCE:
        st.warning(f"Confidence below {LOW_CONFIDENCE:.0%}: route this part to manual inspection.")

    order = np.argsort(probs)
    fig, ax = plt.subplots(figsize=(5, 2.8))
    colors = [BLUE if CLASSES[i] == pred else "#c3c2b7" for i in order]
    ax.barh([CLASS_NAMES[CLASSES[i]] for i in order], probs[order], color=colors, height=0.6)
    for k, i in enumerate(order):
        ax.text(probs[i] + 0.01, k, f"{probs[i]:.1%}", va="center", fontsize=9, color=INK_2)
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}" if v <= 1 else ""))
    ax.grid(axis="y", visible=False)
    ax.set_title("Class probabilities", loc="left")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with st.expander("About this demo"):
    st.markdown(
        "Six classes of hot-rolled steel strip defects from the NEU surface defect database. "
        "The test-set images come from blocks of consecutive frames that were never used for training. "
        "Grad-CAM highlights the image regions that raised the predicted class score. "
        "The camera sliders reproduce the robustness experiment of step 4: compare the plain Small CNN "
        "with the photometrically augmented one under strong exposure changes.")
