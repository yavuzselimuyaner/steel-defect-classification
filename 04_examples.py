"""Figure: one test image under every degradation used in step 4."""
import importlib

from defect.data import ROOT
from defect.plots import INK_2, plt, save
from defect.train import load_tensors

rob = importlib.import_module("04_robustness")
data, _, _ = load_tensors()
img = data["test"][0][[0]]
cols = max(len(v[1]) for v in rob.PERTURBATIONS.values())
fig, axes = plt.subplots(4, cols, figsize=(cols * 1.6, 4 * 1.8))
for r, (kind, (title, levels, _)) in enumerate(rob.PERTURBATIONS.items()):
    for c in range(cols):
        ax = axes[r, c]
        ax.axis("off")
        if c < len(levels):
            ax.imshow(rob.perturb(img, kind, levels[c])[0, 0], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{levels[c]:g}", fontsize=9, color=INK_2)
    axes[r, 0].text(-0.15, 0.5, title, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=9)
fig.suptitle(f"The four test-time degradations, applied to one test image ({data['test'][2]['file'][0]})", y=1.0)
fig.tight_layout()
save(fig, ROOT / "figures" / "04_degradation_examples.png")
