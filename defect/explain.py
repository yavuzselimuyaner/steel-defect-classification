"""Model loading, Grad-CAM and the camera degradations shared by the scripts and the demo."""
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from torchvision.transforms.functional import gaussian_blur

from .models import ResNet18Gray, SmallCNN

# Single-hue overlay: transparent where the map is cold, solid orange where it is hot
HEAT = LinearSegmentedColormap.from_list("heat", [(0.92, 0.41, 0.20, 0.0), (0.92, 0.41, 0.20, 0.85)])

ARCHITECTURES = {
    # file prefix: (constructor, Grad-CAM target layer)
    "cnn": (SmallCNN, lambda m: m.features[-2]),
    "cnn_photometric": (SmallCNN, lambda m: m.features[-2]),
    "cnn_invert": (SmallCNN, lambda m: m.features[-2]),
    "resnet18": (lambda: ResNet18Gray(pretrained=False), lambda m: m.net.layer4),
}


def load_checkpoint(path):
    """Returns (model in eval mode, Grad-CAM layer, pixel mean, pixel std)."""
    prefix = path.stem.rsplit("_seed", 1)[0]
    ctor, layer_of = ARCHITECTURES[prefix]
    ckpt = torch.load(path)
    model = ctor()
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, layer_of(model), ckpt["mean"], ckpt["std"]


def grad_cam(model, layer, x, target=None):
    """Grad-CAM maps (n, H, W) scaled to [0, 1] and class probabilities (n, 6).

    The map is for the predicted class unless `target` gives class indices.
    """
    acts, grads = {}, {}

    def keep(module, inputs, out):
        acts["a"] = out
        out.register_hook(lambda g: grads.__setitem__("g", g))

    hook = layer.register_forward_hook(keep)
    model.zero_grad()
    logits = model(x)
    cls = logits.argmax(1) if target is None else target
    logits.gather(1, cls[:, None]).sum().backward()
    hook.remove()
    a, g = acts["a"], grads["g"]
    cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam - cam.amin((1, 2), keepdim=True)
    cam = cam / cam.amax((1, 2), keepdim=True).clamp_min(1e-8)
    return cam.detach().numpy(), logits.softmax(1).detach().numpy()


def perturb(x, kind, level):
    """Apply one camera degradation to uint8 images; returns float images clipped to 0-255."""
    x = x.float()
    if kind == "exposure":
        x = x * level
    elif kind == "contrast":
        m = x.mean((2, 3), keepdim=True)
        x = (x - m) * level + m
    elif kind == "blur" and level > 0:
        x = gaussian_blur(x, 2 * int(np.ceil(3 * level)) + 1, level)
    elif kind == "noise" and level > 0:
        x = x + torch.randn(x.shape, generator=torch.Generator().manual_seed(0)) * level
    return x.clamp(0, 255)
