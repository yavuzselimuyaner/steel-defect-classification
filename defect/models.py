"""Network architectures. All take a (batch, 1, H, W) grayscale tensor and return 6 logits."""
import torchvision
from torch import nn

N_CLASSES = 6


class MLP(nn.Module):
    """Fully connected baseline on the flattened (downsampled) image: no notion of locality."""

    def __init__(self, size=64, hidden=(512, 128), dropout=0.3):
        super().__init__()
        layers, d = [nn.AdaptiveAvgPool2d(size), nn.Flatten()], size * size
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers, nn.Linear(d, N_CLASSES))

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """Four conv blocks (conv-BN-ReLU-maxpool), global average pooling, linear classifier."""

    def __init__(self, channels=(16, 32, 64, 128), dropout=0.3):
        super().__init__()
        layers, c_in = [], 1
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c),
                       nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(c_in, N_CLASSES))

    def forward(self, x):
        return self.head(self.features(x))


class ResNet18Gray(nn.Module):
    """ImageNet-pretrained ResNet-18 with a new 6-class head.

    The standardized grayscale image is repeated over the 3 input channels; standardized
    pixels are on roughly the same scale as ImageNet-normalized ones.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = torchvision.models.resnet18(weights=weights)
        self.net.fc = nn.Linear(self.net.fc.in_features, N_CLASSES)

    def features(self, x):
        """512-d globally pooled features, i.e. everything except the classifier."""
        n = self.net
        x = x.expand(-1, 3, -1, -1)
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        x = n.layer4(n.layer3(n.layer2(n.layer1(x))))
        return n.avgpool(x).flatten(1)

    def forward(self, x):
        return self.net.fc(self.features(x))
