"""
VGG16-FCN8 segmentation model (single-channel logit -> sigmoid foreground
probability), plus the inference helper used by the postprocessing pipeline.
"""

import cv2
import torch
import torch.nn as nn
from torchvision import models

from config import DEVICE
from dataset import preprocess


class VGG16_FCN8(nn.Module):
    """VGG16-FCN8 architecture using skip connections from pool3 and pool4
    to preserve high-resolution spatial boundaries for higher Dice scores."""

    def __init__(self, pretrained=True):
        super().__init__()
        try:
            weights = models.VGG16_Weights.DEFAULT if pretrained else None
            vgg = models.vgg16(weights=weights)
        except Exception:
            vgg = models.vgg16(weights=None)

        features = list(vgg.features.children())

        # Split VGG16 backbone into stages to capture skip connections
        self.stage1 = nn.Sequential(*features[:17])  # outputs pool3 (256 ch)
        self.stage2 = nn.Sequential(*features[17:24])  # outputs pool4 (512 ch)
        self.stage3 = nn.Sequential(*features[24:])  # outputs pool5 (512 ch)

        # 1x1 convolutions to project feature channels down to 1 logit class
        self.score_pool3 = nn.Conv2d(256, 1, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, 1, kernel_size=1)
        self.score_pool5 = nn.Conv2d(512, 1, kernel_size=1)

        # Transposed convolutions for multi-stage upsampling
        self.upsample2x_pool5 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample2x_pool4 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample8x = nn.ConvTranspose2d(1, 1, kernel_size=16, stride=8, padding=4, bias=False)

    def forward(self, x):
        pool3 = self.stage1(x)
        pool4 = self.stage2(pool3)
        pool5 = self.stage3(pool4)

        # Up 2x from pool5 and combine with score pool4
        s5 = self.score_pool5(pool5)
        up5 = self.upsample2x_pool5(s5)
        s4 = self.score_pool4(pool4)
        up4_in = up5 + s4[:, :, : up5.size(2), : up5.size(3)]

        # Up 2x from pool4 combination and combine with score pool3
        up4 = self.upsample2x_pool4(up4_in)
        s3 = self.score_pool3(pool3)
        up3_in = up4 + s3[:, :, : up4.size(2), : up4.size(3)]

        # Final 8x upsampling to full resolution
        out = self.upsample8x(up3_in)
        return out[:, :, : x.size(2), : x.size(3)]


def load_model(checkpoint_path=None, pretrained=True):
    """Builds a VGG16_FCN8 and, if given, loads trained weights onto DEVICE."""
    model = VGG16_FCN8(pretrained=pretrained).to(DEVICE)
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    model.eval()
    return model


def extract_features(model, img_bgr):
    """Foreground (nucleus) probability map from the TRAINED model."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = preprocess(rgb).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit[0, 0]).cpu().numpy()
        prob = cv2.resize(prob, (img_bgr.shape[1], img_bgr.shape[0]))
    return prob
