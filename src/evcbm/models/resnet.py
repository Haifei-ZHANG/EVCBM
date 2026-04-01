# -*- coding: utf-8 -*-

import torch.nn as nn
from torchvision import models
from .base import ModelWrapper

class BaselineResNet50(ModelWrapper):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        return self.backbone(x)
