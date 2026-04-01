# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchvision import models
from .base import ModelWrapper

class _ResNet50Encoder(nn.Module):
    """ResNet50 as feature extractor: outputs 2048-dim features."""
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            backbone.avgpool,  # -> (B, 2048, 1, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)  # (B, 2048)

class CBMBase(ModelWrapper):
    """Common parts for CBM (encoder + concept head + label head)."""
    def __init__(self, num_concepts: int, num_classes: int, pretrained=True):
        super().__init__()
        self.encoder = _ResNet50Encoder(pretrained=pretrained)
        self.concept_head = nn.Linear(2048, num_concepts)   # concept logits
        self.label_head   = nn.Linear(num_concepts, num_classes)  # consumes concept probabilities
        
    def concepts_only(self, x: torch.Tensor):
        z = self.encoder(x)
        c_logits = self.concept_head(z)      # (B, D)
        c_prob = torch.sigmoid(c_logits)     # (B, D)
        return c_prob, c_logits

    def forward(self, x, return_concepts=False):
        z = self.encoder(x)
        c_logits = self.concept_head(z)             # (B, D) logits
        c = torch.sigmoid(c_logits)                 # (B, D) probabilities in [0,1]
        y_logits = self.label_head(c)               # (B, C) linear over probabilities
        if return_concepts:
            return y_logits, c, c_logits
        return y_logits

    def loss_fn(self, logits, targets):
        # classification CE (targets: LongTensor indices)
        return nn.CrossEntropyLoss()(logits, targets)

class CBMSequential(CBMBase):
    training_type = "cbm_sequential"

class CBMJoint(CBMBase):
    training_type = "cbm_joint"
