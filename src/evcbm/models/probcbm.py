# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchvision import models
from .base import ModelWrapper


class _ResNet50Encoder(nn.Module):
    """ResNet50 feature extractor that outputs a 2048-dim vector."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            backbone.avgpool,  # -> (B, 2048, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)  # (B, 2048)


class ProbCBM(ModelWrapper):
    """
    Probabilistic Concept Bottleneck Model (sequential training).
    - Encoder: ResNet50 -> 2048 features.
    - Concept head outputs two positive parameters per concept (alpha, beta) via softplus.
      The concept probability is the Beta mean: p = alpha / (alpha + beta).
    - Label head consumes the mean concept probabilities.

    Recommended training:
      * Phase A: supervise concepts with BCEWithLogits on logit(mean_prob) (AMP-safe).
      * Phase B: freeze encoder+concept_head; train label_head with CE on GT concepts;
                 validate end-to-end with predicted mean probabilities.

    forward(x, return_concepts=True) returns:
      (y_logits, c_prob, c_logits) where c_prob in [0,1], c_logits = logit(c_prob).
    """
    def __init__(self, num_concepts: int, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes

        self.encoder = _ResNet50Encoder(pretrained=pretrained)
        # Concept parameters: 2048 -> 2*D (alpha, beta per concept)
        self.concept_head = nn.Linear(2048, 2 * num_concepts)
        # Label head consumes mean probabilities
        self.label_head = nn.Linear(num_concepts, num_classes)

        # Utilities
        self.softplus = nn.Softplus()
        self._eps = 1e-6

        # Losses (exposed for engines)
        self._ce = nn.CrossEntropyLoss()
        self._bce_with_logits = nn.BCEWithLogitsLoss()

    # --------- concept parameterization & transforms ---------
    def _alpha_beta(self, z: torch.Tensor):
        """Map encoder features to (alpha, beta) with positivity via softplus."""
        raw = self.concept_head(z)                      # (B, 2D)
        alpha_raw, beta_raw = torch.chunk(raw, 2, dim=1)
        alpha = self.softplus(alpha_raw) + self._eps    # (B, D) strictly positive
        beta  = self.softplus(beta_raw)  + self._eps
        return alpha, beta

    def _mean_probs(self, alpha: torch.Tensor, beta: torch.Tensor):
        """Beta mean per concept: alpha / (alpha + beta)."""
        return alpha / (alpha + beta + self._eps)

    def _logit_from_prob(self, p: torch.Tensor):
        """logit(p) = log(p) - log(1-p) with clamping for numerical stability."""
        p = torch.clamp(p, self._eps, 1.0 - self._eps)
        return torch.log(p) - torch.log(1.0 - p)

    # --------- fast path for Phase A (no label head) ---------
    def concepts_only(self, x: torch.Tensor):
        """
        Run encoder + concept_head only.
        Returns (c_prob, c_logits) with shapes (B, D).
        """
        z = self.encoder(x)
        alpha, beta = self._alpha_beta(z)
        c_prob = self._mean_probs(alpha, beta)
        c_logits = self._logit_from_prob(c_prob)
        return c_prob, c_logits

    # --------- forward ---------
    def forward(self, x: torch.Tensor, return_concepts: bool = False):
        """
        Returns:
          - if return_concepts: (y_logits, c_prob, c_logits)
          - else: y_logits only.
        """
        z = self.encoder(x)
        alpha, beta = self._alpha_beta(z)
        c_prob = self._mean_probs(alpha, beta)                 # (B, D)
        c_logits = self._logit_from_prob(c_prob)               # (B, D)
        y_logits = self.label_head(c_prob)                     # (B, C)

        if return_concepts:
            return y_logits, c_prob, c_logits
        return y_logits

    # --------- losses exposed for training engines ---------
    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor):
        """Classification CE on labels (for Phase B or joint baselines)."""
        return self._ce(logits, targets)

    def concept_bce_with_logits(self, c_logits: torch.Tensor, c_gt: torch.Tensor):
        """Preferred concept loss: AMP-safe BCE on logits."""
        return self._bce_with_logits(c_logits, c_gt)
