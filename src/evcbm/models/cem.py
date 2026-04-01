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


class CEM(ModelWrapper):
    """
    Concept Embedding Model (joint training).
    - Encoder: ResNet50 -> 2048-dim feature.
    - For each concept i: a context generator maps 2048 -> 2*emb_size (optionally with activation).
      The 2*emb_size vector is split into positive/negative embeddings (emb_size each).
    - Probability generator(s): Linear(2*emb_size -> 1), either shared across concepts or per-concept.
      The concept probability p_i = sigmoid(logit_i).
    - Bottleneck: mix positive/negative embeddings by probability: p * pos + (1 - p) * neg.
      The mixed embeddings for all concepts are flattened and passed to a c2y MLP for classification.
    """
    def __init__(self,
                 num_concepts: int,
                 num_classes: int,
                 pretrained: bool = True,
                 emb_size: int = 16,
                 embedding_activation: str = "leakyrelu",  # None | "sigmoid" | "relu" | "leakyrelu"
                 shared_prob_gen: bool = True,
                 c2y_layers=None,
                 training_intervention_prob: float = 0.25):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.emb_size = emb_size
        self.shared_prob_gen = shared_prob_gen
        self.training_intervention_prob = float(training_intervention_prob)

        # Encoder
        self.encoder = _ResNet50Encoder(pretrained=pretrained)

        # Per-concept context generators: 2048 -> 2*emb_size (+ optional activation)
        if embedding_activation is None:
            act = []
        elif embedding_activation.lower() == "sigmoid":
            act = [nn.Sigmoid()]
        elif embedding_activation.lower() == "relu":
            act = [nn.ReLU()]
        elif embedding_activation.lower() == "leakyrelu":
            act = [nn.LeakyReLU()]
        else:
            raise ValueError(f"Unsupported embedding activation: {embedding_activation}")

        self.context_gens = nn.ModuleList([
            nn.Sequential(nn.Linear(2048, 2 * emb_size), *act)
            for _ in range(num_concepts)
        ])

        # Probability generator(s): (2*emb_size) -> 1
        if shared_prob_gen:
            self.prob_gens = nn.ModuleList([nn.Linear(2 * emb_size, 1)])
        else:
            self.prob_gens = nn.ModuleList([nn.Linear(2 * emb_size, 1) for _ in range(num_concepts)])

        self.sigmoid = nn.Sigmoid()  # for concept probabilities

        # c2y head: flatten(concepts × emb_size) -> classes
        in_dim = num_concepts * emb_size
        units = [in_dim] + (c2y_layers or []) + [num_classes]
        layers = []
        for i in range(1, len(units)):
            layers.append(nn.Linear(units[i - 1], units[i]))
            if i != len(units) - 1:
                layers.append(nn.LeakyReLU())
        self.c2y = nn.Sequential(*layers)

        # Loss functions: classification CE (targets are class indices) +
        # concept BCEWithLogits (applied to concept logits)
        self._ce = nn.CrossEntropyLoss()
        self._bce_logits = nn.BCEWithLogitsLoss()

    def _gen_context_and_probs(self, x: torch.Tensor):
        """
        Returns:
          probs:   (B, D) concept probabilities in [0, 1]
          logits:  (B, D) raw concept logits
          pos_emb: (B, D, emb) positive concept embeddings
          neg_emb: (B, D, emb) negative concept embeddings
        """
        z = self.encoder(x)  # (B, 2048)
        contexts = []
        logits_list = []

        for i in range(self.num_concepts):
            ctx = self.context_gens[i](z)  # (B, 2*emb)
            logit_i = (self.prob_gens[0 if self.shared_prob_gen else i])(ctx)  # (B, 1)

            pos_i = ctx[:, :self.emb_size]        # (B, emb)
            neg_i = ctx[:, self.emb_size:]        # (B, emb)
            contexts.append(torch.stack([pos_i, neg_i], dim=1))  # (B, 2, emb)
            logits_list.append(logit_i)                            # (B, 1)

        # Stack over the concept dimension -> (B, D, 2, emb)
        contexts = torch.stack(contexts, dim=1)

        # Concatenate logits along concept dimension -> (B, D)
        logits = torch.cat(logits_list, dim=1)
        probs = torch.sigmoid(logits)  # (B, D)

        # Split contexts into positive/negative embeddings: (B, D, emb)
        pos_emb = contexts[:, :, 0, :]
        neg_emb = contexts[:, :, 1, :]
        return probs, logits, pos_emb, neg_emb

    def forward(
        self,
        x: torch.Tensor,
        c_true: torch.Tensor = None,
        return_concepts: bool = False,
        c_override: torch.Tensor = None
    ):
        """Forward pass.

        Args:
            x: Input images, shape (B, C, H, W).
            c_true: Optional ground-truth concepts (B, D) used for train-time
                interventions (RandInt). If None or if the module is in eval
                mode, no intervention is applied.
            return_concepts: If True, also return (probs, logits).
        """
        probs, logits, pos_emb, neg_emb = self._gen_context_and_probs(x)
        # probs/logits: (B, D)

        # ----- Train-time intervention (RandInt) -----
        probs_for_bottleneck = probs
        if (
            self.training
            and (self.training_intervention_prob is not None)
            and (self.training_intervention_prob > 0.0)
            and (c_true is not None)
        ):
            c_true = c_true.to(probs.device).float()
            with torch.no_grad():
                mask = (torch.rand_like(probs) < self.training_intervention_prob).float()
            # mask == 1.0 -> use ground-truth concept; 0.0 -> keep prediction
            probs_for_bottleneck = probs * (1.0 - mask) + c_true * mask
        # ---------------------------------------------
        
        if c_override is not None:
            c_override = c_override.to(probs_for_bottleneck.device)
            probs_for_bottleneck = c_override

        # Mix embeddings by (possibly intervened) probabilities -> (B, D, emb)
        bottleneck = pos_emb * probs_for_bottleneck.unsqueeze(-1) + \
                     neg_emb * (1 - probs_for_bottleneck).unsqueeze(-1)
        # Flatten and classify
        y_logits = self.c2y(bottleneck.view(bottleneck.size(0), -1))
        if return_concepts:
            # Keep API aligned with CBM: (y_logits, c_prob, c_logits)
            # Note: 返回原始 probs/logits（未干预），用于度量概念性能。
            return y_logits, probs, logits
        return y_logits

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor):
        """Classification loss only (concept loss is added in the training loop)."""
        return self._ce(logits, targets)

    # Helper for engines to compute concept BCEWithLogits
    def concept_bce_logits(self, c_logits: torch.Tensor, c_gt: torch.Tensor):
        return self._bce_logits(c_logits, c_gt)
