# -*- coding: utf-8 -*-

import torch.nn as nn
import torch

class ModelWrapper(nn.Module):
    """Unified interface for seamless integration different models"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.CrossEntropyLoss()(logits, targets)

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=1)
