# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchvision import models
from .base import ModelWrapper


# ---------------- Backbone ----------------
class _ResNet50Encoder(nn.Module):
    """ResNet50 feature extractor → 2048-dim vector."""
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


# ---------------- Optional context head δ(x) ----------------
class _ContextHead(nn.Module):
    """Contextual class-level discount δ(x) ∈ [0,1]^C"""
    def __init__(self, D: int, C: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(D, hidden), nn.ReLU(inplace=True)]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        out = nn.Linear(hidden, C)
        layers.append(out)
        self.net = nn.Sequential(*layers)

        # init: small weights so that initial δ(x) ≈ sigmoid(-4) ≈ 0.018
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        if dropout and dropout > 0:
            pass
        nn.init.xavier_uniform_(out.weight, gain=0.1)
        nn.init.constant_(out.bias, -4.0)  # sigmoid(-4)≈0.018 → almost no discount initially

    def forward(self, s_prob: torch.Tensor) -> torch.Tensor:
        # s_prob: (B, D) in [0,1]
        return torch.sigmoid(self.net(s_prob))  # (B, C)


# ---------------- EVCBM ----------------
class EVCBM(ModelWrapper):
    """
    Evidential CBM (DS/Yager fusion over concept → class masses).

    Pipeline:
      image x → encoder (2048) → concept_head (D logits) → sigmoid → s_prob (B,D)
      s_prob → evidential fusion → belief bel (B,C), m_theta (B,1)
      betp = bel + m_theta / C → log_betp = log(clamp(betp))
      return:
        - forward(x):                log_betp (B,C)
        - forward(x, return_concepts=True): (log_betp, c_prob, c_logits)
    """
    ETA_MIN_THETA = 0.1         # slightly looser guard than 1e-2
    ZETA_HIGH_CONFLICT = 1e-3   # 冲突很高时走 Yager

    def __init__(self,
                 num_concepts: int,
                 num_classes: int,
                 pretrained: bool = True,
                 topk: int = None,
                 lambda_yager_blend: float = 0.0,
                 eps: float = 1e-6,
                 use_adjust_discount: bool = True,
                 discount_hidden: int = 128,
                 discount_dropout: float = 0.0,
                 use_context: bool = True,
                 context_hidden: int = 128,
                 context_dropout: float = 0.0,
                 denom_min: float = 1e-3,
                 detach_denom: bool = True,
                 betp_clip: float = 1e-6):
        super().__init__()
        self.D = int(num_concepts)
        self.C = int(num_classes)
        self.topk = None if topk is None else int(topk)
        self.lam = float(lambda_yager_blend)
        self._eps = float(eps)
        self.use_context = bool(use_context)

        # DST / log 稳定参数
        self._denom_min = float(denom_min)
        self.detach_denom = bool(detach_denom)
        self._betp_clip = float(betp_clip)

        # backbone
        self.encoder = _ResNet50Encoder(pretrained=pretrained)
        self.concept_head = nn.Linear(2048, self.D)  # concept logits

        # base mass over classes per concept: Parameter[D,C] → softmax along C
        self.base_logits = nn.Parameter(torch.empty(self.D, self.C))
        nn.init.normal_(self.base_logits, mean=0.0, std=0.02)  # break symmetry
        
        self.use_adjust_discount = bool(use_adjust_discount)
        
        # concept discount Δ(s): MLP D→D, zero init last for small Δ initially
        disc_layers = [nn.Linear(self.D, discount_hidden), nn.ReLU(inplace=True)]
        if discount_dropout and discount_dropout > 0:
            disc_layers.append(nn.Dropout(discount_dropout))
        disc_layers.append(nn.Linear(discount_hidden, self.D))
        self.discount_net = nn.Sequential(*disc_layers)
        nn.init.zeros_(self.discount_net[-1].weight)
        nn.init.zeros_(self.discount_net[-1].bias)
        
        if not self.use_adjust_discount:
            for p in self.discount_net.parameters():
                p.requires_grad = False
        
        # optional context δ(x) over classes
        if self.use_context:
            self.ctx = _ContextHead(self.D, self.C, hidden=context_hidden, dropout=context_dropout)
        else:
            self.ctx = None

        # losses
        self._bce_logits = nn.BCEWithLogitsLoss()
        self._nll = nn.NLLLoss()

    # ---------- utils ----------
    def _sanitize(self, x: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
        # be gentle: replace NaN with mid, clamp to [lo, hi]
        x = torch.nan_to_num(x, nan=(lo + hi) * 0.5, posinf=hi, neginf=lo)
        return torch.clamp(x, lo, hi)

    def _clamp_prob(self, p: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.nan_to_num(p, nan=0.5), self._eps, 1.0 - self._eps)

    def base_mass_singleton(self) -> torch.Tensor:
        # (D,C)
        return torch.softmax(self.base_logits, dim=-1)

    def sparsity_regularizer(self) -> torch.Tensor:
        """
        对 base mass 的 Shannon 熵做惩罚：
        - p = softmax(base_logits) 形状 (D,C)
        - 返回所有概念熵的平均值
        训练时可以在总 loss 中加：lambda_entropy * model.sparsity_regularizer()
        """
        p = self.base_mass_singleton()  # (D,C)
        p = torch.clamp(p, min=self._eps)
        return -(p * torch.log(p)).sum(dim=-1).mean()

    # ---------- concept discount s -> s' ----------
    def _adjust_discount(self, s_prob: torch.Tensor) -> torch.Tensor:
        """
        s_prob: (B,D) in [0,1]
        s' = sigmoid(logit(s) + Δ(s))
        """
        s = self._clamp_prob(s_prob)
        logit_s = torch.log(s) - torch.log1p(-s)
        delta = self.discount_net(s_prob)  # (B,D)
        s_prime = torch.sigmoid(logit_s + delta)
        # keep within [0, 1-ETA_MIN_THETA]
        return torch.clamp(s_prime, 0.0, 1.0 - self.ETA_MIN_THETA)

    # ---------- combine two sources ----------
    def _combine_pair(self,
                      m1_single: torch.Tensor, m1_theta: torch.Tensor,
                      m2_single: torch.Tensor, m2_theta: torch.Tensor):
        """
        m*_single: (B,C), m*_theta: (B,1)
        returns (m12_single, m12_theta)
        """
        m1_single = self._sanitize(m1_single)
        m2_single = self._sanitize(m2_single)
        m1_theta  = self._sanitize(m1_theta)
        m2_theta  = self._sanitize(m2_theta)

        sum_m2_single = m2_single.sum(dim=-1, keepdim=True)  # (B,1)
        Kconf = (m1_single * (sum_m2_single - m2_single)).sum(dim=-1, keepdim=True)  # (B,1)
        Kconf = self._sanitize(Kconf)

        # ensure 1 - Kconf >= denom_min
        Kconf = torch.clamp(Kconf, 0.0, 1.0 - self._denom_min)
        one_minus_K = 1.0 - Kconf  # (B,1)

        # optionally detach denominator to reduce gradient sensitivity to conflict
        if self.detach_denom:
            one_minus_K_safe = one_minus_K.detach()
        else:
            one_minus_K_safe = one_minus_K
        one_minus_K_safe = torch.clamp(one_minus_K_safe, min=self._denom_min, max=1.0)

        # Dempster
        til_single = (m1_single * m2_theta) + (m1_theta * m2_single) + (m1_single * m2_single)  # (B,C)
        til_theta  = (m1_theta * m2_theta)  # (B,1)

        m12_single_d = til_single / one_minus_K_safe
        m12_theta_d  = til_theta  / one_minus_K_safe

        # Yager (fallback / blend)
        m12_single_y = til_single
        m12_theta_y  = til_theta + Kconf

        use_yager = (one_minus_K < self.ZETA_HIGH_CONFLICT)
        m12_single = torch.where(use_yager, m12_single_y, m12_single_d)
        m12_theta  = torch.where(use_yager, m12_theta_y,  m12_theta_d)

        if self.lam > 0.0:
            m12_single = (1 - self.lam) * m12_single + self.lam * m12_single_y
            m12_theta  = (1 - self.lam) * m12_theta  + self.lam * m12_theta_y

        # normalize
        m12_single = self._sanitize(m12_single)
        m12_theta  = self._sanitize(m12_theta)
        ssum = torch.clamp(m12_single.sum(dim=-1, keepdim=True) + m12_theta,
                           min=self._denom_min)
        m12_single = m12_single / ssum
        m12_theta  = m12_theta  / ssum
        return m12_single, m12_theta

    def _combine_many(self, singles_list, thetas_list):
        """Fold all sources."""
        B, C = singles_list[0].shape
        device = singles_list[0].device
        dtype  = singles_list[0].dtype
        m_single = torch.zeros(B, C, device=device, dtype=dtype)
        m_theta  = torch.ones(B, 1, device=device, dtype=dtype)
        for s, t in zip(singles_list, thetas_list):
            m_single, m_theta = self._combine_pair(m_single, m_theta, s, t)
        return m_single, m_theta

    # ---------- fusion given concept probabilities ----------
    def forward_fusion_only(self, s_prob: torch.Tensor):
        """
        s_prob: (B,D) in [0,1]
        returns belief bel (B,C) and m_theta (B,1)
        """
        B, D = s_prob.shape
        
        # s_adj = self._adjust_discount(s_prob)            # (B,D)
        if self.use_adjust_discount:
            s_adj = self._adjust_discount(s_prob)  # (B,D)
        else:
            # Use s_prob directly (but sanitize + keep theta >= ETA_MIN_THETA)
            s_adj = self._sanitize(s_prob)
            s_adj = torch.clamp(s_adj, 0.0, 1.0 - self.ETA_MIN_THETA)
            
        
        base  = self.base_mass_singleton()               # (D,C)
        delta = self.ctx(s_prob) if self.ctx is not None else None  # (B,C) or None

        singles_list, thetas_list = [], []

        if self.topk is not None and self.topk > 0:
            k = min(self.topk, D)
            topv, topi = torch.topk(s_adj, k=k, dim=1)         # (B,k)
            base_sel = base[topi]                              # (B,k,C)
            md_single = topv.unsqueeze(-1) * base_sel          # (B,k,C)
            md_theta  = 1.0 - topv.unsqueeze(-1)               # (B,k,1)

            if delta is not None:
                md_single_d = md_single * (1.0 - delta.unsqueeze(1))   # (B,k,C)
                moved = (md_single - md_single_d).sum(dim=-1, keepdim=True)  # (B,k,1)
                md_theta   = torch.clamp(md_theta + moved, 0.0, 1.0)
                md_single  = self._sanitize(md_single_d)
            else:
                md_single  = self._sanitize(md_single)

            for j in range(k):
                singles_list.append(md_single[:, j, :])
                thetas_list.append(md_theta[:, j, :])

        else:
            # use all concepts
            for d in range(D):
                md_single = s_adj[:, d:d+1] * base[d].unsqueeze(0)     # (B,C)
                md_theta  = 1.0 - s_adj[:, d:d+1]                      # (B,1)
                if delta is not None:
                    md_single_d = md_single * (1.0 - delta)            # (B,C)
                    moved = (md_single - md_single_d).sum(dim=-1, keepdim=True)  # (B,1)
                    md_theta   = torch.clamp(md_theta + moved, 0.0, 1.0)
                    md_single  = self._sanitize(md_single_d)
                else:
                    md_single  = self._sanitize(md_single)
                singles_list.append(md_single)
                thetas_list.append(md_theta)

        bel, mtheta = self._combine_many(singles_list, thetas_list)
        bel   = self._sanitize(bel)
        mtheta = self._sanitize(mtheta)
        return bel, mtheta

    def _betp_from_bel(self, bel: torch.Tensor, mtheta: torch.Tensor) -> torch.Tensor:
        # betp = bel + mtheta / C, then renorm
        C = bel.shape[1]
        betp = bel + mtheta / float(C)
        betp = torch.clamp(betp, min=self._betp_clip, max=1.0)
        betp = betp / betp.sum(dim=1, keepdim=True)
        return betp

    # ---------- forward ----------
    def forward(self, x: torch.Tensor, return_concepts: bool = False):
        z = self.encoder(x)                       # (B,2048)
        c_logits = self.concept_head(z)           # (B,D)
        c_prob   = torch.sigmoid(c_logits)        # (B,D)

        # When only concepts are requested, DO NOT run evidential fusion.
        if return_concepts:
            return None, c_prob, c_logits

        # Only run evidential fusion when classification output is needed.
        bel, mtheta = self.forward_fusion_only(c_prob)  # (B,C), (B,1)
        betp = self._betp_from_bel(bel, mtheta)         # (B,C)
        log_betp = torch.log(torch.clamp(betp, min=self._betp_clip))
        return log_betp

    def concepts_only(self, x: torch.Tensor):
        """Return (c_prob, c_logits) without triggering evidential fusion.
        This mirrors CBMBase.concepts_only for stage-1 sequential training.
        """
        z = self.encoder(x)                    # (B,2048)
        c_logits = self.concept_head(z)        # (B,D) logits
        c_prob = torch.sigmoid(c_logits)       # (B,D)
        return c_prob, c_logits

    def concept_bce_logits(self, c_logits: torch.Tensor, c_gt: torch.Tensor) -> torch.Tensor:
        return self._bce_logits(c_logits, c_gt)

    def loss_fn(self, log_betp: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # NLL over log-prob (betp)
        return self._nll(log_betp, y)
