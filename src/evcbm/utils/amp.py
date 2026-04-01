# -*- coding: utf-8 -*-

# src/evcbm/utils/amp.py
from contextlib import nullcontext
import torch

def make_amp(use_fp16: bool):
    """
    返回 (scaler, autocast_ctx)
      - 若 use_fp16 且 CUDA 可用：GradScaler() + autocast(device_type='cuda', enabled=True)
      - 否则：scaler=None, autocast 为空上下文
    兼容老版本 PyTorch：GradScaler 不传 device/device_type。
    """
    if use_fp16 and torch.cuda.is_available():
        # 旧版 torch.amp.GradScaler 不支持 device/device_type，直接用默认构造
        scaler = torch.amp.GradScaler(enabled=True)
        autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=True)
    else:
        scaler = None
        autocast_ctx = nullcontext()
    return scaler, autocast_ctx
