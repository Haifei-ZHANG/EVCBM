# -*- coding: utf-8 -*-
import gc
import torch

def cleanup_torch():
    """
    Free Python refs + clear CUDA caches. Call after finishing a fold or a model run.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()   # release cached blocks back to the driver
        torch.cuda.ipc_collect()   # collect interprocess memory (safe even if not using IPC)
