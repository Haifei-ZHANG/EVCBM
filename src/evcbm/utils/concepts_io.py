
# evcbm/utils/concepts_io.py
# Utilities to save/load the "concept backbone" (encoder + concept_head) per fold.
# Safe loading: prefer torch.load(weights_only=True) when available to avoid pickle risks.
# Also allow choosing map_location (GPU/CPU).

import os
import torch

_PREFIXES = ("encoder.", "concept_head.")

def extract_concept_state_dict(full_state_dict: dict) -> dict:
    """Return only encoder.* and concept_head.* parameters from a (module) state_dict."""
    return {k: v for k, v in full_state_dict.items() if k.startswith(_PREFIXES)}

def save_concept_backbone(model, path: str):
    """Save only the encoder + concept_head weights for reuse in sequential Phase B."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = extract_concept_state_dict(model.state_dict())
    torch.save(state, path)

def _safe_torch_load(path: str, map_location: str = "cpu"):
    """Load checkpoint safely: prefer weights_only=True if this torch version supports it."""
    try:
        # PyTorch >= 2.4 (experimental). Avoids arbitrary code execution from untrusted pickles.
        return torch.load(path, map_location=map_location, weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        # Older PyTorch: fall back to classic load.
        return torch.load(path, map_location=map_location)

def load_concept_backbone(model,
                          path: str,
                          freeze: bool = True,
                          device: str = None,
                          prefer_gpu: bool = True):
    """
    Load encoder.+concept_head.* weights into `model` from `path`.
    Args:
        freeze: if True, set requires_grad=False for backbone params.
        device: explicit map_location, e.g., "cuda", "cuda:0", or "cpu".
        prefer_gpu: if True and device is None, will use "cuda" when available, else "cpu".

    Notes:
        - Uses safe loading if available.
        - Only loads intersecting backbone keys (encoder./concept_head.).
        - map_location only affects where the raw checkpoint tensors are loaded;
          parameters will reside on the model's existing device after load_state_dict.
    """
    if device is None:
        device = "cuda" if (prefer_gpu and torch.cuda.is_available()) else "cpu"

    state = _safe_torch_load(path, map_location=device)

    # Some checkpoints may wrap weights under 'state_dict'
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    # Filter to encoder./concept_head. and keys that exist in the current model
    target_keys = set(k for k in model.state_dict().keys() if k.startswith(_PREFIXES))
    filtered = {k: v for k, v in state.items() if k in target_keys}

    # Load non-strictly just in case BN buffers differ; but we only pass filtered keys
    model.load_state_dict(filtered, strict=False)

    if freeze:
        for name, p in model.named_parameters():
            if name.startswith(_PREFIXES):
                p.requires_grad = False
