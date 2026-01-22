import os
import sys
from typing import Optional

import torch


def _ensure_seanet_path():
    seanet_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "SEANet"))
    if seanet_root not in sys.path:
        sys.path.append(seanet_root)


def load_seanet_weights(model: torch.nn.Module, path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"SEANet checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu")
    model_state = model.state_dict()
    for name, param in state.items():
        target_name = name
        if target_name not in model_state:
            if name.startswith("model."):
                stripped = name[len("model.") :]
                if stripped in model_state:
                    target_name = stripped
                else:
                    continue
            else:
                alt_name = f"model.{name}"
                if alt_name in model_state:
                    target_name = alt_name
                else:
                    continue
        if model_state[target_name].size() != param.size():
            raise ValueError(
                f"SEANet parameter size mismatch for {target_name}: "
                f"model={model_state[target_name].size()}, ckpt={param.size()}"
            )
        model_state[target_name].copy_(param)


def build_seanet(checkpoint_path: Optional[str] = None) -> torch.nn.Module:
    _ensure_seanet_path()
    from model.seanet import seanet  # type: ignore

    model = seanet(256, 40, 64, 128, 100, 6)
    if checkpoint_path:
        load_seanet_weights(model, checkpoint_path)
    return model
