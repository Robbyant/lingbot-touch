import hashlib
from importlib.resources import files
import json

import torch
from torch import nn

from .interfaces import FINGER_ORDER, prepare_sensor_input


REFERENCE_SHA256 = "29494b718105c4c3c4499218b154ac25fa0952e6844a746fa01ee0257411fa7a"


class XHandToMagnetic96Input(nn.Module):
    def __init__(self):
        super().__init__()
        raw = files("data_interfaces").joinpath("data/xhand_to_magnetic96_v1.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
            raise ValueError("XHand reference asset checksum mismatch")
        asset = json.loads(raw)
        for name, key, dtype in (
            ("indices", "base_indices", torch.int64),
            ("source_to_target", "source_to_target", torch.int64),
            ("left_indices", "left_source_indices", torch.int64),
            ("left_signs", "left_axis_signs", torch.float32),
            ("right_signs", "right_axis_signs", torch.float32),
            ("target_positions", "target_positions", torch.float32),
        ):
            self.register_buffer(name, torch.tensor(asset[key], dtype=dtype), persistent=False)
        self.register_buffer("coverage_mask", torch.ones(5, 96, dtype=torch.bool), persistent=False)

    def select_raw(self, raw, *, hand):
        if hand not in ("left", "right", "canonical_right"):
            raise ValueError("hand must be left, right, or canonical_right")
        if not isinstance(raw, torch.Tensor) or raw.ndim != 5 or raw.shape[-3:] != (5, 120, 3) or min(raw.shape) < 1:
            raise ValueError("XHand input must have shape [B,T,5,120,3] in reference finger/taxel order")
        if raw.is_complex() or raw.dtype == torch.bool:
            raise ValueError("XHand input must be a real numeric tensor")
        if raw.device != self.indices.device:
            raise ValueError("Move adapter and input to the same device")
        signal = raw.float()
        if not bool(torch.isfinite(signal).all()):
            raise ValueError("XHand input contains nonfinite float32 values")
        if hand == "left":
            permutation = self.left_indices[None, None, ..., None].expand(*signal.shape)
            signal = signal.gather(-2, permutation) * self.left_signs
        elif hand == "right":
            signal = signal * self.right_signs
        base = self.indices.expand(*signal.shape[:2], 5, 96)
        z = signal[..., 2]
        peak = z.argmax(dim=-1, keepdim=True)
        omitted = z.gather(-1, peak) > z.gather(-1, base).amax(dim=-1, keepdim=True)
        target = self.source_to_target.expand(*signal.shape[:2], 5, 120).gather(-1, peak)
        replacement = torch.where(omitted, peak, base.gather(-1, target))
        indices = base.scatter(-1, target, replacement)
        return signal.gather(-2, indices[..., None].expand(*indices.shape, 3))

    def forward(self, raw, *, hand):
        selected = self.select_raw(raw, hand=hand)
        return prepare_sensor_input(selected, self.target_positions, self.coverage_mask, modality="magnetic", finger_names=FINGER_ORDER, measurement="xyz", scales_xyz=[128., 128., 255.])
