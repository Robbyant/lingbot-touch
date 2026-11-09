from dataclasses import dataclass

import torch

from tactile_jepa_inference.config import MODALITIES
from .normalization import normalize_signal


FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")


def encode_tactile(encoder, signal, pos_xyz, *, modality, coverage_mask=None):
    if not isinstance(modality, str) or modality not in MODALITIES:
        raise ValueError("modality must be magnetic, piezoresistive, or visuotactile")
    if not isinstance(signal, torch.Tensor) or not signal.is_floating_point() or signal.ndim not in (4, 5) or signal.shape[-3] != 5 or signal.shape[-1] != 3:
        raise ValueError("signal must be floating [T,5,M,3] or [B,T,5,M,3]")
    if signal.ndim == 4:
        signal = signal.unsqueeze(0)
    if coverage_mask is None:
        coverage_mask = torch.ones((5, signal.shape[-2]), dtype=torch.bool, device=signal.device)
    return encoder.encode(signal, pos_xyz, coverage_mask, modality=modality)


@dataclass(frozen=True)
class TactileInput:
    signal: torch.Tensor
    pos_xyz: torch.Tensor
    coverage_mask: torch.Tensor
    modality: str

    def to(self, device):
        return TactileInput(self.signal.to(device), self.pos_xyz.to(device), self.coverage_mask.to(device), self.modality)

    def encode(self, encoder):
        return encode_tactile(encoder, self.signal, self.pos_xyz, modality=self.modality, coverage_mask=self.coverage_mask)


def prepare_sensor_input(raw, pos_xyz, coverage_mask=None, *, modality, finger_names, measurement, scales_xyz):
    if not isinstance(modality, str) or modality not in MODALITIES:
        raise ValueError("modality must be magnetic, piezoresistive, or visuotactile")
    if measurement not in ("xyz", "z"):
        raise ValueError("measurement must be xyz or z; RGB images are not supported")
    if not isinstance(raw, torch.Tensor) or raw.is_complex() or raw.dtype == torch.bool:
        raise ValueError("raw must be a real numeric tensor")
    expected_ndim = 5 if measurement == "xyz" else 4
    if raw.ndim != expected_ndim or min(raw.shape) < 1 or (measurement == "xyz" and raw.shape[-1] != 3):
        raise ValueError("raw must have shape [B,T,F,M,3] for xyz or [B,T,F,M] for z")
    fingers, points = raw.shape[2:4]
    names = tuple(finger_names)
    if not 1 <= fingers <= 5 or len(names) != fingers or len(set(names)) != fingers or any(n not in FINGER_ORDER for n in names):
        raise ValueError("finger_names must identify each input finger exactly once using canonical names")
    if coverage_mask is None:
        if fingers != 5:
            raise ValueError("coverage_mask is required for incomplete finger input")
        coverage_mask = torch.ones((fingers, points), dtype=torch.bool, device=raw.device)
    if not isinstance(coverage_mask, torch.Tensor) or coverage_mask.dtype != torch.bool or coverage_mask.shape != (fingers, points) or not bool(coverage_mask.any()):
        raise ValueError("coverage_mask must be nonempty boolean [F,M]")
    if not isinstance(pos_xyz, torch.Tensor) or pos_xyz.is_complex() or pos_xyz.shape != (fingers, points, 3):
        raise ValueError("pos_xyz must have shape [F,M,3] in canonical coordinates")
    values = raw.to(dtype=torch.float32)
    positions = pos_xyz.to(device=raw.device, dtype=torch.float32)
    coverage = coverage_mask.to(raw.device)
    if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(positions).all()):
        raise ValueError("Signals and positions must be finite, including padded slots")
    if measurement == "z":
        values = torch.stack((torch.zeros_like(values), torch.zeros_like(values), values), dim=-1)
    values = normalize_signal(values, coverage, {"mode": "per_axis", "scales_xyz": scales_xyz, "clip_to_range": True})
    signal = values.new_zeros((*values.shape[:2], 5, points, 3))
    geometry = positions.new_zeros((5, points, 3))
    mask = torch.zeros((5, points), device=raw.device, dtype=torch.bool)
    slots = torch.tensor([FINGER_ORDER.index(name) for name in names], device=raw.device)
    signal[:, :, slots] = values
    geometry[slots] = torch.where(coverage[..., None], positions, 0.)
    mask[slots] = coverage
    return TactileInput(signal, geometry, mask, modality)
