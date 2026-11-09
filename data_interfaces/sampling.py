import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .interfaces import FINGER_ORDER, TactileInput


_REFERENCE_SHA256 = "53437b563db58b6033e990e3f977b7c84d91e36e412269302b8e2512fbace9fb"
_REFERENCE_COUNTS = {"magnetic": (96, 76, 64, 48, 32, 24), "piezoresistive": (61, 16, 9), "visuotactile": (200, 96, 32)}


def load_reference_geometry(modality, num_points):
    if not isinstance(modality, str) or modality not in _REFERENCE_COUNTS:
        raise ValueError("Reference modality must be magnetic, piezoresistive, or visuotactile")
    if type(num_points) is not int or num_points not in _REFERENCE_COUNTS[modality]:
        raise ValueError(f"Supported {modality} reference counts are {_REFERENCE_COUNTS[modality]}")
    payload = (Path(__file__).parent / "data" / "multires_references_v1.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != _REFERENCE_SHA256:
        raise ValueError("Reference geometry asset checksum mismatch")
    asset = json.loads(payload)
    if asset["schema"] != "multires_references_v1" or tuple(asset["finger_order"]) != FINGER_ORDER or asset["canonical_right"] is not True:
        raise ValueError("Reference geometry schema or hand convention mismatch")
    entry = asset["entries"][f"{modality}_{num_points:03d}"]
    return {
        "modality": entry["modality"],
        "count": entry["count"],
        "pos_xyz": torch.tensor(entry["pos_xyz"], dtype=torch.float32),
        "coverage_mask": torch.tensor(entry["coverage_mask"], dtype=torch.bool),
        "taxel_part": torch.tensor(entry["taxel_part"], dtype=torch.int64),
    }


class FixedSpatialSampler(nn.Module):
    def __init__(self, source_pos_xyz, source_coverage_mask, indices, weights, *, modality, num_points):
        super().__init__()
        reference = load_reference_geometry(modality, num_points)
        if not isinstance(source_pos_xyz, torch.Tensor) or source_pos_xyz.dtype != torch.float32 or source_pos_xyz.ndim != 3 or source_pos_xyz.shape[0] != 5 or source_pos_xyz.shape[1] < 1 or source_pos_xyz.shape[2] != 3 or not bool(torch.isfinite(source_pos_xyz).all()):
            raise ValueError("source_pos_xyz must be finite float32 [5,N,3]")
        source_shape = source_pos_xyz.shape[:2]
        if not isinstance(source_coverage_mask, torch.Tensor) or source_coverage_mask.dtype != torch.bool or source_coverage_mask.shape != source_shape or not bool(source_coverage_mask.any()):
            raise ValueError("source_coverage_mask must be nonempty boolean [5,N]")
        if not isinstance(indices, torch.Tensor) or indices.dtype != torch.int64 or indices.ndim != 3 or indices.shape[:2] != reference["coverage_mask"].shape or indices.shape[2] < 1:
            raise ValueError("indices must be int64 [5,M,K] with K > 0")
        if not isinstance(weights, torch.Tensor) or weights.dtype != torch.float32 or weights.shape != indices.shape:
            raise ValueError("weights must be float32 with the same [5,M,K] shape as indices")
        if any(value.device != source_pos_xyz.device for value in (source_coverage_mask, indices, weights)):
            raise ValueError("Source geometry, coverage, indices, and weights must share a device")
        if not bool(((indices >= 0) & (indices < source_shape[1])).all()):
            raise ValueError("indices must address existing source points within each finger")
        if not bool(torch.isfinite(weights).all()) or not bool((weights >= 0).all()):
            raise ValueError("weights must be finite and nonnegative")
        totals = weights.sum(dim=-1, keepdim=True)
        if not torch.allclose(totals, torch.ones_like(totals), atol=1e-6, rtol=0):
            raise ValueError("weights must sum to one for every target point")
        valid_sources = source_coverage_mask.gather(1, indices.flatten(1)).reshape_as(indices)
        if bool(((weights > 0) & ~valid_sources).any()):
            raise ValueError("Positive weights require valid source coverage for every target point")
        self.modality = modality
        self.num_points = num_points
        for name, value in (("source_pos_xyz", source_pos_xyz), ("source_coverage_mask", source_coverage_mask), ("indices", indices), ("weights", weights)):
            self.register_buffer(name, value.detach().clone())
        self.register_buffer("target_pos_xyz", reference["pos_xyz"].to(source_pos_xyz.device))
        self.register_buffer("target_coverage_mask", reference["coverage_mask"].to(source_pos_xyz.device))

    def forward(self, batch):
        if not isinstance(batch, TactileInput):
            raise ValueError("batch must be a prepared TactileInput")
        modality = batch.modality
        if modality != self.modality:
            raise ValueError("batch modality must match the fixed sampling map")
        signal = batch.signal
        if not isinstance(signal, torch.Tensor) or signal.dtype != torch.float32 or signal.ndim != 5 or min(signal.shape[:2]) < 1 or signal.shape[2:] != self.source_pos_xyz.shape:
            raise ValueError("signal must be normalized float32 [B,T,5,N,3] matching the source geometry")
        if not isinstance(batch.pos_xyz, torch.Tensor) or batch.pos_xyz.dtype != torch.float32 or not isinstance(batch.coverage_mask, torch.Tensor) or batch.coverage_mask.dtype != torch.bool:
            raise ValueError("batch geometry must be float32 and coverage must be boolean")
        if any(value.device != signal.device for value in (batch.pos_xyz, batch.coverage_mask, *self.buffers())):
            raise ValueError("batch and sampling module must share a device; move both with .to(device)")
        if self.source_pos_xyz.dtype != torch.float32 or self.weights.dtype != torch.float32 or self.target_pos_xyz.dtype != torch.float32:
            raise ValueError("Sampling geometry and weights must remain float32")
        if not torch.equal(batch.pos_xyz, self.source_pos_xyz) or not torch.equal(batch.coverage_mask, self.source_coverage_mask):
            raise ValueError("batch geometry and coverage must exactly match the bound source layout")
        valid = signal[:, :, self.source_coverage_mask]
        if not bool(torch.isfinite(valid).all()) or not bool(((valid[..., :2] >= -1) & (valid[..., :2] <= 1)).all()) or not bool(((valid[..., 2] >= 0) & (valid[..., 2] <= 1)).all()):
            raise ValueError("Valid signal values must be finite and normalized: XY in [-1,1], Z in [0,1]")
        if modality != "magnetic" and bool((valid[..., :2] != 0).any()):
            raise ValueError("visuotactile and piezoresistive scalar signal XY must be zero")
        values = torch.where(self.source_coverage_mask[None, None, ..., None], signal, 0.)
        fingers = torch.arange(5, device=signal.device)[:, None, None]
        gathered = values[:, :, fingers, self.indices]
        sampled = (gathered * self.weights[None, None, ..., None]).sum(dim=-2)
        sampled = torch.cat((sampled[..., :2].clamp(-1, 1), sampled[..., 2:].clamp(0, 1)), dim=-1)
        sampled = torch.where(self.target_coverage_mask[None, None, ..., None], sampled, 0.)
        return TactileInput(sampled, self.target_pos_xyz.clone(), self.target_coverage_mask.clone(), batch.modality)
