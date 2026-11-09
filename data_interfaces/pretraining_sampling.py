import hashlib
import json
from pathlib import Path

import torch

from .interfaces import FINGER_ORDER, prepare_sensor_input
from .sampling import load_reference_geometry


_MAP_SHA256 = "c1c6bc2390d6653984745d0cc559b82a85809d6df1e26a68f18902c129b5d57e"
_REFERENCE_SHA256 = "53437b563db58b6033e990e3f977b7c84d91e36e412269302b8e2512fbace9fb"
_LAYOUTS = {
    "magnetic_native116_96_canonical_right_v1": ("magnetic", (5, 116, 3), (96, 76, 64, 48, 32, 24)),
    "visuotactile_deformation240_canonical_right_v1": ("visuotactile", (5, 240, 240), (200, 96, 32)),
}


def prepare_pretraining_reference_input(raw, *, layout_id, num_points):
    if not isinstance(layout_id, str) or layout_id not in _LAYOUTS:
        raise ValueError(f"layout_id must explicitly identify one of {tuple(_LAYOUTS)}")
    modality, shape, supported = _LAYOUTS[layout_id]
    if type(num_points) is not int or num_points not in supported:
        raise ValueError(f"Supported point counts for {layout_id} are {supported}")
    if not isinstance(raw, torch.Tensor) or raw.is_complex() or raw.dtype == torch.bool:
        raise ValueError("raw must be a real numeric tensor")
    if raw.ndim != 5 or min(raw.shape[:2]) < 1 or tuple(raw.shape[2:]) != shape:
        raise ValueError(f"raw must have shape [B,T,{','.join(map(str, shape))}] for the declared source layout")
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("raw must be finite, including padded slots")
    values = raw.to(dtype=torch.float32)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("raw must remain finite in float32")
    payload = (Path(__file__).parent / "data/pretraining_sampling_maps_v1.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != _MAP_SHA256:
        raise ValueError("Native sampling map asset checksum mismatch")
    asset = json.loads(payload)
    if asset["schema"] != "pretraining_sampling_maps_v1" or tuple(asset["finger_order"]) != FINGER_ORDER or asset["canonical_right"] is not True or asset["reference_geometry_sha256"] != _REFERENCE_SHA256:
        raise ValueError("Native sampling map schema or reference geometry mismatch")
    source = asset["layouts"][layout_id]
    if source["modality"] != modality or tuple(source["source_shape"]) != shape:
        raise ValueError("Native sampling map source layout mismatch")
    entry = source["entries"][str(num_points)]
    indices = torch.tensor(entry["indices"], dtype=torch.int64, device=raw.device)
    fingers = torch.arange(5, device=raw.device)[:, None, None]
    if modality == "magnetic":
        weights = torch.tensor(entry["weights"], dtype=torch.float64, device=raw.device)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        gathered = values[:, :, fingers, indices].to(torch.float64)
        sampled = (gathered * weights[None, None, ..., None]).sum(dim=-2).float()
        measurement, scales = "xyz", [2.56, 2.56, 5.12]
    else:
        weights = torch.tensor(entry["weights"], dtype=torch.float32, device=raw.device)
        flattened = values.reshape(*values.shape[:3], 240 * 240)
        gathered = flattened[:, :, fingers, indices]
        sampled = (gathered * weights[None, None]).sum(dim=-1)
        measurement, scales = "z", [1., 1., 255.]
    reference = load_reference_geometry(modality, num_points)
    return prepare_sensor_input(
        sampled,
        reference["pos_xyz"].to(raw.device),
        reference["coverage_mask"].to(raw.device),
        modality=modality,
        finger_names=FINGER_ORDER,
        measurement=measurement,
        scales_xyz=scales,
    )
