import argparse
from collections.abc import Mapping
import json
from pathlib import Path

import torch

from tactile_jepa_inference import load_encoder
from .interfaces import FINGER_ORDER, MODALITIES, encode_tactile


def prepare_normalized(payload, device):
    if not isinstance(payload, Mapping):
        raise ValueError("normalized input must be a mapping")
    for name, value in (("signal_space", "normalized"), ("geometry_space", "canonical_spherical"), ("hand_convention", "canonical_right")):
        if payload.get(name) != value:
            raise ValueError(f"{name} must be {value}")
    names = payload.get("finger_names")
    if not isinstance(names, (list, tuple)) or tuple(names) != FINGER_ORDER:
        raise ValueError("finger_names must be thumb, index, middle, ring, pinky")
    modality = payload.get("modality")
    if not isinstance(modality, str) or modality not in MODALITIES:
        raise ValueError("modality must be magnetic, piezoresistive, or visuotactile")
    if "raw" in payload:
        raise ValueError("normalized input must not contain raw measurements")
    values = []
    for name in ("signal", "pos_xyz"):
        value = payload.get(name)
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided or not value.is_floating_point():
            raise ValueError(f"{name} must be a floating strided tensor")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
        value = value.to(device=device, dtype=torch.float32)
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must remain finite in FP32")
        values.append(value)
    signal, positions = values
    if signal.ndim not in (4, 5) or signal.shape[-3] != 5 or signal.shape[-1] != 3 or min(signal.shape) < 1:
        raise ValueError("signal must be positive [T,5,M,3] or [B,T,5,M,3]")
    batch = signal.shape[0] if signal.ndim == 5 else 1
    points = signal.shape[-2]
    if positions.shape not in ((5, points, 3), (batch, 5, points, 3)):
        raise ValueError("pos_xyz must be [5,M,3] or [B,5,M,3]")
    mask = payload.get("coverage_mask", torch.ones((5, points), dtype=torch.bool))
    if not isinstance(mask, torch.Tensor) or mask.layout != torch.strided or mask.dtype != torch.bool or mask.shape != (5, points) or not bool(mask.any()):
        raise ValueError("coverage_mask must be nonempty boolean [5,M]")
    mask = mask.to(device)
    valid = signal[..., mask, :]
    if bool((valid[..., :2].abs() > 1).any()) or bool(((valid[..., 2] < 0) | (valid[..., 2] > 1)).any()):
        raise ValueError("valid signal XY must be in [-1,1] and Z in [0,1]")
    if modality != "magnetic" and bool((valid[..., :2] != 0).any()):
        raise ValueError("visuotactile and piezoresistive signal XY must be zero")
    return signal, positions, mask, modality


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-format", choices=("sharpa", "xhand", "normalized"), required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args(argv)
    if args.calibration is not None and args.input_format != "sharpa":
        parser.error("--calibration is only supported for sharpa")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.output.resolve() in {args.input.resolve(), args.bundle.resolve(), (args.bundle / "config.json").resolve(), (args.bundle / "encoder.pt").resolve()}:
        raise ValueError("output must not overwrite input or bundle")
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    if args.input_format == "normalized":
        signal, positions, mask, modality = prepare_normalized(payload, args.device)
    else:
        if args.input_format == "sharpa":
            from .infer_sharpa import prepare_sharpa

            if args.calibration is not None and isinstance(payload, Mapping) and payload.get("geometry_space") == "canonical_spherical":
                raise ValueError("Canonical geometry must not be transformed again")
            calibration = json.loads(args.calibration.read_text(encoding="utf-8")) if args.calibration is not None else None
            batch = prepare_sharpa(payload, args.device, calibration)
        else:
            from .infer_xhand import prepare_xhand

            batch = prepare_xhand(payload, args.device)
        signal, positions, mask, modality = batch.signal, batch.pos_xyz, batch.coverage_mask, batch.modality
    encoder = load_encoder(args.bundle, device=args.device)
    if signal.shape[-4] % encoder.config.tubelet_size:
        raise ValueError("signal time length must be divisible by bundle tubelet_size")
    result = encode_tactile(encoder, signal, positions, modality=modality, coverage_mask=mask)
    output = {"latent": result.latent.cpu(), "time_id": result.address.time_id.cpu(), "finger_id": result.address.finger_id.cpu(), "pos_xyz": result.address.pos_xyz.cpu()}
    with args.output.open("xb") as stream:
        torch.save(output, stream)
    print("modality:", modality)
    print("latent:", tuple(result.latent.shape))
    print("output:", args.output)


if __name__ == "__main__":
    main()
