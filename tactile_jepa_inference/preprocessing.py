import math

import torch


def _positive(value, dtype):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Normalization bounds must be numeric")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Normalization bounds must be finite and positive")
    converted = torch.tensor(value, dtype=dtype)
    if not bool(torch.isfinite(converted)) or not bool(converted > 0):
        raise ValueError("Normalization bounds must remain finite and positive in the signal dtype")
    return float(value)


def normalize_signal(signal, coverage_mask, spec):
    if signal.ndim < 3 or signal.shape[-1] != 3 or not signal.is_floating_point():
        raise ValueError("signal must be floating point with trailing XYZ channels")
    if coverage_mask.dtype != torch.bool or tuple(coverage_mask.shape) != tuple(signal.shape[-3:-1]):
        raise ValueError("coverage_mask must be boolean and match the spatial shape")
    if not bool(torch.isfinite(signal).all()):
        raise ValueError("signal contains nonfinite values")
    if not isinstance(spec.get("clip_to_range"), bool):
        raise ValueError("clip_to_range must be explicit boolean")
    mode = spec["mode"]
    valid = coverage_mask.to(signal.device)
    output = torch.zeros_like(signal)
    if mode in {"bounded_unit_range_v1", "per_axis"}:
        scales = spec["scales_xyz"]
        if len(scales) != 3:
            raise ValueError("Three normalization scales are required")
        output = signal / signal.new_tensor([_positive(value, signal.dtype) for value in scales])
        if spec["clip_to_range"]:
            output[..., :2] = output[..., :2].clamp(-1, 1)
            output[..., 2] = output[..., 2].clamp(0, 1)
    elif mode == "range_to_unit_v1":
        measurement = spec["measurement_mode"]
        if measurement not in {"only_z", "force_xyz"}:
            raise ValueError("Unsupported measurement_mode")
        axes = (2,) if measurement == "only_z" else (0, 1, 2)
        for axis in axes:
            key = ("x_abs_max", "y_abs_max", "z_max")[axis]
            unit = signal[..., axis] / _positive(spec["value_range"][key], signal.dtype)
            if spec["clip_to_range"]:
                unit = unit.clamp(-1.0 if axis < 2 else 0.0, 1.0)
            output[..., axis] = unit
    else:
        raise ValueError("Unsupported normalization mode")
    output = torch.where(valid[..., None], output, torch.zeros((), dtype=output.dtype, device=output.device))
    if not bool(torch.isfinite(output).all()):
        raise ValueError("Normalization produced nonfinite values")
    return output
