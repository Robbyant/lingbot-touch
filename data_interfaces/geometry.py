import torch


def _finite64(value, device):
    tensor = torch.as_tensor(value, dtype=torch.float64, device=device)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("Geometry and calibration must be finite")
    return tensor


def height_aligned_origin(geometric_origin, sphere_center):
    origin = _finite64(geometric_origin, "cpu")
    center = _finite64(sphere_center, "cpu")
    if origin.shape != center.shape or origin.ndim not in (1, 2) or origin.shape[-1] != 3:
        raise ValueError("Origins and sphere centers must have matching [3] or [F,3] shapes")
    result = origin.clone()
    result[..., 1] = center[..., 1]
    return result


def normalize_positions(positions, *, rotation, coordinate_origin, radius, offset_xyz):
    values = torch.as_tensor(positions)
    device = values.device
    values = _finite64(values, device)
    if values.ndim != 3 or values.shape[2] != 3 or not 1 <= values.shape[0] <= 5 or values.shape[1] < 1:
        raise ValueError("positions must have shape [F,M,3] with 1 <= F <= 5 and M > 0")
    fingers = values.shape[0]
    matrix = _finite64(rotation, device)
    if matrix.shape == (3, 3):
        matrix = matrix.expand(fingers, 3, 3)
    if matrix.shape != (fingers, 3, 3):
        raise ValueError("rotation must have shape [3,3] or [F,3,3]")
    identity = torch.eye(3, dtype=torch.float64, device=device).expand(fingers, 3, 3)
    if not torch.allclose(matrix @ matrix.transpose(-1, -2), identity, atol=1e-6, rtol=0):
        raise ValueError("rotation must be orthonormal, without scaling or shearing")
    if not torch.allclose(torch.linalg.det(matrix), torch.ones(fingers, dtype=torch.float64, device=device), atol=1e-6, rtol=0):
        raise ValueError("rotation must have determinant +1; resolve hand mirroring separately")
    origin = _finite64(coordinate_origin, device)
    offset = _finite64(offset_xyz, device)
    if origin.shape == (3,):
        origin = origin.expand(fingers, 3)
    if offset.shape == (3,):
        offset = offset.expand(fingers, 3)
    if origin.shape != (fingers, 3) or offset.shape != (fingers, 3):
        raise ValueError("coordinate_origin and offset_xyz must have shape [3] or [F,3]")
    scale = _finite64(radius, device)
    if scale.ndim == 0:
        scale = scale.expand(fingers)
    if scale.shape != (fingers,) or not bool((scale > 0).all()):
        raise ValueError("radius must be a positive scalar or [F] isotropic reference radius")
    result = ((values @ matrix.transpose(-1, -2) - origin[:, None]) / scale[:, None, None] + offset[:, None]).float()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Geometry normalization produced nonfinite float32 positions")
    return result
