from collections.abc import Mapping

from data_interfaces.geometry import normalize_positions
from data_interfaces.interfaces import prepare_sensor_input


def prepare_sharpa(payload, device="cpu", calibration=None):
    if not isinstance(payload, Mapping):
        raise ValueError("Sharpa input must be a mapping")
    if payload.get("signal_space") != "raw":
        raise ValueError("signal_space must be raw; do not normalize the signal twice")
    if payload.get("hand_convention") != "canonical_right":
        raise ValueError("Canonicalize hand, taxel order, and signal axes before inference")
    space = payload.get("geometry_space")
    if space == "canonical_spherical":
        if calibration is not None:
            raise ValueError("Canonical geometry must not be transformed again")
        positions = payload["pos_xyz"]
    elif space == "sensor_xyz" and calibration is not None:
        if not isinstance(calibration, Mapping):
            raise ValueError("calibration must be a mapping")
        positions = normalize_positions(payload["pos_xyz"], **calibration)
    else:
        raise ValueError("Provide canonical_spherical geometry or sensor_xyz with --calibration")
    return prepare_sensor_input(payload["raw"], positions, payload.get("coverage_mask"), modality="visuotactile", finger_names=payload["finger_names"], measurement="z", scales_xyz=[1., 1., 255.]).to(device)
