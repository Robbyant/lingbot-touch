from collections.abc import Mapping

from data_interfaces.interfaces import FINGER_ORDER
from data_interfaces.xhand import XHandToMagnetic96Input


def prepare_xhand(payload, device="cpu"):
    if not isinstance(payload, Mapping):
        raise ValueError("XHand input must be a mapping")
    if payload.get("signal_space") != "raw":
        raise ValueError("signal_space must be raw; selection precedes normalization")
    if payload.get("layout_id") != "xhand120_reference_v1":
        raise ValueError("layout_id must be xhand120_reference_v1; verify hardware point order first")
    if tuple(payload.get("finger_names", ())) != FINGER_ORDER:
        raise ValueError("XHand input finger order must be thumb, index, middle, ring, pinky")
    if "pos_xyz" in payload:
        raise ValueError("XHand 120-to-96 uses its bundled fixed target geometry, not arbitrary pos_xyz")
    adapter = XHandToMagnetic96Input().to(device)
    return adapter(payload["raw"].to(device), hand=payload["hand"])
