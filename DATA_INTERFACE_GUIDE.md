# Data Interface and Inference Guide

[Back to README](README.md)

Use the [README examples](README.md#Example) to run the included data. This guide describes input contracts and new-sensor integration. Inputs start with decoded tactile measurements and known taxel geometry, not RGB images or device packets. Inference returns frozen target-encoder tokens, without a predictor, pooling or downstream head.

## 1. Installation and Model Files

Follow [Environment Setup](README.md#Environment-Setup). The reference is Linux/x86_64, Python 3.12.13, PyTorch 2.11.0 with its CUDA 13.0 build, and NumPy 2.4.4. NumPy is installed as a package dependency to avoid PyTorch's missing-NumPy warning. The supplied commands work on CPU; GPU inference additionally requires compatible hardware and a driver for that PyTorch build. No training environment is required.

Download a model bundle from [ModelScope or Hugging Face](README.md#Models) into `checkpoints/lingbot-touch-tiny/` or `checkpoints/lingbot-touch-small/`. Model files are hosted separately from this source repository. Each model directory must contain a matched `config.json` and `encoder.pt`; do not rename a full training checkpoint or edit one architecture into another. Tiny uses tubelet size 2 and Small uses 4, so the supplied 16-frame samples work with both.

```python
from tactile_jepa_inference import load_encoder

encoder = load_encoder("checkpoints/lingbot-touch-tiny", device="cpu")
```

The loader checks exact keys, shapes, dtypes and finite values before strict loading. The encoder stays frozen and in evaluation mode. Use only trusted model/input files, loaded with `weights_only=True`.

## 2. Input Contracts and Modality

The fixed finger order is **thumb, index, middle, ring, pinky**. Infer the left and right hands independently after applying the appropriate canonical-right conversion; do not concatenate them into ten finger slots. Frame order must be chronological. T must be positive and divisible by the model's `tubelet_size`; 16 frames is the example setting, not an automatic temporal resampling operation.

| Prepared field | Contract |
| --- | --- |
| `signal` | Floating `[T,5,M,3]` or `[B,T,5,M,3]`. The public helper adds B=1 for a single sample. |
| `pos_xyz` | Floating `[5,M,3]` or `[B,5,M,3]`, matching the signal's taxel order in canonical space. |
| `coverage_mask` | Optional shared bool `[5,M]`; omitted means all slots are valid. Supply it explicitly for padding. |
| `modality` | Exactly `magnetic`, `piezoresistive` or `visuotactile`, selected by sensing principle. |

For magnetic measurements, each taxel is `[X_norm,Y_norm,Z_norm]`. For single-channel visuotactile or piezoresistive measurements it is `[0,0,Z_norm]`. Measurement XY must be in `[-1,1]` and Z in `[0,1]`. All signals and coordinates, including padding, must be finite in float32. Geometry coordinates are not measurement channels and need not be clipped to the measurement range.

The config's `group_by_source` and tokenizer keys use the three identically named modalities. Each call selects one tokenizer; modalities and different coverage layouts require separate calls. Merely changing a label does not physically adapt another sensor.

<a id="five-finger-coverage"></a>

### Five-Finger Coverage and Padding

Preserve complete five-finger input when available. **Real taxels remain `True` even when their signal is zero.** For valid counts `m0,...,m4`, set `M=max(m0,...,m4)`, place each finger's real values and coordinates in corresponding slots, and zero-pad nonexistent slots with `coverage_mask=False`. For example, `[124,61,61,61,61]` valid points require `[5,124]` coverage with only the first 61 slots valid on each non-thumb finger.

Coverage is not contact detection, random pretraining masking or a recommended way to discard fingers. It controls which spatial slots generate tokens and participate in self-attention; changing it can change all representations. The included Sharpa and adapted XHand samples have no padding, so all-`True` defaults are appropriate.

The generic raw adapter accepts F=1...5 with accurate `finger_names` and explicit `[F,M]` coverage when F<5. It places missing fingers in zero/`False` slots. Do not invent extra contacts by copying signals, or mark nonexistent points `True`. Partial-hand downstream quality requires separate validation; the fixed XHand adapter itself requires all five fingers.

## 3. Signal and Geometry Normalization

### Fixed-Range Signal Normalization

[prepare_sensor_input](data_interfaces/interfaces.py) calls [normalize_signal](data_interfaces/normalization.py). Given confirmed positive sensor ranges `s_x,s_y,s_z`, it applies:

```text
X_norm = clip(X / s_x, -1, 1)
Y_norm = clip(Y / s_y, -1, 1)
Z_norm = clip(Z / s_z,  0, 1)
```

For scalar data use `measurement="z"`, which fills XY with zeros; three-axis data uses `measurement="xyz"`. The generic adapter takes raw `[B,T,F,M]` or `[B,T,F,M,3]` respectively, shared canonical `pos_xyz:[F,M,3]`, fixed ranges and actual finger names, and returns a prepared `TactileInput`.

| Existing interface | Fixed normalization | Modality |
| --- | --- | --- |
| Sharpa decoded scalar intensity | Z/255; XY=0 | `visuotactile` |
| XHand raw XYZ, after its fixed selection | Divide by `[128,128,255]` | `magnetic` |
| New scalar sensor | Its confirmed positive Z reference range; XY=0 | Select by sensing principle |

Ranges are sensor-specific conventions, not universal physical units. Establish signal axes, positive contact direction and any zero-offset/drift correction beforehand. Do not estimate a new scale from each frame or clip. Monitor excessive clipping and check calibration/ranges if it occurs. Normalize only once: XHand needs no additional multiplication or division by another sensor's ranges. Inference adds no random signal/coordinate augmentation or random masks.

### Canonical Geometry

[normalize_positions](data_interfaces/geometry.py) performs this fixed transform independently per finger, using float64 intermediates and returning float32:

```text
p_canonical = (p_sensor @ rotation.T - coordinate_origin) / radius + offset_xyz
```

| Calibration key | Shape and meaning |
| --- | --- |
| `rotation` | `[3,3]` or `[F,3,3]`; orthonormal, determinant +1. Resolve hand reflections separately. |
| `coordinate_origin` | `[3]` or `[F,3]`; reference origin in the rotated coordinate frame. |
| `radius` | Positive scalar or `[F]`; fixed isotropic reference radius, in the same units as the points. |
| `offset_xyz` | `[3]` or `[F,3]`; fixed offset in canonical space. |

These parameters come from the complete sensor geometry and confirmed reference template, not current tactile intensity. They do not project each point radially onto a unit sphere. [height_aligned_origin](data_interfaces/geometry.py) can replace the origin's Y coordinate with the supplied fitted sphere center's Y; it does not fit the sphere. Both origins are in the rotated frame. Preserve an established layout's reference radius and offsets, including any template-specific alignment, rather than expanding a local patch into a whole fingertip.

New sensors must provide verified coordinate axes, units, hand/taxel correspondence and fixed calibration. The model config does not contain or infer those quantities. The included Sharpa files already have canonical coordinates; XHand obtains fixed canonical target geometry from its adapter. **Do not transform either example again.**

## 4. Sharpa Wave: Raw Scalar Input to Visuotactile Tokens

[prepare_sharpa](data_interfaces/infer_sharpa.py) accepts this decoded input contract:

| Field | Required value |
| --- | --- |
| `raw` | Real numeric `[B,T,5,M]`; included samples use `[1,16,5,200]`. Not RGB. |
| `pos_xyz` | Corresponding `[5,M,3]` coordinates. |
| `finger_names` | Five actual names; the generic adapter reorders them into the fixed order. |
| `signal_space` | `raw`; do not divide by 255 beforehand. |
| `geometry_space` | `canonical_spherical`, or `sensor_xyz` with fixed calibration. |
| `hand_convention` | `canonical_right`; original handedness has already been resolved. |
| `coverage_mask` | Optional bool `[5,M]`, required if padding is present. |

Run the complete bundled example from the release root:

```bash
mkdir -p outputs/sharpa
for HAND in left right; do
  python infer.py --input-format sharpa \
    --bundle checkpoints/lingbot-touch-tiny \
    --input "sample_data/sharpa/001_${HAND}.pt" \
    --output "outputs/sharpa/001_${HAND}_features.pt" --device cpu
done
```

Equivalent Python preparation and encoding:

```python
import torch
from data_interfaces.infer_sharpa import prepare_sharpa
from tactile_jepa_inference import load_encoder

payload = torch.load("sample_data/sharpa/001_left.pt", map_location="cpu", weights_only=True)
batch = prepare_sharpa(payload, device="cpu")
encoder = load_encoder("checkpoints/lingbot-touch-tiny", device="cpu")
result = batch.encode(encoder)
print(result.latent.shape)
```

For new Sharpa-format measurements with sensor-space positions, set `geometry_space="sensor_xyz"` and provide a JSON containing the four calibration keys above using `--calibration data/geometry_calibration.json`. This calibration file is not bundled: it must be supplied for the actual new layout. The Python equivalent is `prepare_sharpa(payload, calibration=calibration_dict)`. A calibration argument is rejected for already canonical coordinates to prevent duplicate transformation. This numerical interface does not compute deformation or force from RGB images.

## 5. XHand: Raw XYZ Input to Magnetic Tokens

[prepare_xhand](data_interfaces/infer_xhand.py) uses [XHandToMagnetic96Input](data_interfaces/xhand.py) and the packaged [fixed mapping](data_interfaces/data/xhand_to_magnetic96_v1.json).

| Field | Required value |
| --- | --- |
| `raw` | Real numeric `[B,T,5,120,3]` in the verified XHand120 finger/taxel/axis convention. |
| `finger_names` | Exactly `thumb,index,middle,ring,pinky`, in that order. |
| `signal_space` | `raw`; selection operates before clipping/normalization. |
| `layout_id` | `xhand120_reference_v1`; it asserts a verified layout, not arbitrary 120-point compatibility. |
| `hand` | `left` or `right`; use `canonical_right` only if the adapter's original conversion has already been applied. |

Do not include arbitrary `pos_xyz`: this path uses its fixed target coordinates. Left input receives the packaged point permutation and signal-axis signs; right input receives its corresponding axis signs. The adapter starts from 96 fixed points and, when the full finger's raw Z maximum is omitted, restores its complete XYZ vector to the mapped target slot per frame/finger. It is not Gaussian averaging and does not promise to preserve every local peak or the entire distribution.

Selection precedes division by `[128,128,255]` and clipping. The result is `[B,T,5,96,3]`, `modality="magnetic"`, fixed magnetic96 canonical coordinates and all-`True` `[5,96]` coverage. Do not clip raw inputs first or perform another normalization. SDK/layout changes require mapping verification; shape equality alone is insufficient.

```bash
mkdir -p outputs/xhand
for HAND in left right; do
  python infer.py --input-format xhand \
    --bundle checkpoints/lingbot-touch-tiny \
    --input "sample_data/xhand/001_${HAND}.pt" \
    --output "outputs/xhand/001_${HAND}_features.pt" --device cpu
done
```

```python
import torch
from data_interfaces.infer_xhand import prepare_xhand
from tactile_jepa_inference import load_encoder

payload = torch.load("sample_data/xhand/001_left.pt", map_location="cpu", weights_only=True)
batch = prepare_xhand(payload, device="cpu")
encoder = load_encoder("checkpoints/lingbot-touch-tiny", device="cpu")
result = batch.encode(encoder)
print(result.latent.shape)
```

## 6. Other Sensors and Prepared Inputs

For a new sensor, establish its axes, fixed ranges, finger/taxel order, handedness and canonical geometry first. Use `prepare_sensor_input(raw, pos_xyz, coverage_mask, modality=..., finger_names=..., measurement=..., scales_xyz=...)` with those values. The adapter does not infer physical calibration or choose a branch from signal shape.

For already normalized data, `encode_tactile` accepts four- or five-dimensional signals, adds a single-sample batch dimension if needed and defaults coverage to all `True`. It does not normalize, resample, cast or move the supplied tensors; match the encoder's device and use float32.

Save any prepared `batch` from the adapters above without normalizing again:

```python
from pathlib import Path
import torch
from data_interfaces import FINGER_ORDER

Path("data").mkdir(exist_ok=True)
torch.save({
    "signal": batch.signal.cpu(), "pos_xyz": batch.pos_xyz.cpu(),
    "coverage_mask": batch.coverage_mask.cpu(), "modality": batch.modality,
    "finger_names": list(FINGER_ORDER), "signal_space": "normalized",
    "geometry_space": "canonical_spherical", "hand_convention": "canonical_right",
}, "data/prepared_input.pt")
```

```bash
mkdir -p outputs
python infer.py --input-format normalized \
  --bundle checkpoints/lingbot-touch-tiny \
  --input data/prepared_input.pt --output outputs/prepared_features.pt --device cpu
```

The normalized contract uses the prepared shapes in Section 2; `raw` must be absent. The CLI casts floating signals/positions to float32 and moves them to the requested device, but rejects nonfinite values, invalid shapes/ranges, nonzero scalar XY and incorrect metadata. It does not repair, clip or renormalize already prepared measurements. `--calibration` applies only to raw Sharpa-format input, not normalized or XHand input.

## 7. Optional Sampling

The package provides six magnetic references (**96,76,64,48,32,24** points/finger), three piezoresistive references (**61,16,9**), and three visuotactile references (**200,96,32**). The piezoresistive 61-point layout retains 124 thumb points and 61 points on each other finger, padded to 124 slots with a coverage mask. Matching confirmed geometry/resolution can help adaptation, but is optional and does not guarantee downstream gains. The included 200-point Sharpa input needs no further sampling.

[load_reference_geometry](data_interfaces/sampling.py) returns `pos_xyz:[5,M,3]`, bool `coverage_mask:[5,M]`, int64 `taxel_part:[5,M]`, `modality` and `count` from the [twelve reference geometries](data_interfaces/data/multires_references_v1.json). Use `load_reference_geometry("piezoresistive", count)` with `count=61`, `16` or `9`; the 61-point reference has `M=124` and valid counts `[124,61,61,61,61]`. References do not calibrate arbitrary sensors or define new source-to-target correspondences.

[prepare_pretraining_reference_input](data_interfaces/pretraining_sampling.py) reuses [fixed native-layout maps](data_interfaces/data/pretraining_sampling_maps_v1.json):

- `magnetic_native116_96_canonical_right_v1`: raw `[B,T,5,116,3]`; 116 thumb points and 96 valid points followed by 20 padded slots on each other finger. It supports the six magnetic targets, using raw sampling then normalization by `[2.56,2.56,5.12]`.
- `visuotactile_deformation240_canonical_right_v1`: raw scalar `[B,T,5,240,240]` deformation maps, not RGB or the 200-point Sharpa files. It supports the three visuotactile targets, using raw sampling then Z/255 normalization.

Both require finite input in the verified native order, canonical-right handedness and the fixed five-finger order. They return `TactileInput`; they do not fit geometry, convert unknown hand conventions or create new maps. Never apply them to already normalized inputs.

For another confirmed layout, [FixedSpatialSampler](data_interfaces/sampling.py) consumes a prepared `TactileInput`. Its constructor binds exact source float32 positions `[5,N,3]`, bool coverage `[5,N]`, int64 source `indices:[5,M,K]`, float32 `weights:[5,M,K]`, `modality` and target `num_points`. Weights must be finite, nonnegative and sum to one per target; positive weights may reference only valid source taxels on the same finger. Input geometry/coverage must match exactly and share the module's device. The weighted complete XYZ outputs are clipped to normalized bounds and returned with target reference geometry and unchanged modality.

New-layout indices and weights must be provided and verified by the user; point count alone cannot establish correspondence. This normalized-input weighted operator is not equivalent to XHand's raw peak-restoring selection, nor necessarily to native raw sampling before clipping. Different resolutions need not produce identical representations.

## 8. Outputs, Tests and Supported Entry Points

All three formats share one argument parser, one model load, one encode and one exclusive output save. Run `python infer.py --help` from the release root or `python -m data_interfaces.infer --help` after installation. Existing output files are refused, not overwritten. Output parents must exist; CPU is the default and a compatible GPU can use `--device cuda:0`.

Each output contains exactly `latent:[B,N,C]`, `time_id:[N]`, `finger_id:[N]` and `pos_xyz:[B,N,3]`, where `N=(T/tubelet_size)*coverage_mask.sum()` and `C=embed_dim*len(output_layers)`. Time and finger indices are shared across the batch; expand them explicitly if a downstream interface needs `[B,N]`. The encoder produces no pooled embedding or task prediction. Preserve token addresses and use attention-based readout for downstream integration; selecting token addresses does not undo contextual mixing.

After the README batch commands, read `outputs/batch_sharpa/001_left_features.pt` and `outputs/batch_xhand/001_left_features.pt`. The separate single-example commands instead write to `outputs/sharpa/` and `outputs/xhand/`.

<a id="interface-validation"></a>

These optional checks verify inference correctness and are not required to run inference. `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4` limit CPU parallelism to four threads. Run from the release root with both actual model bundles available:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m unittest discover -s tests -v
```

Tests cover sample hashes, real bilateral Tiny/Small inference, raw/normalized CLI agreement, single-pass parsing, calibration rejection, normalization, padding, modality routes and invalid inputs. Missing weights or LFS pointers are failures, not skipped model checks. Tests do not establish downstream accuracy, real-time latency, arbitrary SDK ordering or new-hardware calibration; the piezoresistive route check is synthetic.

`sample_data/` contains 30 bilateral 16-frame input pairs per platform, 120 single-hand files total. Infer each hand independently. The [manifest](sample_data/manifest.json) lists file paths, hashes, shapes and sensing modalities. These clips are interface examples, not a representative evaluation set.

Use `tactile_jepa_inference` for the encoder and `data_interfaces` for input preparation, geometry and sampling.
