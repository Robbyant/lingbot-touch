from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from data_interfaces import FINGER_ORDER, XHandToMagnetic96Input, encode_tactile, normalize_positions, prepare_sensor_input
from data_interfaces.infer import prepare_normalized
from tactile_jepa_inference import load_encoder


ROOT = Path(__file__).resolve().parents[1]


def prepare_sample(sensor, hand):
    payload = torch.load(ROOT / f"sample_data/{sensor}/001_{hand}.pt", map_location="cpu", weights_only=True)
    if sensor == "xhand":
        return XHandToMagnetic96Input()(payload["raw"], hand=payload["hand"])
    return prepare_sensor_input(
        payload["raw"], payload["pos_xyz"], payload.get("coverage_mask"),
        modality="visuotactile", finger_names=payload["finger_names"],
        measurement="z", scales_xyz=[1., 1., 255.],
    )


def normalized_payload(batch):
    return {
        "signal": batch.signal, "pos_xyz": batch.pos_xyz,
        "coverage_mask": batch.coverage_mask, "modality": batch.modality,
        "signal_space": "normalized", "geometry_space": "canonical_spherical",
        "hand_convention": "canonical_right", "finger_names": list(FINGER_ORDER),
    }


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.encoders = {}
        for size in ("tiny", "small"):
            bundle = ROOT / f"checkpoints/lingbot-touch-{size}"
            path = bundle / "encoder.pt"
            if not path.is_file():
                raise RuntimeError("Download both model bundles before testing")
            with path.open("rb") as stream:
                if stream.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
                    raise RuntimeError("Download actual model tensors from ModelScope or Hugging Face; see README Models")
            cls.encoders[size] = load_encoder(bundle, device="cpu")

    def assert_output_equal(self, actual, expected):
        self.assertTrue(torch.equal(actual.latent, expected.latent))
        for name in ("time_id", "finger_id", "pos_xyz"):
            self.assertTrue(torch.equal(getattr(actual.address, name), getattr(expected.address, name)))

    def test_real_bilateral_samples_public_and_direct_encoder(self):
        for size, encoder in self.encoders.items():
            self.assertFalse(encoder.training)
            self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))
            for sensor in ("sharpa", "xhand"):
                for hand in ("left", "right"):
                    with self.subTest(model=size, sensor=sensor, hand=hand):
                        batch = prepare_sample(sensor, hand)
                        actual = encode_tactile(encoder, batch.signal[0], batch.pos_xyz, modality=batch.modality)
                        expected = encoder.encode(batch.signal, batch.pos_xyz, batch.coverage_mask, modality=batch.modality)
                        self.assert_output_equal(actual, expected)
                        count = batch.signal.shape[1] // encoder.config.tubelet_size * int(batch.coverage_mask.sum())
                        channels = encoder.config.embed_dim * len(encoder.config.output_layers)
                        self.assertEqual(tuple(actual.latent.shape), (1, count, channels))
                        self.assertEqual(tuple(actual.address.time_id.shape), (count,))
                        self.assertEqual(tuple(actual.address.finger_id.shape), (count,))
                        self.assertEqual(tuple(actual.address.pos_xyz.shape), (1, count, 3))
                        self.assertTrue(bool(torch.isfinite(actual.latent).all()))
                        self.assertFalse(actual.latent.requires_grad)

    def test_three_modality_routes(self):
        encoder = self.encoders["tiny"]
        signal = torch.zeros(1, 4, 5, 2, 3)
        signal[..., 2] = 0.2
        positions = torch.zeros(5, 2, 3)
        coverage = torch.ones(5, 2, dtype=torch.bool)
        for modality in ("magnetic", "piezoresistive", "visuotactile"):
            with self.subTest(modality=modality):
                actual = encode_tactile(encoder, signal, positions, modality=modality)
                expected = encoder.encode(signal, positions, coverage, modality=modality)
                self.assert_output_equal(actual, expected)

    def test_zero_signal_retains_valid_taxels_and_excludes_padding(self):
        coverage = torch.ones(5, 3, dtype=torch.bool)
        coverage[2, 2] = False
        batch = prepare_sensor_input(
            torch.zeros(1, 4, 5, 3), torch.zeros(5, 3, 3), coverage,
            modality="visuotactile", finger_names=FINGER_ORDER, measurement="z", scales_xyz=[1., 1., 255.],
        )
        self.assertTrue(torch.equal(batch.coverage_mask, coverage))
        output = batch.encode(self.encoders["tiny"])
        self.assertEqual(output.latent.shape, (1, 28, 768))
        self.assertTrue(bool(torch.isfinite(output.latent).all()))

    def test_scalar_normalization_and_geometry_formula(self):
        raw = torch.tensor([-1., 127.5, 300.]).reshape(1, 1, 1, 3).expand(1, 4, 5, 3)
        positions = torch.arange(45, dtype=torch.float32).reshape(5, 3, 3)
        batch = prepare_sensor_input(raw, positions, modality="visuotactile", finger_names=FINGER_ORDER, measurement="z", scales_xyz=[1., 1., 255.])
        self.assertTrue(torch.equal(batch.signal[..., 2], (raw / 255).clamp(0, 1)))
        self.assertEqual(torch.count_nonzero(batch.signal[..., :2]), 0)
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        actual = normalize_positions(positions, rotation=rotation, coordinate_origin=[1., 2., 3.], radius=2., offset_xyz=[0., 1., 0.])
        expected = (positions @ rotation.T - torch.tensor([1., 2., 3.])) / 2 + torch.tensor([0., 1., 0.])
        self.assertTrue(torch.equal(actual, expected))

    def test_normalized_input_rejects_invalid_values(self):
        batch = prepare_sample("sharpa", "left")
        for value, channel in ((float("nan"), 2), (float("inf"), 2), (1.1, 2), (-0.1, 2), (0.1, 0)):
            with self.subTest(value=value, channel=channel):
                payload = normalized_payload(batch)
                payload["signal"] = batch.signal.clone()
                payload["signal"][0, 0, 0, 0, channel] = value
                with self.assertRaises(ValueError):
                    prepare_normalized(payload, "cpu")

    def test_cli_raw_and_normalized_match_and_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for size in ("tiny", "small"):
                for sensor in ("sharpa", "xhand"):
                    with self.subTest(model=size, sensor=sensor):
                        batch = prepare_sample(sensor, "left")
                        prepared = root / f"{size}_{sensor}_input.pt"
                        torch.save(normalized_payload(batch), prepared)
                        expected = batch.encode(self.encoders[size])
                        for input_format, input_path in ((sensor, ROOT / f"sample_data/{sensor}/001_left.pt"), ("normalized", prepared)):
                            output = root / f"{size}_{sensor}_{input_format}_output.pt"
                            command = [sys.executable, str(ROOT / "infer.py"), "--input-format", input_format, "--bundle", str(ROOT / f"checkpoints/lingbot-touch-{size}"), "--input", str(input_path), "--output", str(output), "--device", "cpu"]
                            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=180)
                            result = torch.load(output, map_location="cpu", weights_only=True)
                            self.assertEqual(set(result), {"latent", "time_id", "finger_id", "pos_xyz"})
                            self.assertTrue(torch.equal(result["latent"], expected.latent))
                            for name in ("time_id", "finger_id", "pos_xyz"):
                                self.assertTrue(torch.equal(result[name], getattr(expected.address, name)))
                            before = output.read_bytes()
                            refused = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
                            self.assertNotEqual(refused.returncode, 0)
                            self.assertIn("FileExistsError", refused.stderr)
                            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
