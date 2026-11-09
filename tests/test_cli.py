import argparse
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from data_interfaces import FINGER_ORDER, XHandToMagnetic96Input, normalize_positions, prepare_sensor_input
from data_interfaces.infer import main


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)

    def adapter(self, sensor):
        module = importlib.import_module(f"data_interfaces.infer_{sensor}")
        self.assertTrue(callable(getattr(module, f"prepare_{sensor}", None)))
        return getattr(module, f"prepare_{sensor}")

    def sample(self, sensor, hand="left"):
        return torch.load(ROOT / f"sample_data/{sensor}/001_{hand}.pt", map_location="cpu", weights_only=True)

    def assert_batch_equal(self, actual, expected):
        self.assertEqual(actual.modality, expected.modality)
        for name in ("signal", "pos_xyz", "coverage_mask"):
            self.assertTrue(torch.equal(getattr(actual, name), getattr(expected, name)), name)

    def test_sharpa_preparation_matches_public_adapter(self):
        prepare = self.adapter("sharpa")
        for hand in ("left", "right"):
            with self.subTest(hand=hand):
                payload = self.sample("sharpa", hand)
                expected = prepare_sensor_input(payload["raw"], payload["pos_xyz"], payload.get("coverage_mask"), modality="visuotactile", finger_names=payload["finger_names"], measurement="z", scales_xyz=[1., 1., 255.])
                self.assert_batch_equal(prepare(payload), expected)

    def test_sharpa_calibrated_geometry_and_duplicate_transform(self):
        prepare = self.adapter("sharpa")
        payload = self.sample("sharpa")
        calibration = {"rotation": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], "coordinate_origin": [1., 2., 3.], "radius": 2., "offset_xyz": [0., 0., 0.]}
        with self.assertRaisesRegex(ValueError, "transformed again"):
            prepare(payload, calibration=calibration)
        payload["geometry_space"] = "sensor_xyz"
        with self.assertRaisesRegex(ValueError, "calibration"):
            prepare(payload)
        actual = prepare(payload, calibration=calibration)
        self.assertTrue(torch.equal(actual.pos_xyz, normalize_positions(payload["pos_xyz"], **calibration)))

    def test_xhand_preparation_matches_bilateral_reference(self):
        prepare = self.adapter("xhand")
        reference = XHandToMagnetic96Input()
        for hand in ("left", "right"):
            with self.subTest(hand=hand):
                payload = self.sample("xhand", hand)
                self.assert_batch_equal(prepare(payload), reference(payload["raw"], hand=payload["hand"]))

    def test_sensor_preparation_rejects_wrong_contract(self):
        for sensor in ("sharpa", "xhand"):
            prepare = self.adapter(sensor)
            payload = self.sample(sensor)
            bad = dict(payload, signal_space="normalized")
            with self.subTest(sensor=sensor), self.assertRaises(ValueError):
                prepare(bad)
            with self.assertRaises(ValueError):
                prepare([])
        with self.assertRaises(ValueError):
            self.adapter("xhand")(dict(self.sample("xhand"), layout_id="unknown"))
        with self.assertRaises(ValueError):
            self.adapter("xhand")(dict(self.sample("xhand"), pos_xyz=torch.zeros(5, 120, 3)))
        with self.assertRaises(ValueError):
            self.adapter("sharpa")(dict(self.sample("sharpa"), hand_convention="unknown"))

    def test_cli_parses_arguments_once_for_each_input_format(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for input_format in ("sharpa", "xhand", "normalized"):
                payload = self.sample("xhand" if input_format == "xhand" else "sharpa")
                if input_format == "normalized":
                    raw = payload["raw"]
                    signal = torch.stack((torch.zeros_like(raw), torch.zeros_like(raw), (raw / 255).clamp(0, 1)), dim=-1)
                    payload = {"signal": signal, "pos_xyz": payload["pos_xyz"], "modality": "visuotactile", "signal_space": "normalized", "geometry_space": "canonical_spherical", "hand_convention": "canonical_right", "finger_names": list(FINGER_ORDER)}
                source = root / f"{input_format}_input.pt"
                output = root / f"{input_format}_features.pt"
                torch.save(payload, source)
                calls = []
                parse = argparse.ArgumentParser.parse_args
                def counted(parser, *args, **kwargs):
                    calls.append(parser)
                    return parse(parser, *args, **kwargs)
                with patch.object(argparse.ArgumentParser, "parse_args", counted):
                    main(["--input-format", input_format, "--bundle", str(ROOT / "checkpoints/lingbot-touch-tiny"), "--input", str(source), "--output", str(output), "--device", "cpu"])
                with self.subTest(input_format=input_format):
                    self.assertEqual(len(calls), 1)
                    result = torch.load(output, map_location="cpu", weights_only=True)
                    self.assertEqual(set(result), {"latent", "time_id", "finger_id", "pos_xyz"})
                    self.assertTrue(bool(torch.isfinite(result["latent"]).all()))

    def test_cli_rejects_calibration_for_other_formats(self):
        for input_format in ("xhand", "normalized"):
            with self.subTest(input_format=input_format), self.assertRaises(SystemExit) as error:
                main(["--input-format", input_format, "--bundle", "checkpoints/lingbot-touch-tiny", "--input", "absent.pt", "--output", "unused.pt", "--calibration", "absent.json"])
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
