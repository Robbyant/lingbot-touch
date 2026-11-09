from dataclasses import fields
import inspect
import json
from pathlib import Path
import importlib
import unittest

import torch

from data_interfaces import FINGER_ORDER, TactileInput, prepare_sensor_input
from data_interfaces.pretraining_sampling import prepare_pretraining_reference_input
from data_interfaces.sampling import FixedSpatialSampler, load_reference_geometry
from tactile_jepa_inference import EncoderConfig, TargetEncoder


ROOT = Path(__file__).resolve().parents[1]
MODALITIES = {"magnetic", "piezoresistive", "visuotactile"}


class ModalityContractTests(unittest.TestCase):
    def test_bundles_expose_only_three_identity_routes(self):
        for size in ("tiny", "small"):
            with self.subTest(size=size):
                root = ROOT / "checkpoints" / f"lingbot-touch-{size}"
                config = json.loads((root / "config.json").read_text())
                self.assertEqual(config["group_by_source"], {name: name for name in MODALITIES})
                state = torch.load(root / "encoder.pt", map_location="cpu", weights_only=True)
                groups = {key.split(".")[2] for key in state if key.startswith("tokenizer.tokenizers.")}
                self.assertEqual(groups, MODALITIES)
                self.assertTrue(all(isinstance(t, torch.Tensor) and t.dtype == torch.float32 and bool(torch.isfinite(t).all()) for t in state.values()))

    def test_input_and_encoder_use_modality_directly(self):
        self.assertEqual({f.name for f in fields(TactileInput)}, {"signal", "pos_xyz", "coverage_mask", "modality"})
        for method in (TargetEncoder.encode, TargetEncoder.forward):
            self.assertEqual(set(inspect.signature(method).parameters), {"self", "signal", "pos_xyz", "coverage_mask", "modality"})

    def test_configuration_rejects_missing_extra_or_swapped_routes(self):
        config = json.loads((ROOT / "checkpoints/lingbot-touch-tiny/config.json").read_text())
        routes = {name: name for name in MODALITIES}
        invalid = [dict(routes, unknown="magnetic"), {k: v for k, v in routes.items() if k != "visuotactile"}, dict(routes, visuotactile="magnetic")]
        for mapping in invalid:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                EncoderConfig.from_dict(dict(config, group_by_source=mapping))

    def test_all_native_maps_and_fixed_sampling_preserve_modality(self):
        torch.set_num_threads(4)
        for modality, shape, counts, layout in (
            ("magnetic", (1, 4, 5, 116, 3), (96, 76, 64, 48, 32, 24), "magnetic_native116_96_canonical_right_v1"),
            ("visuotactile", (1, 4, 5, 240, 240), (200, 96, 32), "visuotactile_deformation240_canonical_right_v1"),
        ):
            for count in counts:
                with self.subTest(modality=modality, count=count):
                    batch = prepare_pretraining_reference_input(torch.zeros(shape), layout_id=layout, num_points=count)
                    self.assertEqual(batch.modality, modality)
                    self.assertEqual(batch.to("cpu").modality, modality)
                    self.assertEqual(batch.signal.shape, (1, 4, 5, count, 3))
                    self.assertTrue(bool(batch.coverage_mask.all()))
                    self.assertEqual(torch.count_nonzero(batch.signal), 0)
        reference = load_reference_geometry("magnetic", 24)
        indices = torch.arange(24).reshape(1, 24, 1).expand(5, -1, -1).clone()
        sampler = FixedSpatialSampler(reference["pos_xyz"], reference["coverage_mask"], indices, torch.ones(5, 24, 1), modality="magnetic", num_points=24)
        batch = prepare_sensor_input(torch.full((1, 4, 5, 24, 3), 0.25), reference["pos_xyz"], modality="magnetic", finger_names=FINGER_ORDER, measurement="xyz", scales_xyz=[1., 1., 1.])
        result = sampler(batch)
        self.assertEqual(result.modality, "magnetic")
        for field in ("signal", "pos_xyz", "coverage_mask"):
            self.assertTrue(torch.equal(getattr(result, field), getattr(batch, field)))

    def test_supported_entrypoints_import(self):
        for name, entry in (("data_interfaces.infer", "main"), ("data_interfaces.infer_sharpa", "prepare_sharpa"), ("data_interfaces.infer_xhand", "prepare_xhand")):
            with self.subTest(module=name):
                module = importlib.import_module(name)
                self.assertTrue(callable(getattr(module, entry, None)))

    def test_piezoresistive_references_preserve_geometry_padding_and_scalar_values(self):
        for count, points, valid_counts in ((61, 124, [124, 61, 61, 61, 61]), (16, 16, [16] * 5), (9, 9, [9] * 5)):
            with self.subTest(count=count):
                reference = load_reference_geometry("piezoresistive", count)
                self.assertEqual(reference["pos_xyz"].shape, (5, points, 3))
                self.assertEqual(reference["coverage_mask"].sum(1).tolist(), valid_counts)
                self.assertEqual(reference["taxel_part"].shape, (5, points))
                indices = torch.arange(points).reshape(1, points, 1).expand(5, -1, -1).clone()
                indices[~reference["coverage_mask"]] = 0
                sampler = FixedSpatialSampler(reference["pos_xyz"], reference["coverage_mask"], indices, torch.ones(5, points, 1), modality="piezoresistive", num_points=count)
                batch = prepare_sensor_input(torch.full((1, 4, 5, points), 0.25), reference["pos_xyz"], reference["coverage_mask"], modality="piezoresistive", finger_names=FINGER_ORDER, measurement="z", scales_xyz=[1., 1., 1.])
                result = sampler(batch)
                self.assertEqual(result.modality, "piezoresistive")
                for field in ("signal", "pos_xyz", "coverage_mask"):
                    self.assertTrue(torch.equal(getattr(result, field), getattr(batch, field)))
                batch.signal[..., 0] = 0.1
                with self.assertRaisesRegex(ValueError, "XY must be zero"):
                    sampler(batch)


if __name__ == "__main__":
    unittest.main()
