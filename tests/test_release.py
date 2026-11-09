import hashlib
import json
from pathlib import Path
import re
import tomllib
import unittest
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_manifest_contains_only_released_sample_accounting(self):
        manifest = json.loads((ROOT / "sample_data/manifest.json").read_text())
        self.assertEqual(set(manifest), {"schema", "frames_per_clip", "pairs_per_platform", "single_hand_files_per_platform", "samples"})
        allowed = {"platform", "pair_id", "hand", "path", "sha256", "raw_shape", "raw_dtype", "modality"}
        for sample in manifest["samples"]:
            self.assertEqual(set(sample), allowed)
        self.assertFalse("candidate_recordings" in json.dumps(manifest), "Remove unreleased candidate recording totals")
        self.assertEqual(manifest["pairs_per_platform"], 30)
        self.assertEqual(manifest["single_hand_files_per_platform"], 60)

    def test_all_sample_hashes_and_pairs(self):
        root = ROOT / "sample_data"
        manifest = json.loads((root / "manifest.json").read_text())
        seen = set()
        for item in manifest["samples"]:
            relative = Path(item["path"])
            self.assertFalse(relative.is_absolute())
            path = (root / relative).resolve()
            self.assertTrue(path.is_relative_to(root.resolve()))
            self.assertNotIn(relative.as_posix(), seen)
            seen.add(relative.as_posix())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])
        expected = {
            f"{sensor}/{index:03d}_{hand}.pt"
            for sensor in ("sharpa", "xhand")
            for index in range(1, 31)
            for hand in ("left", "right")
        }
        self.assertEqual(seen, expected)
        self.assertEqual(seen, {p.relative_to(root).as_posix() for p in root.rglob("*.pt")})

    def test_readme_uses_public_model_names(self):
        text = (ROOT / "README.md").read_text()
        self.assertFalse("HaT-" in text, "Use LingBot-Touch model names consistently")
        for size in ("Tiny", "Small"):
            self.assertIn(f"LingBot-Touch-{size}", text)

    def test_documentation_links_resolve(self):
        for name in ("README.md", "DATA_INTERFACE_GUIDE.md"):
            text = (ROOT / name).read_text()
            links = re.findall(r"\]\(([^\s)]+)\)", text)
            links += re.findall(r'(?:src|href)="([^"]+)"', text)
            for link in links:
                parsed = urlsplit(link)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                with self.subTest(document=name, link=link):
                    self.assertTrue((ROOT / unquote(parsed.path)).exists())

    def test_ignore_rules_keep_published_assets_visible(self):
        rules = (ROOT / ".gitignore").read_text().splitlines()
        for rule in ("__pycache__/", "*.py[cod]", "/outputs/", "/build/", "/dist/", "*.egg-info/", ".env"):
            self.assertIn(rule, rules)
        self.assertNotIn("*.pt", rules)
        self.assertIn("/checkpoints/", rules)

    def test_only_supported_release_files_remain(self):
        for relative in ("LEGAL.md", "examples", "tactile_jepa_inference/data", "tactile_jepa_inference/geometry.py", "tactile_jepa_inference/interfaces.py", "tactile_jepa_inference/xhand.py"):
            with self.subTest(path=relative):
                self.assertFalse((ROOT / relative).exists())
        for relative in ("LICENSE", "THIRD_PARTY_NOTICES.md", "data_interfaces/data/xhand_to_magnetic96_v1.json"):
            self.assertTrue((ROOT / relative).is_file())

    def test_runtime_dependencies_include_verified_numpy(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(config["project"]["dependencies"], ["torch==2.11.0", "numpy==2.4.4"])
        self.assertNotIn("tactile_jepa_inference", config["tool"]["setuptools"]["package-data"])

    def test_readme_batch_readout_and_model_table(self):
        text = (ROOT / "README.md").read_text()
        section = text.split("### Run All Samples and Read the Results", 1)[1].split("### Run the Tests", 1)[0]
        self.assertIn('f"outputs/batch_{sensor}/001_left_features.pt"', section)
        self.assertNotIn('torch.load("outputs/sharpa/', section)
        self.assertNotIn("#Parameters", text)
        self.assertNotRegex(text, r"5\.46\s*M|21\.76\s*M|5,455,488|21,755,904")


if __name__ == "__main__":
    unittest.main()
