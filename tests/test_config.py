import tempfile
import unittest
from pathlib import Path

from cmapss_knn.config import load_config
from cmapss_knn.exceptions import ConfigError


class ConfigTests(unittest.TestCase):
    def test_default_config_loads(self):
        cfg = load_config()
        self.assertEqual(cfg.rul_cap, 125)
        self.assertEqual(set(cfg.subsets), {"FD001", "FD002", "FD003", "FD004"})
        self.assertEqual(cfg.subsets["FD001"].n_train_rows, 20631)
        self.assertEqual(cfg.subsets["FD004"].fault_modes, ("HPC", "Fan"))
        self.assertTrue(cfg.paths.raw_data_dir.is_absolute())
        self.assertEqual(len(cfg.archive_sha256), 64)

    def test_phase2_sections(self):
        cfg = load_config()
        self.assertEqual(cfg.features.stats, ("mean", "slope"))
        self.assertEqual(cfg.validation.n_folds, 5)
        self.assertEqual(cfg.grid.sensor_configs, ("A", "B"))
        self.assertIn(75, cfg.grid.window)
        self.assertEqual(max(cfg.grid.k), 200)
        self.assertEqual(cfg.validation.stress_cutoffs, (5, 10, 20))

    def test_missing_file_raises(self):
        with self.assertRaises(ConfigError):
            load_config(Path("does/not/exist.yaml"))

    def test_missing_key_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_dir = Path(tmp) / "configs"
            cfg_dir.mkdir()
            (cfg_dir / "bad.yaml").write_text("project: {seed: 1, log_level: INFO}\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(cfg_dir / "bad.yaml")


if __name__ == "__main__":
    unittest.main()
