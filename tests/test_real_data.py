"""Integration tests against the real C-MAPSS archive. Skipped if it is not present."""

import unittest

from cmapss_knn.config import load_config
from cmapss_knn.data.ingestion import read_subset, verify_archive
from cmapss_knn.data.targets import add_train_rul, truncated_engine_table
from cmapss_knn.data.validation import validate_subset

CFG = load_config()
ARCHIVE = CFG.paths.raw_data_dir / CFG.archive_name


@unittest.skipUnless(ARCHIVE.is_file(), f"{ARCHIVE} not found; see data/README.md")
class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.subset = read_subset("FD001", CFG.paths.raw_data_dir, CFG.archive_name)

    def test_archive_digest(self):
        self.assertTrue(verify_archive(CFG.paths.raw_data_dir, CFG.archive_name, CFG.archive_sha256).matches_expected)

    def test_fd001_validates(self):
        self.assertTrue(validate_subset(self.subset, CFG.subsets["FD001"], raise_on_error=False).passed)

    def test_all_subsets_validate(self):
        for name, spec in CFG.subsets.items():
            with self.subTest(subset=name):
                subset = read_subset(name, CFG.paths.raw_data_dir, CFG.archive_name)
                self.assertTrue(validate_subset(subset, spec, raise_on_error=False).passed)

    def test_fd001_target_facts(self):
        train = add_train_rul(self.subset.train, cap=CFG.rul_cap)
        self.assertEqual(train["rul"].min(), 0)
        self.assertEqual(train["rul_clipped"].max(), CFG.rul_cap)
        table = truncated_engine_table(self.subset.test, self.subset.rul)
        self.assertEqual(len(table), 100)
        self.assertEqual(int(table["observed_cycles"].min()), 31)


if __name__ == "__main__":
    unittest.main()
