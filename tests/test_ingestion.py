import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cmapss_knn.data.ingestion import read_subset, sha256_of, verify_archive
from cmapss_knn.exceptions import DataFormatError, DataNotFoundError

from .helpers import make_subset, write_raw_files


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.expected = make_subset()

    def tearDown(self):
        self._tmp.cleanup()

    def _assert_roundtrip(self, loaded):
        self.assertEqual(loaded.train.shape, self.expected.train.shape)
        self.assertEqual(loaded.test.shape, self.expected.test.shape)
        self.assertEqual(list(loaded.train.columns), list(self.expected.train.columns))
        self.assertEqual(str(loaded.train["unit"].dtype), "int64")
        self.assertEqual(str(loaded.train["cycle"].dtype), "int64")
        pd.testing.assert_frame_equal(loaded.train, self.expected.train, atol=1e-4, rtol=0)
        self.assertEqual(list(loaded.rul.index), list(self.expected.rul.index))
        self.assertEqual(list(loaded.rul), list(self.expected.rul))

    def test_reads_extracted_files(self):
        write_raw_files(self.dir, self.expected)
        self._assert_roundtrip(read_subset("FD001", self.dir))

    def test_reads_from_zip_archive(self):
        write_raw_files(self.dir, self.expected, as_zip=True)
        self._assert_roundtrip(read_subset("FD001", self.dir))

    def test_missing_data_raises_not_found(self):
        with self.assertRaises(DataNotFoundError):
            read_subset("FD001", self.dir)

    def test_unknown_subset_raises(self):
        with self.assertRaises(ValueError):
            read_subset("FD999", self.dir)

    def test_wrong_column_count_raises_format_error(self):
        write_raw_files(self.dir, self.expected)
        (self.dir / "train_FD001.txt").write_text("1 1 0.1 0.2\n1 2 0.1 0.2\n", encoding="utf-8")
        with self.assertRaises(DataFormatError):
            read_subset("FD001", self.dir)

    def test_corrupt_zip_raises_format_error(self):
        (self.dir / "CMAPSSData.zip").write_bytes(b"this is not a zip file")
        with self.assertRaises(DataFormatError):
            read_subset("FD001", self.dir)

    def test_archive_hash_check(self):
        self.assertIsNone(verify_archive(self.dir, "CMAPSSData.zip", "0" * 64))
        write_raw_files(self.dir, self.expected, as_zip=True)
        digest = sha256_of(self.dir / "CMAPSSData.zip")
        self.assertTrue(verify_archive(self.dir, "CMAPSSData.zip", digest).matches_expected)
        self.assertFalse(verify_archive(self.dir, "CMAPSSData.zip", "0" * 64).matches_expected)


if __name__ == "__main__":
    unittest.main()
