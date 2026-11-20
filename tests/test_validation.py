import unittest
from dataclasses import replace

import pandas as pd

from cmapss_knn.data.ingestion import CMAPSSSubset
from cmapss_knn.data.validation import validate_subset
from cmapss_knn.exceptions import DataValidationError

from .helpers import make_subset, spec_for


def _failed(report):
    return {c.name for c in report.failures}


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.subset = make_subset()
        self.spec = spec_for(self.subset)

    def _with_train(self, train):
        return replace(self.subset, train=train)

    def test_clean_data_passes(self):
        report = validate_subset(self.subset, self.spec)
        self.assertTrue(report.passed)
        self.assertIn("train: constant columns", {c.name for c in report.checks})

    def test_missing_value_detected(self):
        train = self.subset.train.copy()
        train.loc[3, "s2"] = float("nan")
        report = validate_subset(self._with_train(train), self.spec, raise_on_error=False)
        self.assertIn("train: no missing values", _failed(report))

    def test_cycle_gap_detected(self):
        train = self.subset.train.drop(index=5).reset_index(drop=True)
        report = validate_subset(self._with_train(train), replace(self.spec, n_train_rows=len(train)), raise_on_error=False)
        self.assertIn("train: cycles start at 1 and increase by 1", _failed(report))

    def test_duplicate_row_detected(self):
        train = pd.concat([self.subset.train, self.subset.train.iloc[[0]]], ignore_index=True)
        report = validate_subset(self._with_train(train), self.spec, raise_on_error=False)
        self.assertIn("train: no duplicate (unit, cycle)", _failed(report))

    def test_unsorted_rows_detected(self):
        train = self.subset.train.iloc[::-1].reset_index(drop=True)
        report = validate_subset(self._with_train(train), self.spec, raise_on_error=False)
        self.assertIn("train: rows sorted by (unit, cycle)", _failed(report))

    def test_noncontiguous_units_detected(self):
        train = self.subset.train.copy()
        train.loc[train["unit"] == 3, "unit"] = 99
        report = validate_subset(self._with_train(train), self.spec, raise_on_error=False)
        self.assertIn("train: unit ids are 1..N", _failed(report))

    def test_rul_length_mismatch_detected(self):
        bad = replace(self.subset, rul=self.subset.rul.iloc[:-1])
        report = validate_subset(bad, self.spec, raise_on_error=False)
        self.assertIn("RUL file: one value per test engine", _failed(report))

    def test_negative_rul_detected(self):
        rul = self.subset.rul.copy()
        rul.iloc[0] = -1
        report = validate_subset(replace(self.subset, rul=rul), self.spec, raise_on_error=False)
        self.assertIn("RUL file: non-negative whole numbers", _failed(report))

    def test_spec_mismatch_detected(self):
        report = validate_subset(self.subset, replace(self.spec, n_train_units=self.spec.n_train_units + 1), raise_on_error=False)
        self.assertIn("train engines match spec", _failed(report))

    def test_wrong_schema_short_circuits(self):
        train = self.subset.train.rename(columns={"s1": "oops"})
        report = validate_subset(self._with_train(train), self.spec, raise_on_error=False)
        self.assertIn("train: column schema", _failed(report))

    def test_raises_by_default_on_failure(self):
        train = self.subset.train.copy()
        train.loc[0, "s2"] = float("nan")
        with self.assertRaises(DataValidationError):
            validate_subset(self._with_train(train), self.spec)


if __name__ == "__main__":
    unittest.main()
