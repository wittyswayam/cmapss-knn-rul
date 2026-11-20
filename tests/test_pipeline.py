import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from cmapss_knn.data.schema import SENSOR_COLUMNS
from cmapss_knn.data.targets import add_train_rul
from cmapss_knn.exceptions import ModelNotFittedError
from cmapss_knn.features import SensorSelector, WindowConfig, build_features, feature_column_indices
from cmapss_knn.models.knn_sklearn import SklearnKNNRegressor
from cmapss_knn.models.pipeline import PipelineConfig, RULKNNPipeline

from .helpers import make_subset


def _train(seed=0, n=12):
    return add_train_rul(make_subset(n_train=n, seed=seed).train, cap=40)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.train = _train()
        self.cfg = PipelineConfig(WindowConfig(10, ("mean", "slope"), "adaptive", "causal", 5), "A", 5, "euclidean", "distance", "sklearn")

    def test_matches_manual_composition_of_the_primitives(self):
        model = RULKNNPipeline(self.cfg).fit(self.train)
        sel = SensorSelector("A").fit(self.train)
        table = build_features(self.train, self.cfg.window)
        cols = feature_column_indices(table.columns, sel.selected_sensors_)
        scaler = StandardScaler().fit(table.X[:, cols])
        knn = SklearnKNNRegressor(5, "euclidean", "distance", "brute").fit(scaler.transform(table.X[:, cols]), self.train["rul_clipped"])
        np.testing.assert_allclose(model.predict_rows(self.train), knn.predict(scaler.transform(table.X[:, cols])), atol=1e-9)
        self.assertEqual(model.feature_names_, [table.columns[i] for i in cols])

    def test_backends_agree(self):
        test = _train(seed=1, n=4)
        a = RULKNNPipeline(self.cfg).fit(self.train).predict_rows(test)
        from dataclasses import replace
        b = RULKNNPipeline(replace(self.cfg, backend="scratch")).fit(self.train).predict_rows(test)
        np.testing.assert_allclose(a, b, atol=1e-9)

    def test_prediction_uses_only_history_up_to_each_row(self):
        model = RULKNNPipeline(self.cfg).fit(self.train)
        test = _train(seed=2, n=3)
        full = model.predict_rows(test)
        cut = test[test.cycle <= 25].reset_index(drop=True)
        np.testing.assert_allclose(model.predict_rows(cut), full[(test.cycle <= 25).to_numpy()], atol=1e-12)

    def test_engines_do_not_influence_each_other(self):
        model = RULKNNPipeline(self.cfg).fit(self.train)
        test = _train(seed=3, n=3)
        alone = model.predict_rows(test[test.unit == 2].reset_index(drop=True))
        together = model.predict_rows(test)[(test.unit == 2).to_numpy()]
        np.testing.assert_allclose(alone, together, atol=1e-12)

    def test_predict_last_cycle_one_row_per_engine(self):
        model = RULKNNPipeline(self.cfg).fit(self.train)
        test = _train(seed=4, n=4)
        last = model.predict_last_cycle(test)
        self.assertEqual(list(last.index), [1, 2, 3, 4])
        rows = model.predict_rows(test)
        for unit in (1, 4):
            self.assertAlmostEqual(last[unit], rows[(test.unit == unit).to_numpy()][-1])

    def test_scaler_and_sensors_come_from_training_data_only(self):
        model = RULKNNPipeline(self.cfg).fit(self.train)
        shifted = self.train.copy()
        shifted[list(SENSOR_COLUMNS)] += 1000.0  # new data far outside the training range
        pred = model.predict_rows(shifted)  # must not refit anything, just extrapolate
        self.assertTrue(np.isfinite(pred).all())
        self.assertTrue(((pred >= -1e-9) & (pred <= 40 + 1e-9)).all())  # a weighted mean stays inside the target range (up to rounding)

    def test_regime_normalisation_option(self):
        from cmapss_knn.features.regimes import RegimeNormalizer
        from dataclasses import replace
        from .test_regimes import _multi_regime
        df, _ = _multi_regime(n_units=8, T=70)
        df = add_train_rul(df, cap=40)
        train, test = df[df.unit <= 6].reset_index(drop=True), df[df.unit > 6].reset_index(drop=True)
        cfg = replace(self.cfg, normalization="regime")
        m1 = RULKNNPipeline(cfg).fit(train)
        norm = RegimeNormalizer(seed=cfg.seed).fit(train)
        m2 = RULKNNPipeline(cfg).fit(train, normalizer=norm)  # pre-fitted normaliser gives the same model
        np.testing.assert_allclose(m1.predict_rows(test), m2.predict_rows(test), atol=1e-12)
        manual = RULKNNPipeline(replace(self.cfg, normalization="global")).fit(norm.transform(train).assign(rul_clipped=train["rul_clipped"]))
        np.testing.assert_allclose(m1.predict_rows(test), manual.predict_rows(norm.transform(test)), atol=1e-9)
        self.assertEqual(m1.normalizer_.n_regimes_, 3)
        with self.assertRaises(ValueError):
            RULKNNPipeline(replace(self.cfg, normalization="regimes")).fit(train)

    def test_not_fitted_and_bad_backend(self):
        with self.assertRaises(ModelNotFittedError):
            RULKNNPipeline(self.cfg).predict_rows(self.train)
        from dataclasses import replace
        with self.assertRaises(ValueError):
            RULKNNPipeline(replace(self.cfg, backend="faiss")).fit(self.train)


if __name__ == "__main__":
    unittest.main()
