import unittest

import numpy as np

from cmapss_knn.exceptions import ModelNotFittedError
from cmapss_knn.models import ScratchKNNRegressor, SklearnKNNRegressor, aggregate_neighbours
from cmapss_knn.models.knn_scratch import pairwise_distances

BOTH = (ScratchKNNRegressor, SklearnKNNRegressor)


class HandComputedTests(unittest.TestCase):
    """A 1-D example small enough to verify by hand."""

    X = np.array([[0.0], [1.0], [3.0], [6.0]])
    y = np.array([10.0, 20.0, 30.0, 40.0])

    def test_uniform_prediction(self):
        for cls in BOTH:
            with self.subTest(cls=cls.__name__):
                # query 2.0: distances 2,1,1,4 -> two nearest are x=1 (y=20) and x=3 (y=30)
                self.assertAlmostEqual(cls(2, "euclidean", "uniform").fit(self.X, self.y).predict([[2.0]])[0], 25.0)
                self.assertAlmostEqual(cls(1, "euclidean", "uniform").fit(self.X, self.y).predict([[0.4]])[0], 10.0)

    def test_distance_weighting(self):
        for cls in BOTH:
            with self.subTest(cls=cls.__name__):
                # query 0.5: neighbours x=0 (d=.5,y=10) and x=1 (d=.5,y=20) -> equal weights -> 15
                self.assertAlmostEqual(cls(2, "euclidean", "distance").fit(self.X, self.y).predict([[0.5]])[0], 15.0)
                # query 0.25, k=2: d=.25 (y=10), d=.75 (y=20): (10/.25 + 20/.75) / (1/.25 + 1/.75)
                expected = (10 / 0.25 + 20 / 0.75) / (1 / 0.25 + 1 / 0.75)
                self.assertAlmostEqual(cls(2, "euclidean", "distance").fit(self.X, self.y).predict([[0.25]])[0], expected)

    def test_euclidean_vs_manhattan_pick_different_neighbours(self):
        X = np.array([[3.0, 3.0], [4.5, 0.0]])  # from the origin: euclid 4.24 vs 4.5, manhattan 6 vs 4.5
        y = np.array([1.0, 2.0])
        for cls in BOTH:
            self.assertEqual(cls(1, "euclidean").fit(X, y).predict([[0.0, 0.0]])[0], 1.0)
            self.assertEqual(cls(1, "manhattan").fit(X, y).predict([[0.0, 0.0]])[0], 2.0)

    def test_pairwise_distances(self):
        a, b = np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]])
        self.assertAlmostEqual(pairwise_distances(a, b, "euclidean")[0, 0], 5.0)
        self.assertAlmostEqual(pairwise_distances(a, b, "manhattan")[0, 0], 7.0)

    def test_aggregate_zero_distance_rule(self):
        d = np.array([[0.0, 0.0, 2.0]])
        t = np.array([[10.0, 30.0, 100.0]])
        self.assertAlmostEqual(aggregate_neighbours(d, t, "distance")[0], 20.0)  # only the coinciding points count
        self.assertAlmostEqual(aggregate_neighbours(d, t, "uniform")[0], 140 / 3)
        with self.assertRaises(ValueError):
            aggregate_neighbours(d, t, "gaussian")


class KneighborsTests(unittest.TestCase):
    def test_sorted_nearest_first_and_consistent(self):
        rng = np.random.default_rng(0)
        X, Q = rng.normal(size=(60, 4)), rng.normal(size=(7, 4))
        y = rng.normal(size=60)
        for cls in BOTH:
            d, idx = cls(5).fit(X, y).kneighbors(Q, 8)
            self.assertEqual(d.shape, (7, 8))
            self.assertTrue((np.diff(d, axis=1) >= 0).all())
            true = np.sqrt(((Q[:, None, :] - X[idx]) ** 2).sum(-1))
            np.testing.assert_allclose(d, true, atol=1e-9)

    def test_chunk_size_does_not_change_result(self):
        rng = np.random.default_rng(1)
        X, Q, y = rng.normal(size=(80, 3)), rng.normal(size=(25, 3)), rng.normal(size=80)
        ref = ScratchKNNRegressor(6, "manhattan", "distance", chunk_size=1000).fit(X, y).predict(Q)
        for chunk in (1, 7, 25):
            np.testing.assert_array_equal(ScratchKNNRegressor(6, "manhattan", "distance", chunk_size=chunk).fit(X, y).predict(Q), ref)


class EdgeCaseTests(unittest.TestCase):
    def test_k_equals_n_train_gives_global_mean(self):
        rng = np.random.default_rng(2)
        X, y = rng.normal(size=(9, 2)), rng.normal(size=9)
        for cls in BOTH:
            pred = cls(9, "euclidean", "uniform").fit(X, y).predict(rng.normal(size=(3, 2)))
            np.testing.assert_allclose(pred, y.mean())

    def test_single_training_sample(self):
        for cls in BOTH:
            self.assertEqual(cls(1).fit([[1.0, 2.0]], [7.0]).predict([[5.0, 5.0]])[0], 7.0)

    def test_single_feature_and_single_query(self):
        for cls in BOTH:
            self.assertEqual(cls(1).fit([[1.0], [5.0]], [1.0, 5.0]).predict([[4.0]]).shape, (1,))

    def test_constant_features(self):
        # Every distance is 0, so all training points tie. The scratch model deterministically
        # uses the first k rows; scikit-learn may pick any k rows (so only the range is checked).
        X, y = np.ones((10, 3)), np.arange(10.0)
        q = np.ones((2, 3))
        np.testing.assert_allclose(ScratchKNNRegressor(4).fit(X, y).predict(q), np.full(2, y[:4].mean()))
        sk = SklearnKNNRegressor(4).fit(X, y).predict(q)
        self.assertTrue(((sk >= 0.0) & (sk <= 9.0)).all())

    def test_accepts_lists_and_int_arrays(self):
        for cls in BOTH:
            self.assertEqual(cls(1).fit([[0, 0], [10, 10]], [1, 2]).predict([[1, 1]])[0], 1.0)

    def test_fit_does_not_alias_caller_arrays(self):
        X, y = np.array([[0.0], [1.0]]), np.array([1.0, 2.0])
        for cls in BOTH:
            m = cls(1).fit(X, y)
            X[0, 0], y[0] = 100.0, 100.0
            self.assertEqual(m.predict([[0.1]])[0], 1.0)
            X[0, 0], y[0] = 0.0, 1.0


class InvalidInputTests(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(20.0).reshape(10, 2)
        self.y = np.arange(10.0)

    def test_bad_constructor_arguments(self):
        for cls in BOTH:
            for kw in ({"n_neighbors": 0}, {"n_neighbors": -3}, {"n_neighbors": 2.5}, {"n_neighbors": "5"}, {"n_neighbors": True},
                       {"metric": "cosine"}, {"metric": None}, {"weights": "gaussian"}, {"weights": None}):
                with self.subTest(cls=cls.__name__, kw=kw), self.assertRaises(ValueError):
                    cls(**kw)
        with self.assertRaises(ValueError):
            SklearnKNNRegressor(algorithm="octree")
        with self.assertRaises(ValueError):
            ScratchKNNRegressor(chunk_size=0)

    def test_bad_training_data(self):
        bad_X = {"nan": np.array([[1.0, np.nan]] * 10), "inf": np.array([[1.0, np.inf]] * 10), "1d": np.arange(10.0),
                 "3d": np.zeros((2, 2, 2)), "empty": np.zeros((0, 2)), "no features": np.zeros((5, 0)), "strings": np.array([["a", "b"]] * 10)}
        for cls in BOTH:
            for name, X in bad_X.items():
                with self.subTest(cls=cls.__name__, X=name), self.assertRaises(ValueError):
                    cls(2).fit(X, np.arange(len(X)) if np.ndim(X) else self.y)
            with self.assertRaises(ValueError):
                cls(2).fit(self.X, self.y[:-1])  # length mismatch
            with self.assertRaises(ValueError):
                cls(2).fit(self.X, np.where(self.y == 3, np.nan, self.y))
            with self.assertRaises(ValueError):
                cls(2).fit(self.X, self.y.reshape(-1, 1))  # y must be 1-D
            with self.assertRaises(ValueError):
                cls(11).fit(self.X, self.y)  # k > n_train

    def test_bad_queries(self):
        for cls in BOTH:
            m = cls(3).fit(self.X, self.y)
            for name, Q in {"wrong width": np.zeros((2, 3)), "nan": np.array([[np.nan, 1.0]]), "1d": np.zeros(2), "empty": np.zeros((0, 2))}.items():
                with self.subTest(cls=cls.__name__, Q=name), self.assertRaises(ValueError):
                    m.predict(Q)
            with self.assertRaises(ValueError):
                m.kneighbors(self.X, n_neighbors=11)
            with self.assertRaises(ValueError):
                m.kneighbors(self.X, n_neighbors=0)

    def test_not_fitted(self):
        for cls in BOTH:
            with self.assertRaises(ModelNotFittedError):
                cls(3).predict(self.X)
            with self.assertRaises(ModelNotFittedError):
                cls(3).kneighbors(self.X)


if __name__ == "__main__":
    unittest.main()
