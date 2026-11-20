"""Scratch KNN vs scikit-learn on identical data, scaling, metric, k and weighting.

Findings encoded below (see also results/phase2/equivalence_report.md):
* Without ties the two agree: same neighbours, predictions equal to ~1e-12.
* Exact duplicates / zero distances under distance weighting are handled
  identically, except in scikit-learn's brute-force Euclidean path, where the
  dot-product expansion can return a tiny non-zero distance for identical points.
* When equidistant points straddle the k-th position, *which* of them is used is
  implementation-defined. The scratch model always takes the lowest training
  index; scikit-learn may pick another. Its prediction must still be one of the
  valid tie resolutions, which is what we assert.
"""

import itertools
import unittest

import numpy as np

from cmapss_knn.models import ScratchKNNRegressor, SklearnKNNRegressor

METRICS = ("euclidean", "manhattan")
WEIGHTS = ("uniform", "distance")
ALGOS = ("brute", "kd_tree", "ball_tree")


def _pair(k, metric, weights, algo, X, y):
    return ScratchKNNRegressor(k, metric, weights).fit(X, y), SklearnKNNRegressor(k, metric, weights, algo).fit(X, y)


class ContinuousDataEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        cls.X, cls.y = rng.normal(size=(300, 12)), rng.normal(size=300) * 30 + 60
        cls.Q = rng.normal(size=(80, 12))

    def test_predictions_neighbours_and_distances_agree(self):
        for k, metric, weights, algo in itertools.product((1, 2, 3, 5, 10, 25, 100, 300), METRICS, WEIGHTS, ALGOS):
            with self.subTest(k=k, metric=metric, weights=weights, algo=algo):
                a, b = _pair(k, metric, weights, algo, self.X, self.y)
                np.testing.assert_allclose(a.predict(self.Q), b.predict(self.Q), rtol=1e-9, atol=1e-9)
                da, ia = a.kneighbors(self.Q)
                db, ib = b.kneighbors(self.Q)
                np.testing.assert_array_equal(ia, ib)  # no ties in continuous data
                np.testing.assert_allclose(da, db, rtol=1e-7, atol=1e-7)

    def test_k_can_be_overridden_per_call(self):
        a, b = _pair(5, "euclidean", "uniform", "brute", self.X, self.y)
        for k in (1, 7, 50):
            np.testing.assert_array_equal(a.kneighbors(self.Q, k)[1], b.kneighbors(self.Q, k)[1])

    def test_scaled_data_like_the_real_pipeline(self):
        rng = np.random.default_rng(7)
        raw = rng.normal(size=(400, 8)) * np.array([1, 10, 100, 1e3, 1, 1, 5, 0.01]) + np.array([0, 5, -50, 1e4, 0, 3, 2, 0.5])
        scaled = (raw - raw.mean(0)) / raw.std(0)
        y = rng.uniform(0, 125, 400)
        for metric, weights in itertools.product(METRICS, WEIGHTS):
            a, b = _pair(10, metric, weights, "brute", scaled[:300], y[:300])
            np.testing.assert_allclose(a.predict(scaled[300:]), b.predict(scaled[300:]), rtol=1e-9, atol=1e-9)


class DuplicateObservations(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        dup = np.tile([1.0, 2.0, 3.0], (5, 1))  # five identical training points...
        self.X = np.vstack([dup, rng.normal(size=(40, 3)) * 5 + 10])
        self.y = np.concatenate([[10.0, 20.0, 30.0, 40.0, 50.0], rng.uniform(0, 100, 40)])
        self.q = np.array([[1.0, 2.0, 3.0]])  # ...and a query that coincides with them

    def test_scratch_exact_zero_distance_rule(self):
        for metric in METRICS:
            d, idx = ScratchKNNRegressor(5, metric).fit(self.X, self.y).kneighbors(self.q)
            self.assertTrue((d[0] == 0.0).all())  # exact zeros, no rounding noise
            self.assertEqual(sorted(idx[0]), [0, 1, 2, 3, 4])
            for w in WEIGHTS:
                self.assertAlmostEqual(ScratchKNNRegressor(5, metric, w).fit(self.X, self.y).predict(self.q)[0], 30.0)  # mean of the 5 duplicates
            # k > number of duplicates: distance weighting ignores everything but the duplicates
            self.assertAlmostEqual(ScratchKNNRegressor(8, metric, "distance").fit(self.X, self.y).predict(self.q)[0], 30.0)

    def test_agreement_with_sklearn_when_all_duplicates_fit_in_k(self):
        for metric, algo in itertools.product(METRICS, ("kd_tree", "ball_tree")):
            for k in (5, 8):
                for w in WEIGHTS:
                    with self.subTest(metric=metric, algo=algo, k=k, weights=w):
                        a, b = _pair(k, metric, w, algo, self.X, self.y)
                        if w == "uniform" and k == 8:
                            continue  # depends on the 3 non-duplicate neighbours; covered by the continuous test
                        np.testing.assert_allclose(a.predict(self.q), b.predict(self.q), atol=1e-9)

    def test_scratch_breaks_duplicate_ties_by_lowest_index(self):
        # k=3 of five identical points: scratch deterministically takes rows 0,1,2 -> 20
        for metric, w in itertools.product(METRICS, WEIGHTS):
            m = ScratchKNNRegressor(3, metric, w).fit(self.X, self.y)
            self.assertEqual(list(m.kneighbors(self.q)[1][0]), [0, 1, 2])
            self.assertAlmostEqual(m.predict(self.q)[0], 20.0)
            sk = SklearnKNNRegressor(3, metric, w, "kd_tree").fit(self.X, self.y).predict(self.q)[0]
            self.assertTrue(10.0 <= sk <= 50.0)  # a valid choice among the duplicates; may differ from 20

    def test_sklearn_brute_euclidean_duplicate_precision_is_bounded(self):
        # Known numerical difference: sklearn's brute-force Euclidean path may return ~1e-8 instead of 0.
        # It must not matter much: the prediction stays within the duplicates' target range.
        sk = SklearnKNNRegressor(5, "euclidean", "distance", "brute").fit(self.X, self.y).predict(self.q)[0]
        self.assertTrue(10.0 <= sk <= 50.0)


class TiedDistances(unittest.TestCase):
    """Integer lattice data: many exactly equal distances, including at the k-th boundary."""

    def setUp(self):
        gx, gy = np.meshgrid(np.arange(8.0), np.arange(8.0))
        self.X = np.column_stack([gx.ravel(), gy.ravel()])
        self.y = np.random.default_rng(5).uniform(0, 100, len(self.X))
        self.Q = np.array([[3.0, 3.0], [3.5, 3.5], [3.5, 3.0], [0.0, 0.0], [2.0, 5.0]])

    @staticmethod
    def _valid_range(dist_row, y, k, weights):
        """Min and max prediction over all legitimate ways to resolve the tie at position k."""
        order = np.sort(dist_row)
        dk = order[k - 1]
        inside = dist_row < dk
        tied = np.isclose(dist_row, dk, rtol=0, atol=1e-12)
        r = k - int(inside.sum())  # slots left for tied points
        ty = np.sort(y[tied])
        if weights == "uniform":
            base, wt = y[inside].sum(), 1.0
            lo, hi = (base + ty[:r].sum()) / k, (base + ty[-r:].sum()) / k
        else:
            with np.errstate(divide="ignore"):
                w_in, wt = 1.0 / dist_row[inside], 1.0 / dk
            lo = ((w_in * y[inside]).sum() + wt * ty[:r].sum()) / (w_in.sum() + r * wt)
            hi = ((w_in * y[inside]).sum() + wt * ty[-r:].sum()) / (w_in.sum() + r * wt)
        return lo, hi

    def test_tie_handling(self):
        n_boundary_ties = 0
        for k, metric, weights in itertools.product((1, 2, 3, 4, 5, 8, 12), METRICS, ("uniform",)):
            a = ScratchKNNRegressor(k, metric, weights).fit(self.X, self.y)
            for algo in ALGOS:
                b = SklearnKNNRegressor(k, metric, weights, algo).fit(self.X, self.y)
                for q in self.Q:
                    dist = np.abs(self.X - q).sum(1) if metric == "manhattan" else np.sqrt(((self.X - q) ** 2).sum(1))
                    order = np.sort(dist)
                    straddles = k < len(order) and np.isclose(order[k - 1], order[k], atol=1e-12)
                    pa, pb = a.predict([q])[0], b.predict([q])[0]
                    with self.subTest(k=k, metric=metric, algo=algo, q=tuple(q)):
                        lo, hi = self._valid_range(dist, self.y, k, weights)
                        self.assertTrue(lo - 1e-9 <= pa <= hi + 1e-9)  # both are legitimate resolutions
                        self.assertTrue(lo - 1e-9 <= pb <= hi + 1e-9)
                        if not straddles:
                            self.assertAlmostEqual(pa, pb, places=9)  # no ambiguity -> must agree
                        else:
                            n_boundary_ties += 1
                            # scratch rule: among tied points take the lowest training indices
                            ref = np.lexsort((np.arange(len(dist)), dist))[:k]
                            self.assertAlmostEqual(pa, self.y[ref].mean(), places=9)
        self.assertGreater(n_boundary_ties, 0, "the test data must actually contain boundary ties")

    def test_distance_weighting_with_ties_and_exact_hits(self):
        for metric in METRICS:
            a = ScratchKNNRegressor(5, metric, "distance").fit(self.X, self.y)
            b = SklearnKNNRegressor(5, metric, "distance", "kd_tree").fit(self.X, self.y)
            q = np.array([[3.0, 3.0]])  # exactly on a training point
            self.assertAlmostEqual(a.predict(q)[0], self.y[3 * 8 + 3], places=12)
            self.assertAlmostEqual(b.predict(q)[0], self.y[3 * 8 + 3], places=12)


if __name__ == "__main__":
    unittest.main()
