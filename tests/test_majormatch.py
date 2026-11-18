"""Unit and reproducibility tests.  Run:  python -m unittest discover tests"""
import unittest

import numpy as np

from src.baselines.models import MODELS, build
from src.cmn import config
from src.cmn.data import encode, load_survey, make_synthetic, split
from src.majormatch.knn import KNN, jaccard_distance
from src.majormatch.mnb import MultinomialNaiveBayes

X_TOY = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]])
Y_TOY = np.array(["Computer Science", "Computer Science", "Music", "Music", "Mathematics"],
                 dtype=object)


class TestJaccard(unittest.TestCase):
    def test_values(self):
        self.assertEqual(jaccard_distance([1, 1, 0], [1, 1, 0]), 0.0)
        self.assertEqual(jaccard_distance([1, 0, 0], [0, 1, 0]), 1.0)
        self.assertAlmostEqual(jaccard_distance([1, 1, 0], [1, 0, 1]), 1 - 1 / 3)
        self.assertEqual(jaccard_distance([0, 0, 0], [0, 0, 0]), 0.0)

    def test_vectorised_matches_reference(self):
        X, y, _ = load_survey()
        knn = KNN(k=3).fit(X[:50], y[:50])
        d = knn.distances(X[60])
        ref = [jaccard_distance(X[60], row) for row in X[:50]]
        np.testing.assert_allclose(d, ref)


class TestToyExample(unittest.TestCase):
    """The worked example of the Milestone 3 report: same input -> same output."""

    def test_knn(self):
        self.assertEqual(KNN(k=3).fit(X_TOY, Y_TOY).rank([[1, 1, 0, 0]], n=2)[0],
                         ["Computer Science", "Mathematics"])

    def test_mnb(self):
        mnb = MultinomialNaiveBayes().fit(X_TOY, Y_TOY)
        self.assertEqual(mnb.rank([[1, 1, 0, 0]], n=2)[0], ["Computer Science", "Mathematics"])
        self.assertEqual(mnb.rank([[0, 0, 1, 1]], n=1)[0], ["Music"])

    def test_match_percent_sums_to_100(self):
        for model in (KNN(k=3).fit(X_TOY, Y_TOY), MultinomialNaiveBayes().fit(X_TOY, Y_TOY)):
            self.assertAlmostEqual(sum(model.match_percent([1, 1, 0, 0]).values()), 100.0)


class TestProtocol(unittest.TestCase):
    def test_split_is_stratified_and_fixed(self):
        X, y, ids = load_survey()
        a = split(X, y, ids, seed=2026)
        b = split(X, y, ids, seed=2026)
        np.testing.assert_array_equal(a[5], b[5])                 # same test ids
        self.assertEqual(len(a[5]), int(len(y) * config.TEST_SIZE))
        share_all = np.mean(y == "Physics")
        share_test = np.mean(a[3] == "Physics")
        self.assertAlmostEqual(share_all, share_test, delta=0.01)

    def test_benchmark_is_reproducible_and_eta1_matches_survey(self):
        a, b = make_synthetic(0.5, seed=7), make_synthetic(0.5, seed=7)
        np.testing.assert_array_equal(a[0], b[0])
        X1, _, _ = make_synthetic(1.0, n=20000, seed=1)
        Xs, _, _ = load_survey()
        np.testing.assert_allclose(X1.mean(axis=0), Xs.mean(axis=0), atol=0.02)

    def test_every_model_runs_and_ranks(self):
        X, y, ids = make_synthetic(0.0, n=200, seed=3)
        X_dev, X_te, y_dev, _, _, _ = split(X, y, ids, seed=3)
        for name, (_, grid, _) in MODELS.items():
            model = build(name, 3, **grid[0]).fit(X_dev, y_dev)
            ranked = model.rank(X_te[:5], n=2)
            self.assertEqual(len(ranked), 5)
            proba = model.predict_proba(X_te[:5])
            np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_encode(self):
        self.assertEqual(int(encode(["Programming", "Coding Club"]).sum()), 2)
        with self.assertRaises(ValueError):
            encode(["Swimming"])


if __name__ == "__main__":
    unittest.main()
