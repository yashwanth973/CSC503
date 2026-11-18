"""k-Nearest-Neighbour major recommender with Jaccard distance (proposed method 1).

  d_J(x, x') = 1 - |x AND x'| / |x OR x'|        (0 when both vectors are empty)
  N_k(x)     = k training students closest to x   (stable sort -> deterministic ties)
  v(m)       = #neighbours in N_k(x) with major m
  delta(m)   = distance of the closest neighbour with major m
  ranking    = majors with v(m) > 0 sorted by (-v(m), delta(m))
  p_{u,m}    = 100 * v(m) / k                      (match value of Milestone 1)
"""
import numpy as np


def jaccard_distance(o1, o2):
    """Jaccard distance between two binary vectors (reference implementation)."""
    intersection = sum(1 for a, b in zip(o1, o2) if a == 1 and b == 1)
    union = sum(1 for a, b in zip(o1, o2) if a == 1 or b == 1)
    if union == 0:
        return 0.0
    return 1 - intersection / union


class KNN:
    """Lazy learner: `fit` stores the training students, `rank` recommends majors."""

    name = "kNN-Jaccard"

    def __init__(self, k=7):
        if k <= 0:
            raise ValueError("The provided value k must be positive")
        self.k = k

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=int)
        self.y = np.asarray(y, dtype=object)
        self.classes_ = np.array(sorted(set(self.y)), dtype=object)
        self._row_sum = self.X.sum(axis=1)
        return self

    def distances(self, x):
        """Jaccard distance from x to every training student (vectorised Eq. 2)."""
        x = np.asarray(x, dtype=int)
        inter = self.X @ x
        union = self._row_sum + x.sum() - inter
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.where(union == 0, 0.0, 1.0 - inter / union)
        return d

    def _neighbours(self, x):
        d = self.distances(x)
        idx = np.argsort(d, kind="stable")[: min(self.k, len(d))]
        return d[idx], self.y[idx]

    def rank_one(self, x, n=2):
        dists, majors = self._neighbours(x)
        votes, closest = {}, {}
        for dist, major in zip(dists, majors):
            votes[major] = votes.get(major, 0) + 1
            closest.setdefault(major, dist)       # first seen = closest (sorted)
        ranked = sorted(votes, key=lambda m: (-votes[m], closest[m]))
        return ranked[:n]

    def rank(self, X, n=2):
        return [self.rank_one(x, n) for x in np.atleast_2d(X)]

    def predict_proba(self, X):
        """Neighbour share of each major (rows sum to 1); columns follow classes_."""
        out = np.zeros((len(np.atleast_2d(X)), len(self.classes_)))
        col = {c: i for i, c in enumerate(self.classes_)}
        for r, x in enumerate(np.atleast_2d(X)):
            _, majors = self._neighbours(x)
            for m in majors:
                out[r, col[m]] += 1
        return out / out.sum(axis=1, keepdims=True)

    def match_percent(self, x):
        """p_{u,m} of the proposal for one student, in [0, 100]."""
        proba = self.predict_proba([x])[0]
        return {m: 100 * p for m, p in zip(self.classes_, proba) if p > 0}
