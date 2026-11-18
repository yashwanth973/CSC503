"""Multinomial Naive Bayes major recommender with Laplace smoothing (proposed method 2).

  pi_m        = N_m / N
  theta_{m,j} = (c_{m,j} + alpha) / (sum_j' c_{m,j'} + alpha |F|)
  score(m|x)  = log pi_m + sum_j x_j log theta_{m,j}
Majors are ranked by score (ties by name); p_{u,m} is the softmax of the scores.
"""
import numpy as np


class MultinomialNaiveBayes:

    name = "MNB"

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=object)
        self.classes_ = np.array(sorted(set(y)), dtype=object)
        counts = np.array([X[y == m].sum(axis=0) for m in self.classes_])      # c_{m,j}
        self.log_prior = np.log(np.array([np.mean(y == m) for m in self.classes_]))
        theta = (counts + self.alpha) / (counts.sum(axis=1, keepdims=True)
                                         + self.alpha * X.shape[1])
        self.log_theta = np.log(theta)
        return self

    def scores(self, X):
        """Log-posterior score (up to a constant) of every major; columns follow classes_."""
        return self.log_prior + np.atleast_2d(np.asarray(X, dtype=float)) @ self.log_theta.T

    def predict_proba(self, X):
        s = self.scores(X)
        e = np.exp(s - s.max(axis=1, keepdims=True))          # stable softmax
        return e / e.sum(axis=1, keepdims=True)

    def rank(self, X, n=2):
        out = []
        for row in self.scores(X):
            order = sorted(range(len(row)), key=lambda i: (-row[i], self.classes_[i]))
            out.append([self.classes_[i] for i in order[:n]])
        return out

    def match_percent(self, x):
        return {m: 100 * p for m, p in zip(self.classes_, self.predict_proba([x])[0])}
