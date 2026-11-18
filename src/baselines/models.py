"""Baselines (scikit-learn) behind the same interface as the proposed methods:
fit(X, y), predict_proba(X) with columns = classes_, rank(X, n) -> top-n majors.

Every model -- proposed or baseline -- is listed in MODELS with its hyperparameter
grid, so all of them go through exactly the same splits, folds and metrics.
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.cmn import config
from src.majormatch.knn import KNN
from src.majormatch.mnb import MultinomialNaiveBayes


class SklearnRecommender:
    """Adapter: ranks majors by the classifier's predicted probability (ties by name)."""

    def __init__(self, estimator, name):
        self.estimator = estimator
        self.name = name

    def fit(self, X, y):
        self.estimator.fit(np.asarray(X), np.asarray(y, dtype=object))
        self.classes_ = np.asarray(self.estimator.classes_, dtype=object)
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(np.atleast_2d(X))

    def rank(self, X, n=2):
        out = []
        for row in self.predict_proba(X):
            order = sorted(range(len(row)), key=lambda i: (-row[i], self.classes_[i]))
            out.append([self.classes_[i] for i in order[:n]])
        return out


def _sk(name, factory):
    return lambda seed, **hp: SklearnRecommender(factory(seed, **hp), name)


# name -> (constructor(seed, **hp), grid as list of dicts, proposed?)
MODELS = {
    "kNN-Jaccard": (lambda seed, k: KNN(k=k),
                    [{"k": k} for k in config.K_GRID], True),
    "MNB": (lambda seed, alpha: MultinomialNaiveBayes(alpha=alpha),
            [{"alpha": a} for a in config.ALPHA_GRID], True),
    "kNN-Euclid": (_sk("kNN-Euclid", lambda seed, k: KNeighborsClassifier(
                       n_neighbors=k, metric="euclidean")),
                   [{"k": k} for k in config.K_GRID], False),
    "LR": (_sk("LR", lambda seed, C: LogisticRegression(C=C, max_iter=2000, random_state=seed)),
           [{"C": c} for c in [0.01, 0.1, 1.0, 10.0, 100.0]], False),
    "RF": (_sk("RF", lambda seed, depth, trees: RandomForestClassifier(
               n_estimators=trees, max_depth=depth, random_state=seed, n_jobs=1)),
           [{"depth": d, "trees": t} for d in [3, 5, 10, None] for t in [100, 300]], False),
    # RBF SVM; class probabilities by Platt scaling (internal 3-fold) for ranking/AUC
    "SVM": (_sk("SVM", lambda seed, C: CalibratedClassifierCV(
                SVC(C=C, kernel="rbf", random_state=seed), method="sigmoid", cv=3,
                ensemble=False)),
            [{"C": c} for c in [0.1, 1.0, 10.0]], False),
    "Majority": (_sk("Majority", lambda seed: DummyClassifier(strategy="prior")),
                 [{}], False),
}


def build(name, seed, **hp):
    return MODELS[name][0](seed, **hp)
