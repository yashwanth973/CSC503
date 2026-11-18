"""Hyperparameter selection by stratified n-fold cross-validation on the development set.

The SAME folds are used for every model and every configuration (fair comparison),
and the final test set is never touched here.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.cmn import config
from src.cmn.metrics import macro_prf, top_n_accuracy


def make_folds(y_dev, seed, n_folds=config.N_FOLDS):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y_dev)), y_dev))


def cross_validate(build, grid, X_dev, y_dev, folds, labels, top_n=config.TOP_N):
    """Mean/std CV macro-F1 and top-n accuracy for every configuration in `grid`.
    Returns (best_config, rows); best = highest mean macro-F1, ties -> earlier in grid."""
    rows = []
    for hp in grid:
        f1s, topn = [], []
        for tr, va in folds:
            model = build(**hp).fit(X_dev[tr], y_dev[tr])
            ranked = model.rank(X_dev[va], n=top_n)
            f1s.append(macro_prf(y_dev[va], [r[0] for r in ranked], labels)[2])
            topn.append(top_n_accuracy(ranked, y_dev[va], top_n))
        rows.append({**hp, "cv_macro_f1": float(np.mean(f1s)), "cv_macro_f1_std": float(np.std(f1s)),
                     f"cv_top{top_n}_accuracy": float(np.mean(topn))})
    best = max(range(len(grid)), key=lambda i: (round(rows[i]["cv_macro_f1"], 12), -i))
    return grid[best], rows
