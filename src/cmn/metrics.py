"""Evaluation metrics shared by all models (multiclass, macro-averaged)."""
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def macro_prf(y_true, y_pred, labels):
    """Macro-averaged precision, recall and F1 over `labels` (every class weighs the same)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f = [], [], []
    for c in labels:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        p.append(prec)
        r.append(rec)
        f.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(p)), float(np.mean(r)), float(np.mean(f))


def top_n_accuracy(ranked, y_true, n):
    """Fraction of students whose true major is among the first n recommendations."""
    return float(np.mean([y in r[:n] for r, y in zip(ranked, y_true)]))


def macro_auc(y_true, proba, labels):
    """One-vs-rest macro AUC from per-class scores (columns ordered as `labels`)."""
    try:
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                                   labels=labels))
    except ValueError:            # e.g. a constant scorer
        return 0.5


def evaluate(model, X, y, labels, n):
    """All metrics for one fitted model on (X, y)."""
    ranked = model.rank(X, n=max(n, 1))
    top1 = [r[0] for r in ranked]
    p, r, f = macro_prf(y, top1, labels)
    return {"macro_precision": p, "macro_recall": r, "macro_f1": f,
            "top1_accuracy": top_n_accuracy(ranked, y, 1),
            f"top{n}_accuracy": top_n_accuracy(ranked, y, n),
            "macro_auc": macro_auc(y, model.predict_proba(X), labels)}, top1


def confusion(y_true, y_pred, labels):
    return confusion_matrix(y_true, y_pred, labels=labels)
