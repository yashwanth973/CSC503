"""Data layer.

* load_survey()      : the team's student survey (data/raw) -> X in {0,1}^{n x 12}, y
* make_synthetic()   : controlled benchmark with a known feature-major relationship
* split()            : stratified development / final-test split
* encode()           : list of feature names -> binary vector (for queries)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.cmn import config


def load_raw(path: Path = config.RAW_DATA) -> pd.DataFrame:
    """Read the spreadsheet export (row 0 is a group header, row 1 the column names)."""
    df = pd.read_csv(path, header=1)
    missing = [c for c in config.FEATURES + [config.LABEL] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from {path.name}: {missing}")
    return df


def load_survey():
    """Return (X, y, ids) for the survey: the 12 binary skill/club features and the major."""
    df = load_raw()
    df = df[["StudentID"] + config.FEATURES + [config.LABEL]].dropna()
    X = (df[config.FEATURES].to_numpy(dtype=int) > 0).astype(int)
    return X, df[config.LABEL].to_numpy(dtype=object), df["StudentID"].to_numpy(dtype=int)


def make_synthetic(eta: float, n: int = config.SYNTH_SIZE, seed: int = config.SEED):
    """Controlled benchmark.

    Each student draws a major m with the survey's class proportions, then each
    feature j independently with probability
        p_{m,j}(eta) = (1 - eta) * profile_{m,j} + eta * prevalence_j,
    where prevalence_j is feature j's rate in the survey. eta = 1 therefore
    reproduces the survey's (signal-free) distribution exactly.
    """
    rng = np.random.default_rng(seed)
    X_s, y_s, _ = load_survey()
    majors, counts = np.unique(y_s, return_counts=True)
    prevalence = X_s.mean(axis=0)
    y = rng.choice(majors, size=n, p=counts / counts.sum())
    profiles = np.array([config.MAJOR_PROFILES[m] for m in y])
    p = (1 - eta) * profiles + eta * prevalence
    X = (rng.random((n, len(config.FEATURES))) < p).astype(int)
    return X, y.astype(object), np.arange(1, n + 1)


def split(X, y, ids, seed: int = config.SEED, test_size: float = config.TEST_SIZE):
    """Stratified 80/20 development / final-test split (fixed seed)."""
    return train_test_split(X, y, ids, test_size=test_size, stratify=y, random_state=seed)


def save_split(name, X_dev, X_test, y_dev, y_test, id_dev, id_test):
    """Write the exact partition used, so every result can be traced back to its data."""
    config.PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    for part, X, y, ids in (("dev", X_dev, y_dev, id_dev), ("test", X_test, y_test, id_test)):
        df = pd.DataFrame(X, columns=config.FEATURES)
        df.insert(0, "StudentID", ids)
        df[config.LABEL] = y
        df.to_csv(config.PREPROCESSED_DIR / f"{name}_{part}.csv", index=False)
    pd.DataFrame({"StudentID": id_test, config.LABEL: y_test}).to_csv(
        config.GROUND_TRUTH_DIR / f"{name}_test_labels.csv", index=False)


def encode(selected):
    """List of feature names (e.g. ["Programming", "Coding Club"]) -> binary vector."""
    unknown = [s for s in selected if s not in config.FEATURES]
    if unknown:
        raise ValueError(f"Unknown features: {unknown}. Valid: {config.FEATURES}")
    return np.array([1 if f in selected else 0 for f in config.FEATURES])


def statistics(X, y):
    """Dataset statistics reported in the paper (Table 1)."""
    majors, counts = np.unique(y, return_counts=True)
    share = counts / counts.sum()
    return {"instances": len(y), "features": X.shape[1], "classes": len(majors),
            "min_class_share": round(float(share.min()), 3),
            "max_class_share": round(float(share.max()), 3),
            "avg_features_per_student": round(float(X.sum(axis=1).mean()), 2),
            "feature_prevalence_min": round(float(X.mean(axis=0).min()), 2),
            "feature_prevalence_max": round(float(X.mean(axis=0).max()), 2),
            "missing_values": 0}
