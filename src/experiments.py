"""Milestone 4 evaluation: runs every experiment of the report and writes all tables/figures.

    python -m src.experiments            # full study (5 seeds), ~ a few minutes
    python -m src.experiments --quick    # 1 seed, for a fast check

Protocol (identical for every model):
  1. stratified 80/20 development / final-test split (seed s)
  2. stratified 5-fold CV on the development set, same folds for all models,
     grid search, best configuration = highest mean CV macro-F1
  3. refit the best configuration on the whole development set
  4. evaluate ONCE on the final test set: macro P/R/F1, top-1/top-2 accuracy, macro AUC
Repeated for seeds 2026..2030; tables report mean +- std.

RQ1: test performance of the proposed methods vs. baselines (survey + benchmark eta=0.5)
RQ2: CV macro-F1 as a function of k (kNN) and alpha (MNB)
RQ3: test macro-F1 as the benchmark's signal strength eta goes from 0 (clean) to 1 (survey-like)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from src.baselines.models import MODELS, build
from src.cmn import config, plots
from src.cmn.data import load_survey, make_synthetic, save_split, split, statistics
from src.cmn.metrics import confusion, evaluate
from src.cmn.seed import set_seed
from src.majormatch.model_selection import cross_validate, make_folds


def run_dataset(name, X, y, ids, seed, models, save=False):
    """Full protocol on one dataset and one seed. Returns (test_rows, cv_rows, confusions)."""
    set_seed(seed)
    X_dev, X_test, y_dev, y_test, id_dev, id_test = split(X, y, ids, seed=seed)
    if save:
        save_split(name, X_dev, X_test, y_dev, y_test, id_dev, id_test)
    labels = sorted(set(y))
    folds = make_folds(y_dev, seed)
    test_rows, cv_rows, confs = [], [], {}
    for mname in models:
        _, grid, proposed = MODELS[mname]
        best, rows = cross_validate(lambda **hp: build(mname, seed, **hp), grid,
                                    X_dev, y_dev, folds, labels)
        cv_rows += [{"dataset": name, "seed": seed, "model": mname, **r} for r in rows]
        t0 = time.perf_counter()
        model = build(mname, seed, **best).fit(X_dev, y_dev)
        metrics, top1 = evaluate(model, X_test, y_test, labels, config.TOP_N)
        elapsed = time.perf_counter() - t0
        test_rows.append({"dataset": name, "seed": seed, "model": mname, "proposed": proposed,
                          "best_config": json.dumps(best), **metrics, "seconds": elapsed})
        confs[mname] = confusion(y_test, top1, labels)
    return test_rows, cv_rows, confs


def summarise(df, by):
    metrics = ["macro_f1", "macro_precision", "macro_recall", "top1_accuracy",
               f"top{config.TOP_N}_accuracy", "macro_auc", "seconds"]
    g = df.groupby(by, sort=False)[metrics]
    out = g.mean().round(4).add_suffix("_mean").join(g.std(ddof=0).round(4).add_suffix("_std"))
    return out.reset_index()


def figures_only():
    """Redraw every figure from the saved CSVs (no retraining)."""
    cv_df = pd.read_csv(config.RESULTS_DIR / "cv_results_all_runs.csv")
    rq3 = pd.read_csv(config.RESULTS_DIR / "rq3_eta_sweep.csv")
    cm_df = pd.read_csv(config.RESULTS_DIR / "confusion_benchmark_kNN-Jaccard.csv", index_col=0)
    labels = list(cm_df.index)
    plots.rq2_hyperparameters(cv_df)
    plots.rq3_signal(rq3)
    plots.evaluation_panels(cv_df, rq3, cm_df.to_numpy(), labels)


def main():
    ap = argparse.ArgumentParser(description="MajorMatch Milestone 4 experiments")
    ap.add_argument("--quick", action="store_true", help="one seed instead of five")
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw figures from saved results without retraining")
    args = ap.parse_args()
    if args.figures_only:
        figures_only()
        return
    seeds = config.SEEDS[:1] if args.quick else config.SEEDS
    models = list(MODELS)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    # ---------------- datasets
    Xs, ys, ids_s = load_survey()
    stats = {"survey": statistics(Xs, ys)}

    test_all, cv_all, conf_main = [], [], {}
    # ---------------- RQ1 + RQ2: survey and benchmark (eta fixed a priori)
    for seed in seeds:
        Xb, yb, idb = make_synthetic(config.ETA_MAIN, seed=seed)
        if seed == config.SEED:
            stats["benchmark"] = statistics(Xb, yb)
        for name, X, y, ids in (("survey", Xs, ys, ids_s), ("benchmark", Xb, yb, idb)):
            t, c, conf = run_dataset(name, X, y, ids, seed, models, save=(seed == config.SEED))
            test_all += t
            cv_all += c
            if seed == config.SEED:
                conf_main[name] = conf
        print(f"[RQ1/RQ2] seed {seed} done ({time.time() - start:.0f}s)")

    # ---------------- RQ3: signal-strength sweep on the benchmark
    for eta in config.ETA_GRID:
        for seed in seeds:
            Xb, yb, idb = make_synthetic(eta, seed=seed)
            t, _, _ = run_dataset(f"eta={eta}", Xb, yb, idb, seed, models)
            for r in t:
                r["eta"] = eta
            test_all += t
        print(f"[RQ3] eta={eta} done ({time.time() - start:.0f}s)")

    # ---------------- write results
    test_df, cv_df = pd.DataFrame(test_all), pd.DataFrame(cv_all)
    test_df.to_csv(config.RESULTS_DIR / "test_results_all_runs.csv", index=False)
    cv_df.to_csv(config.RESULTS_DIR / "cv_results_all_runs.csv", index=False)
    rq1 = summarise(test_df[test_df.dataset.isin(["survey", "benchmark"])], ["dataset", "model"])
    rq1.to_csv(config.RESULTS_DIR / "rq1_test_summary.csv", index=False)
    rq3 = summarise(test_df[test_df.dataset.str.startswith("eta=")], ["eta", "model"])
    rq3.to_csv(config.RESULTS_DIR / "rq3_eta_sweep.csv", index=False)
    best_cfg = test_df[test_df.dataset.isin(["survey", "benchmark"])][
        ["dataset", "seed", "model", "best_config"]]
    best_cfg.to_csv(config.RESULTS_DIR / "best_configs.csv", index=False)
    with open(config.RESULTS_DIR / "dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    labels = sorted(set(ys))
    for name, confs in conf_main.items():
        for m, cm in confs.items():
            pd.DataFrame(cm, index=labels, columns=labels).to_csv(
                config.RESULTS_DIR / f"confusion_{name}_{m}.csv")

    # ---------------- figures
    plots.rq2_hyperparameters(cv_df)
    plots.rq3_signal(rq3)
    plots.confusion_matrix(conf_main["benchmark"]["kNN-Jaccard"], labels,
                           "confusion_benchmark_knn.png")
    plots.class_balance(ys)
    plots.evaluation_panels(cv_df, rq3, conf_main["benchmark"]["kNN-Jaccard"], labels)

    pd.set_option("display.width", 200)
    cols = ["dataset", "model", "macro_f1_mean", "macro_f1_std", "top2_accuracy_mean",
            "macro_auc_mean", "seconds_mean"]
    print(rq1[cols].to_string(index=False))
    print(f"[done] {time.time() - start:.0f}s -> {config.RESULTS_DIR.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
