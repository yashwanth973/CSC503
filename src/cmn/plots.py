"""Figures written to evaluation/results/figures/ by the experiments.

Colour follows the model (fixed order, never cycled); line style follows the dataset
(solid = benchmark, dashed = survey). A grey dotted line marks random guessing.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.cmn import config  # noqa: E402

COLORS = {"kNN-Jaccard": "#2a78d6", "MNB": "#eb6834", "kNN-Euclid": "#1baf7a",
          "LR": "#eda100", "RF": "#e87ba4", "SVM": "#008300", "Majority": "#8a8984"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "legend.frameon": False, "lines.linewidth": 2})


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def _chance(ax):
    ax.axhline(1 / 6, color=MUTED, linestyle=":", linewidth=1.2)
    ax.text(ax.get_xlim()[1], 1 / 6, " chance", va="center", ha="left", fontsize=8, color=MUTED)


def _curve(ax, cv, model, param, dataset, style, label):
    d = cv[(cv.model == model) & (cv.dataset == dataset)]
    g = d.groupby(param)["cv_macro_f1"].mean()
    ax.plot(g.index, g.values, style, color=COLORS[model], label=label,
            marker="o", markersize=3.5)


def rq2_hyperparameters(cv):
    """Two panels: CV macro-F1 vs k (both kNNs) and vs alpha (MNB), both datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    ax = axes[0]
    for model in ("kNN-Jaccard", "kNN-Euclid"):
        for ds, st in (("benchmark", "-"), ("survey", "--")):
            _curve(ax, cv, model, "k", ds, st, f"{model}, {ds}")
    ax.set_xscale("log")
    ax.set_xticks(config.K_GRID[::2])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("number of neighbours $k$ (log scale)")
    ax.set_ylabel("mean 5-fold CV macro-F1")
    _chance(ax)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, 0.93), ncol=1)
    ax = axes[1]
    for ds, st in (("benchmark", "-"), ("survey", "--")):
        _curve(ax, cv, "MNB", "alpha", ds, st, f"MNB, {ds}")
    ax.set_xscale("log")
    ax.set_xlabel(r"Laplace smoothing $\alpha$ (log scale)")
    _chance(ax)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, 0.93))
    return _save(fig, "rq2_hyperparameters.png")


def rq3_signal(rq3):
    """Test macro-F1 (mean +- std over seeds) vs. benchmark noise eta, one line per model."""
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    order = ["kNN-Jaccard", "MNB", "kNN-Euclid", "LR", "RF", "SVM", "Majority"]
    for m in order:
        d = rq3[rq3.model == m].sort_values("eta")
        lw = 2.4 if m in ("kNN-Jaccard", "MNB") else 1.4
        ax.plot(d.eta, d.macro_f1_mean, color=COLORS[m], label=m, linewidth=lw,
                marker="o", markersize=3)
        ax.fill_between(d.eta, d.macro_f1_mean - d.macro_f1_std,
                        d.macro_f1_mean + d.macro_f1_std, color=COLORS[m], alpha=0.10,
                        linewidth=0)
    ax.set_xlabel(r"benchmark noise $\eta$ (1 = survey-like, no signal)")
    ax.set_ylabel("test macro-F1")
    ax.set_xticks(config.ETA_GRID)
    _chance(ax)
    ax.legend(fontsize=6.5, ncol=2, loc="upper right")
    return _save(fig, "rq3_signal_strength.png")


def confusion_matrix(cm, labels, name):
    short = {"Biology": "Bio", "Computer Science": "CS", "History": "Hist",
             "Mathematics": "Math", "Physics": "Phys", "Psychology": "Psych"}
    fig, ax = plt.subplots(figsize=(2.6, 2.3))
    ax.grid(False)
    ax.imshow(cm, cmap="Blues")
    ticks = [short.get(lbl, lbl) for lbl in labels]
    ax.set_xticks(range(len(labels)), ticks, rotation=45, fontsize=7)
    ax.set_yticks(range(len(labels)), ticks, fontsize=7)
    ax.set_xlabel("predicted major")
    ax.set_ylabel("true major")
    vmax = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > 0.6 * vmax else INK)
    return _save(fig, name)


def class_balance(y):
    majors, counts = np.unique(y, return_counts=True)
    fig, ax = plt.subplots(figsize=(3.2, 1.8))
    ax.barh(range(len(majors)), counts, color=COLORS["kNN-Jaccard"], height=0.6)
    ax.set_yticks(range(len(majors)), majors, fontsize=7)
    ax.set_xlabel("students in the survey")
    for i, c in enumerate(counts):
        ax.text(c + 2, i, str(c), va="center", fontsize=7, color=MUTED)
    return _save(fig, "survey_class_balance.png")


def load_stats():
    with open(config.RESULTS_DIR / "dataset_statistics.json", encoding="utf-8") as f:
        return json.load(f)


def evaluation_panels(cv, rq3, cm, labels):
    """One full-width figure for the paper: (a) k, (b) alpha [RQ2], (c) eta [RQ3],
    (d) confusion matrix of kNN-Jaccard on the benchmark [RQ1]."""
    fig = plt.figure(figsize=(7.1, 2.25))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.75, 1.0, 1.0], wspace=0.5)
    order = ["kNN-Jaccard", "MNB", "kNN-Euclid", "LR", "RF", "SVM", "Majority"]

    ax = fig.add_subplot(gs[0])                                    # (a) k
    for model in ("kNN-Jaccard", "kNN-Euclid"):
        for ds, st in (("benchmark", "-"), ("survey", "--")):
            _curve(ax, cv, model, "k", ds, st, None)
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 15, 51])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("$k$ (log)")
    ax.set_ylabel("CV macro-F1")
    ax.axhline(1 / 6, color=MUTED, linestyle=":", linewidth=1.1)
    ax.set_title("(a) RQ2: $k$", fontsize=9)

    ax2 = fig.add_subplot(gs[1], sharey=ax)                        # (b) alpha
    for ds, st in (("benchmark", "-"), ("survey", "--")):
        _curve(ax2, cv, "MNB", "alpha", ds, st, None)
    ax2.set_xscale("log")
    ax2.set_xticks([0.01, 1, 10], ["0.01", "1", "10"])
    ax2.minorticks_off()
    ax2.set_xlabel(r"$\alpha$ (log)")
    ax2.axhline(1 / 6, color=MUTED, linestyle=":", linewidth=1.1)
    ax2.tick_params(labelleft=False)
    ax2.set_title(r"(b) RQ2: $\alpha$", fontsize=9)

    ax3 = fig.add_subplot(gs[2])                                   # (c) eta
    for m in order:
        d = rq3[rq3.model == m].sort_values("eta")
        lw = 2.2 if m in ("kNN-Jaccard", "MNB") else 1.2
        ax3.plot(d.eta, d.macro_f1_mean, color=COLORS[m], linewidth=lw, marker="o",
                 markersize=2.5)
        ax3.fill_between(d.eta, d.macro_f1_mean - d.macro_f1_std,
                         d.macro_f1_mean + d.macro_f1_std, color=COLORS[m], alpha=0.08,
                         linewidth=0)
    ax3.axhline(1 / 6, color=MUTED, linestyle=":", linewidth=1.1)
    ax3.set_xticks([0, 0.5, 1])
    ax3.set_xlabel(r"noise $\eta$ (1 = survey-like)")
    ax3.set_ylabel("test macro-F1")
    ax3.set_title(r"(c) RQ3: signal strength", fontsize=9)

    ax4 = fig.add_subplot(gs[3])                                   # (d) confusion
    ax4.grid(False)
    ax4.imshow(cm, cmap="Blues")
    short = ["Bio", "CS", "Hist", "Math", "Phys", "Psy"]
    ax4.set_xticks(range(6), short, rotation=90, fontsize=7)
    ax4.set_yticks(range(6), short, fontsize=7)
    ax4.set_xlabel("predicted", labelpad=1)
    ax4.set_ylabel("true", labelpad=1)
    for i in range(6):
        for j in range(6):
            ax4.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                     color="white" if cm[i, j] > 0.6 * cm.max() else INK)
    ax4.set_title("(d) kNN-J, benchmark", fontsize=9)

    handles = [matplotlib.lines.Line2D([], [], color=COLORS[m], linewidth=2, label=m)
               for m in order]
    handles += [matplotlib.lines.Line2D([], [], color=INK, linestyle="-", linewidth=1.2,
                                        label="benchmark"),
                matplotlib.lines.Line2D([], [], color=INK, linestyle="--", linewidth=1.2,
                                        label="survey"),
                matplotlib.lines.Line2D([], [], color=MUTED, linestyle=":", linewidth=1.2,
                                        label="chance (1/6)")]
    fig.legend(handles=handles, loc="upper center", ncol=10, fontsize=6.6,
               bbox_to_anchor=(0.5, 1.13), handlelength=1.6, columnspacing=1.0)
    return _save(fig, "evaluation_panels.png")
