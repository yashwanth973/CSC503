"""MajorMatch recommendation driver (the tool a student would use).

    python -m src.main --query Programming "Data Analysis" "Coding Club"
    python -m src.main --query Leadership "Debate Club" --model mnb --top-n 3
    python -m src.main --query Programming --dataset benchmark

Trains the chosen recommender on the development split of the chosen dataset
and prints the top-n majors with their match value p_{u,m} (Milestone 1).
For the full evaluation (Milestone 4) run:  python -m src.experiments
"""
import argparse

from src.cmn import config
from src.cmn.data import encode, load_survey, make_synthetic, split
from src.cmn.seed import set_seed
from src.majormatch.knn import KNN
from src.majormatch.mnb import MultinomialNaiveBayes


def parse_args():
    p = argparse.ArgumentParser(description="MajorMatch: recommend university majors "
                                            "from a student's skills and clubs.")
    p.add_argument("--query", nargs="+", metavar="FEATURE", required=True,
                   help=f"the student's skills/clubs; choose from: {config.FEATURES}")
    p.add_argument("--model", choices=["knn", "mnb", "all"], default="all")
    p.add_argument("--dataset", choices=["survey", "benchmark"], default="survey")
    p.add_argument("--k", type=int, default=config.K_DEFAULT)
    p.add_argument("--alpha", type=float, default=config.ALPHA_DEFAULT)
    p.add_argument("--top-n", type=int, default=config.TOP_N)
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    X, y, ids = (load_survey() if args.dataset == "survey"
                 else make_synthetic(config.ETA_MAIN, seed=args.seed))
    X_dev, _, y_dev, _, _, _ = split(X, y, ids, seed=args.seed)
    models = {}
    if args.model in ("knn", "all"):
        models[f"kNN-Jaccard (k={args.k})"] = KNN(k=args.k).fit(X_dev, y_dev)
    if args.model in ("mnb", "all"):
        models[f"MNB (alpha={args.alpha})"] = MultinomialNaiveBayes(args.alpha).fit(X_dev, y_dev)
    x = encode(args.query)
    for name, model in models.items():
        pct = model.match_percent(x)
        top = model.rank([x], n=args.top_n)[0]
        print(f"[{args.dataset}] {name}: {args.query} -> "
              + ", ".join(f"{m} ({pct.get(m, 0):.0f}%)" for m in top))


if __name__ == "__main__":
    main()
