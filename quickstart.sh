#!/usr/bin/env bash
# MajorMatch quick start (macOS / Linux): install -> test -> recommend -> full evaluation.
# Usage:  git clone https://github.com/yashwanth973/CSC503.git && cd CSC503 && bash quickstart.sh
#         bash quickstart.sh --quick     # 1 seed (~2 min) instead of 5 (~10 min)
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

echo "== 1/4 Installing requirements"
$PY -m pip install -q -r requirements.txt

echo "== 2/4 Unit + reproducibility tests"
$PY -m unittest discover tests

echo "== 3/4 Sample recommendation"
$PY -m src.main --query Programming "Data Analysis" "Coding Club" --dataset benchmark

echo "== 4/4 Milestone 4 evaluation (all tables and figures of the report)"
$PY -m src.experiments "$@"
echo "Results: Evaluation/results/   Figures: Evaluation/results/figures/"
