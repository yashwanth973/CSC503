@echo off
REM MajorMatch quick start (Windows): install -> test -> recommend -> full evaluation.
REM Usage:  quickstart.bat            (5 seeds, ~10 min)
REM         quickstart.bat --quick    (1 seed, ~2 min)
cd /d "%~dp0"

echo == 1/4 Installing requirements
python -m pip install -q -r requirements.txt || exit /b 1

echo == 2/4 Unit + reproducibility tests
python -m unittest discover tests || exit /b 1

echo == 3/4 Sample recommendation
python -m src.main --query Programming "Data Analysis" "Coding Club" --dataset benchmark || exit /b 1

echo == 4/4 Milestone 4 evaluation (all tables and figures of the report)
python -m src.experiments %* || exit /b 1
echo Results: Evaluation\results\   Figures: Evaluation\results\figures\
