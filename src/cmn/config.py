"""Shared configuration: paths, feature schema, splits, hyperparameter grids.

Every path is built with pathlib relative to the repository root, so the
project runs the same way on Windows, macOS and Linux.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = ROOT / "data" / "raw" / "student_data_extended.csv"
PREPROCESSED_DIR = ROOT / "data" / "preprocessed"
OUTPUT_DIR = ROOT / "output"
GROUND_TRUTH_DIR = ROOT / "evaluation" / "ground_truth"
RESULTS_DIR = ROOT / "evaluation" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# ---------------------------------------------------------------- schema
# F = F_skills U F_clubs: each student is a binary vector over these 12 features.
SKILLS = ["Artistic", "Data Analysis", "Leadership",
          "Problem Solving", "Programming", "Public Speaking"]
CLUBS = ["Art Club", "Coding Club", "Debate Club",
         "Music Club", "Sports Team", "Volunteer Group"]
FEATURES = SKILLS + CLUBS
LABEL = "Major"
MAJORS = ["Biology", "Computer Science", "History",
          "Mathematics", "Physics", "Psychology"]

# ---------------------------------------------------------------- protocol
SEED = 2026                 # global random seed (reproducibility)
SEEDS = [2026, 2027, 2028, 2029, 2030]   # repeated runs for mean +- std
TEST_SIZE = 0.20            # stratified 80 % development / 20 % final test
N_FOLDS = 5                 # stratified n-fold CV on the development set
TOP_N = 2                   # majors recommended to a student
PRIMARY_METRIC = "macro_f1" # model selection criterion (mean CV macro-F1)

# ---------------------------------------------------------------- proposed methods
K_DEFAULT = 7
K_GRID = [1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51]
ALPHA_DEFAULT = 1.0
ALPHA_GRID = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# ---------------------------------------------------------------- controlled benchmark
# Signal strength of the synthetic benchmark: eta = 0 -> pure major profiles,
# eta = 1 -> every feature independent of the major (same marginals as the survey).
ETA_MAIN = 0.5              # fixed a priori for RQ1 (moderate noise)
ETA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SYNTH_SIZE = 1000           # same size as the survey

# Prior beliefs about how typical each feature is of each major, used ONLY to
# generate the controlled benchmark. H = high, M = medium, L = low probability.
H, M, L = 0.85, 0.60, 0.35
MAJOR_PROFILES = {
    #                Art  Data Lead Prob Prog Spk | ArtC CodeC Deb  Mus  Spt  Vol
    "Biology":          [L, M, L, M, L, L,   L, L, L, L, M, H],
    "Computer Science": [L, M, L, H, H, L,   L, H, L, L, L, L],
    "History":          [M, L, M, L, L, H,   M, L, H, L, L, M],
    "Mathematics":      [L, H, L, H, M, L,   L, M, L, M, L, L],
    "Physics":          [L, M, L, H, M, L,   L, M, L, L, M, L],
    "Psychology":       [M, M, H, L, L, M,   M, L, L, M, L, H],
}
