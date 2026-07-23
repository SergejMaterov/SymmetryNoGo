#!/usr/bin/env bash
# Reproduces every numerical claim in Section 5 ("Proposition C, Strengthened")
# of the paper, in order. Run from the repository root.
set -euo pipefail

echo "############################################################"
echo "# 1. Exact closed-form results (d = 2)"
echo "############################################################"
python3 src/exact_formulas.py

echo
echo "############################################################"
echo "# 2. Independent deterministic-quadrature verification"
echo "############################################################"
python3 src/quadrature_verification.py

echo
echo "############################################################"
echo "# 3. Monte Carlo verification (20,000,000 samples, as quoted"
echo "#    in the paper). This step takes ~30-60s."
echo "############################################################"
python3 src/monte_carlo_d2.py --n 20000000 --seed 0

echo
echo "############################################################"
echo "# 4. General-d scan (3,000,000 samples per d, d=2..6)."
echo "#    This step takes a few minutes."
echo "############################################################"
python3 src/general_d_scan.py --n 3000000 --seed 12345 --out results/general_d_table.csv

echo
echo "Done. Table written to results/general_d_table.csv"
