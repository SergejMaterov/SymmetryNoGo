"""
Monte Carlo cross-check of the exact d=2 closed-form result in exact_formulas.py.

Draws N independent GOE(2) matrices with the exact normalization stated there,
computes the splitting ratio R = max(|lambda_i|)/min(|lambda_i|), and compares
empirical percentiles / tail probabilities against the exact formula.

Usage:
    python3 monte_carlo_d2.py --n 20000000 --seed 0
"""

import argparse
import numpy as np

from exact_formulas import tail_prob, percentile


def sample_R(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n independent GOE(2) splitting ratios."""
    x = rng.normal(0.0, 1.0, n)
    y = rng.normal(0.0, 1.0, n)
    z = rng.normal(0.0, np.sqrt(0.5), n)

    lam1 = (x + y) / 2 + np.sqrt(((x - y) / 2) ** 2 + z ** 2)
    lam2 = (x + y) / 2 - np.sqrt(((x - y) / 2) ** 2 + z ** 2)

    a, b = np.abs(lam1), np.abs(lam2)
    hi, lo = np.maximum(a, b), np.minimum(a, b)
    mask = lo > 1e-12  # exclude the (measure-zero) degenerate case
    return hi[mask] / lo[mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    R = sample_R(args.n, rng)

    print(f"Monte Carlo, N = {len(R):,} samples (seed={args.seed})")
    print("-" * 60)
    print(f"{'quantity':<18}{'Monte Carlo':>15}{'exact':>15}{'abs. diff':>12}")
    for p in (0.50, 0.90, 0.99):
        mc = np.percentile(R, p * 100)
        ex = percentile(p)
        print(f"{f'{int(p*100)}th pctile':<18}{mc:>15.4f}{ex:>15.4f}{abs(mc-ex):>12.5f}")
    for r0 in (20.0, 200.0):
        mc = float(np.mean(R > r0)) * 100
        ex = tail_prob(r0) * 100
        print(f"{f'P(R>{r0:.0f}) [%]':<18}{mc:>15.4f}{ex:>15.4f}{abs(mc-ex):>12.5f}")


if __name__ == "__main__":
    main()
