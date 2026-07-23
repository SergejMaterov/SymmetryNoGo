"""
High-precision Monte Carlo scan of the splitting-ratio distribution for GOE(d),
d = 2..6, using the exact O(d)-invariant maximum-entropy normalization derived
in Section 5.1 of the paper (diagonal variance sigma^2, off-diagonal variance
sigma^2/2).

No closed form is available for d > 2 (see Section 5.3 discussion), so this
script reports Monte Carlo estimates together with bootstrap standard errors
on the median, to make the numerical uncertainty explicit rather than
implicit as in the original (20,000-sample, no error bars) version.

Usage:
    python3 general_d_scan.py --n 3000000 --seed 12345 --out ../results/general_d_table.csv
"""

import argparse
import numpy as np
import pandas as pd


def sample_goe(d: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw n independent GOE(d) matrices with diag ~ N(0,1), off-diag ~ N(0,1/2)
    (sigma=1; the ratio R is scale-invariant so sigma is WLOG 1), and return
    the splitting ratios R = max_i|lambda_i| / min_i|lambda_i|.
    """
    diag = rng.standard_normal((n, d))
    iu = np.triu_indices(d, 1)
    n_off = len(iu[0])
    off = rng.standard_normal((n, n_off)) * np.sqrt(0.5)

    M = np.zeros((n, d, d))
    M[:, iu[0], iu[1]] = off
    M[:, iu[1], iu[0]] = off
    M[:, np.arange(d), np.arange(d)] = diag

    eigs = np.linalg.eigvalsh(M)
    absE = np.abs(eigs)
    hi = absE.max(axis=1)
    lo = absE.min(axis=1)
    mask = lo > 1e-10
    return hi[mask] / lo[mask]


def bootstrap_median_se(R: np.ndarray, n_boot: int = 200, rng=None) -> float:
    """Bootstrap standard error of the sample median (subsampled for speed)."""
    rng = rng or np.random.default_rng(0)
    sub_n = min(200_000, len(R))
    idx_pool = rng.choice(len(R), size=(n_boot, sub_n), replace=True)
    meds = np.median(R[idx_pool], axis=1)
    return float(meds.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3_000_000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--dmax", type=int, default=6)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for d in range(2, args.dmax + 1):
        R = sample_goe(d, args.n, rng)
        med = float(np.median(R))
        med_se = bootstrap_median_se(R, rng=rng)
        p90 = float(np.percentile(R, 90))
        p99 = float(np.percentile(R, 99))
        pg20 = float(np.mean(R > 20)) * 100
        pg200 = float(np.mean(R > 200)) * 100
        rows.append(dict(d=d, n=len(R), median=med, median_se=med_se,
                          p90=p90, p99=p99, P_gt_20_pct=pg20, P_gt_200_pct=pg200))
        print(f"d={d}: median={med:.3f} (SE~{med_se:.4f})  p90={p90:.2f}  p99={p99:.2f}  "
              f"P(>20)={pg20:.3f}%  P(>200)={pg200:.4f}%")

    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
