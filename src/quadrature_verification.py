"""
Independent, non-random verification of the exact d=2 result.

This computes P(R > r0) by direct deterministic numerical integration of the
joint density of (m, r) over the region {(m+r)/|m-r| > r0}, WITHOUT using the
v = r/|m| substitution used to derive the closed form in exact_formulas.py.
Agreement between this script and exact_formulas.py is therefore a genuine
independent check of the algebra, not a restatement of it.

Usage:
    python3 quadrature_verification.py
"""

import numpy as np
from scipy import integrate

from exact_formulas import tail_prob, percentile


def fm(m):
    """Density of m' = |m|, m ~ N(0, 1/2)  (folded normal), sigma=1."""
    return (2.0 / np.sqrt(np.pi)) * np.exp(-m ** 2)


def fr(r):
    """Density of r ~ Rayleigh(1/2), sigma=1."""
    return 2.0 * r * np.exp(-r ** 2)


def P_R_greater(r0: float) -> float:
    """
    P(R > r0) via direct 2D quadrature over (m, r), m >= 0 (by symmetry m -> -m),
    r >= 0, split into the two branches r <= m and r > m.
    """
    k1 = (r0 - 1) / (r0 + 1)  # branch 1 (r<=m): condition reduces to r > k1*m
    k2 = (r0 + 1) / (r0 - 1)  # branch 2 (r> m): condition reduces to r < k2*m

    def inner(m):
        val1, _ = integrate.quad(fr, k1 * m, m, limit=200)
        val2, _ = integrate.quad(fr, m, k2 * m, limit=200)
        return (val1 + val2) * fm(m)

    val, _ = integrate.quad(inner, 0, 60, limit=400)
    return val


def main():
    print("Deterministic quadrature vs. exact closed form")
    print("-" * 60)
    print(f"{'r0':<10}{'quadrature':>15}{'exact':>15}{'abs. diff':>12}")
    for r0 in (2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0):
        q = P_R_greater(r0)
        ex = tail_prob(r0)
        print(f"{r0:<10.1f}{q:>15.8f}{ex:>15.8f}{abs(q-ex):>12.2e}")

    print()
    print("Percentiles solved from quadrature (bisection) vs. exact:")
    from scipy.optimize import brentq
    for p in (0.50, 0.90, 0.99):
        target = 1.0 - p
        r0_q = brentq(lambda r0: P_R_greater(r0) - target, 1.0001, 1e6)
        r0_ex = percentile(p)
        print(f"  {int(p*100)}th pctile: quadrature={r0_q:.6f}  exact={r0_ex:.6f}  diff={abs(r0_q-r0_ex):.2e}")


if __name__ == "__main__":
    main()
