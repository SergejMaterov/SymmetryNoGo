"""
Exact closed-form results for the GOE(2) splitting-ratio distribution.

Setup (see paper, Section 5.2):
    M = [[x, z], [z, y]],  x, y ~ N(0, sigma^2),  z ~ N(0, sigma^2/2),  independent.
    (This is the O(2)-invariant, maximum-entropy GOE(2) normalization; see Section 5.1.)

    Eigenvalues: lambda_{1,2} = m +/- r
        m = (x+y)/2 ~ N(0, sigma^2/2)
        r = sqrt(u^2 + z^2), u = (x-y)/2  ->  r ~ Rayleigh(sigma^2/2), independent of m

    R := max(|lambda_1|,|lambda_2|) / min(|lambda_1|,|lambda_2|)

Derivation summary (full algebra in the paper):
    Let v = r / |m|. Then R is a deterministic function of v alone:
        R(v) = (1+v)/|1-v|
    and v has the exact density (independent of sigma):
        f_v(v) = v * (1+v^2)^(-3/2),   v >= 0
        F_v(v) = 1 - (1+v^2)^(-1/2)

    Integrating over both branches of R(v) gives the exact tail probability:
        P(R > r0) = sqrt( 2 / (1 + r0^2) ),   r0 >= 1

    Inverting for percentiles:
        r0(alpha) = sqrt( 2/alpha^2 - 1 ),  where alpha = P(R > r0)
"""

import math


def tail_prob(r0: float) -> float:
    """Exact P(R > r0) for r0 >= 1."""
    if r0 < 1:
        raise ValueError("R >= 1 always; tail probability undefined for r0 < 1.")
    return math.sqrt(2.0 / (1.0 + r0 ** 2))


def percentile(p: float) -> float:
    """
    Exact value r0 such that P(R <= r0) = p, i.e. P(R > r0) = 1 - p.
    p in (0, 1), e.g. p=0.5 for the median, p=0.90, p=0.99.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1).")
    alpha = 1.0 - p
    return math.sqrt(2.0 / alpha ** 2 - 1.0)


def mean_splitting(sigma: float = 1.0) -> float:
    """Exact E[s], s = lambda_1 - lambda_2 = 2r. Equals sigma * sqrt(pi)."""
    return sigma * math.sqrt(math.pi)


if __name__ == "__main__":
    print("Exact d=2 GOE splitting-ratio results")
    print("-" * 45)
    for p in (0.50, 0.90, 0.99):
        print(f"  {int(p*100):>2}th percentile of R = {percentile(p):.6f}")
    for r0 in (20.0, 200.0):
        print(f"  P(R > {r0:>5.0f}) = {tail_prob(r0)*100:.6f} %")
    print(f"  E[s] (sigma=1)      = {mean_splitting():.6f}  (= sqrt(pi) = {math.sqrt(math.pi):.6f})")
