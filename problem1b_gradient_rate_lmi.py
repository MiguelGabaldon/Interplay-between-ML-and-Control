"""Problem 1(b): convergence rate of the gradient method via a 2x2 LMI.

For x_{k+1} = x_k - alpha * grad f(x_k), with f L-smooth and m-strongly convex,
the rate rho is certified if there exists lambda >= 0 such that

    M(rho^2, lambda) = [[1 - rho^2, -alpha], [-alpha, alpha^2]]
                       + lambda * [[-2 m L, m + L], [m + L, -2]]  <=  0.

M is affine in (rho^2, lambda), so minimizing rho^2 subject to M <= 0 is an
SDP. It is solved with Clarabel, the free interior-point solver shipped with
CVXPY.
"""
import numpy as np
import cvxpy as cp

PAIRS = [(1, 10), (1, 100), (0.1, 20), (1, 1000), (1, 5000), (1, 50000)]

# Step sizes as fractions of 2/L (all satisfy 0 < alpha < 2/L), plus the
# classical optimal step 2/(m+L).
ALPHA_RULES = [
    ("0.5/L",    lambda m, L: 0.5 / L),
    ("1/L",      lambda m, L: 1.0 / L),
    ("2/(m+L)",  lambda m, L: 2.0 / (m + L)),
    ("1.5/L",    lambda m, L: 1.5 / L),
    ("1.9/L",    lambda m, L: 1.9 / L),
]

SOLVER = cp.CLARABEL


def lmi_matrix(m, L, alpha, rho2, lam):
    """M(rho^2, lambda); works for both CVXPY expressions and floats."""
    E11 = np.array([[1.0, 0.0], [0.0, 0.0]])
    M0 = np.array([[1.0, -alpha], [-alpha, alpha**2]])
    S = np.array([[-2 * m * L, m + L], [m + L, -2.0]])
    return M0 - rho2 * E11 + lam * S


def solve_rate_lmi(m, L, alpha):
    """Minimize rho^2 subject to the LMI. Returns (rho2, lam, status)."""
    rho2 = cp.Variable(nonneg=True)
    lam = cp.Variable(nonneg=True)
    prob = cp.Problem(cp.Minimize(rho2), [lmi_matrix(m, L, alpha, rho2, lam) << 0])
    prob.solve(solver=SOLVER)
    if rho2.value is None:
        return np.nan, np.nan, prob.status
    return float(rho2.value), float(lam.value), prob.status


def run():
    print(f"\n=== Solver: {SOLVER} (CVXPY {cp.__version__}) ===")
    header = (f"{'m':>5} {'L':>7} {'alpha':>8} {'alpha val':>10} {'status':>18} "
              f"{'1 - rho':>11} {'1 - rho_th':>11} {'rel err':>9} "
              f"{'lambda':>11} {'lambda_th':>11} {'max eig M':>10}")
    print(header)
    print("-" * len(header))
    for m, L in PAIRS:
        for name, rule in ALPHA_RULES:
            alpha = rule(m, L)
            rho2, lam, status = solve_rate_lmi(m, L, alpha)
            rho = np.sqrt(rho2)
            rho_th = max(abs(1 - m * alpha), abs(1 - L * alpha))
            # det(-M) = -(L-m)^2 lam^2 + 2 lam (rho^2 - (1-m a)(1-L a)) - a^2 rho^2.
            # Its max over lam is >= 0 iff rho >= max(1-m a, L a-1), attained at
            # lam = alpha * rho / (L - m).
            lam_th = alpha * rho_th / (L - m)
            rel = abs((1 - rho) - (1 - rho_th)) / abs(1 - rho_th)
            max_eig = np.linalg.eigvalsh(lmi_matrix(m, L, alpha, rho2, lam)).max()
            print(f"{m:>5g} {L:>7g} {name:>8} {alpha:>10.3e} {status:>18} "
                  f"{1 - rho:>11.4e} {1 - rho_th:>11.4e} {rel:>9.1e} "
                  f"{lam:>11.4e} {lam_th:>11.4e} {max_eig:>10.1e}")
        print()


if __name__ == "__main__":
    run()
