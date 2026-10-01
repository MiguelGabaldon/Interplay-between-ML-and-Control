"""Numerical experiment: LipSDP vs. spectral-norm product for a 3-layer ReLU net.

f_a(x) = W3 sigma(W2 sigma(W1 x)), zero biases, with
    W1 = [[1, 1], [a, -a]],   W2 = [[a, 1], [-a, 1]],   W3 = [[1, -1]].

(i)  L_prod = ||W3||_2 ||W2||_2 ||W1||_2.

(ii) LipSDP-Neuron. ReLU is slope-restricted on [0, 1], so for any diagonal
     T >= 0 each layer satisfies [u; v]^T [[0, T], [T, -2T]] [u; v] >= 0 with
     v = sigma(u). Stacking z = [x0; x1; x2] (x1 = sigma(W1 x0), x2 = sigma(W2 x1)),
     f is L-Lipschitz if there exist diagonal T1, T2 >= 0 such that

        M(zeta, T1, T2) = [[ -zeta I,   W1^T T1,             0          ],
                           [  T1 W1,   -2 T1,             W2^T T2       ],
                           [  0,        T2 W2,    -2 T2 + W3^T W3       ]]  <= 0,

     with zeta = L^2. We minimize zeta over zeta >= 0, T1, T2 >= 1e-6 I
     (diagonal), and report L_SDP = sqrt(zeta*). Solver: Clarabel (free,
     interior-point, shipped with CVXPY).

(iii) Tabulate L_prod, L_SDP, L_prod / L_SDP and lambda_max(M*).

(Optional) Any input pair gives a lower bound on the true Lipschitz constant:
     L_lower = |f(x) - f(y)| / ||x - y||  at x = (2, -1), y = (3, 0),
     so L_lower <= L_true <= L_SDP <= L_prod.
"""
import numpy as np
import cvxpy as cp

A_VALUES = [1, 2, 5, 10]
T_MIN = 1e-6
FEAS_TOL = 1e-5
SOLVER = cp.CLARABEL

# Input pair for the optional lower bound.
X_LOW = np.array([2.0, -1.0])
Y_LOW = np.array([3.0, 0.0])


def weights(a):
    W1 = np.array([[1.0, 1.0], [a, -a]])
    W2 = np.array([[a, 1.0], [-a, 1.0]])
    W3 = np.array([[1.0, -1.0]])
    return W1, W2, W3


def relu_net(W1, W2, W3, x):
    """f_a(x) = W3 sigma(W2 sigma(W1 x)) with sigma = ReLU."""
    relu = lambda u: np.maximum(u, 0.0)
    return W3 @ relu(W2 @ relu(W1 @ x))


def lipsdp_matrix(W1, W2, W3, zeta, T1, T2):
    """Left-hand side of the LipSDP LMI; works for CVXPY expressions and arrays."""
    n0, n1, n2 = W1.shape[1], W1.shape[0], W2.shape[0]
    Z01 = np.zeros((n0, n2))
    block = cp.bmat if isinstance(zeta, cp.Expression) else np.block
    return block([
        [-zeta * np.eye(n0), W1.T @ T1,  Z01],
        [T1 @ W1,            -2 * T1,    W2.T @ T2],
        [Z01.T,              T2 @ W2,    -2 * T2 + W3.T @ W3],
    ])


def solve_lipsdp(W1, W2, W3):
    """Minimize zeta subject to the LipSDP LMI. Returns (zeta, T1, T2, status)."""
    zeta = cp.Variable(nonneg=True)
    t1 = cp.Variable(W1.shape[0])
    t2 = cp.Variable(W2.shape[0])
    M = lipsdp_matrix(W1, W2, W3, zeta, cp.diag(t1), cp.diag(t2))
    constraints = [
        (M + M.T) / 2 << 0,  # M is symmetric by construction; this tells CVXPY so.
        t1 >= T_MIN,
        t2 >= T_MIN,
    ]
    prob = cp.Problem(cp.Minimize(zeta), constraints)
    prob.solve(solver=SOLVER)
    if zeta.value is None:
        return np.nan, None, None, prob.status
    return float(zeta.value), np.diag(t1.value), np.diag(t2.value), prob.status


def main():
    print(f"Solver: {SOLVER} (CVXPY {cp.__version__})\n")
    header = (f"{'a':>4} {'L_prod':>10} {'L_SDP':>10} {'L_prod/L_SDP':>13} "
              f"{'lambda_max(M*)':>15} {'LMI ok':>7} {'status':>10}")
    print(header)
    print("-" * len(header))
    lower_rows = []
    for a in A_VALUES:
        W1, W2, W3 = weights(a)
        L_prod = np.prod([np.linalg.norm(W, 2) for W in (W3, W2, W1)])

        zeta, T1, T2, status = solve_lipsdp(W1, W2, W3)
        L_sdp = np.sqrt(zeta)
        # M* evaluated at the unrounded solver output.
        M_star = lipsdp_matrix(W1, W2, W3, zeta, T1, T2)
        lam_max = np.linalg.eigvalsh(M_star).max()

        print(f"{a:>4} {L_prod:>10.4f} {L_sdp:>10.4f} {L_prod / L_sdp:>13.4f} "
              f"{lam_max:>15.4e} {str(lam_max <= FEAS_TOL):>7} {status:>10}")

        fx, fy = relu_net(W1, W2, W3, X_LOW), relu_net(W1, W2, W3, Y_LOW)
        L_low = np.linalg.norm(fx - fy) / np.linalg.norm(X_LOW - Y_LOW)
        lower_rows.append((a, fx.item(), fy.item(), L_low, L_sdp, L_prod))

    print(f"\nOptional: L_lower = |f(x) - f(y)| / ||x - y|| at x = {X_LOW.tolist()}, "
          f"y = {Y_LOW.tolist()}\n")
    header = (f"{'a':>4} {'f(x)':>8} {'f(y)':>8} {'L_lower':>10} {'L_SDP':>10} {'L_prod':>10} "
              f"{'L_SDP/L_lower':>14} {'L_prod/L_lower':>15}")
    print(header)
    print("-" * len(header))
    for a, fx, fy, L_low, L_sdp, L_prod in lower_rows:
        print(f"{a:>4} {fx:>8.4f} {fy:>8.4f} {L_low:>10.4f} {L_sdp:>10.4f} {L_prod:>10.4f} "
              f"{L_sdp / L_low:>14.4f} {L_prod / L_low:>15.4f}")


if __name__ == "__main__":
    main()
