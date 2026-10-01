"""Problem 1(a): discrete-time Lyapunov stability test via an SDP.

Find P such that
    P > 0,
    A^T P A - P < 0.

Both LMIs are homogeneous in P (if P works, so does c*P for any c > 0), so the
strict inequalities can be replaced without loss of generality by
    P >= I,
    A^T P A - P <= -I,
which a numerical SDP solver can handle.
"""
import numpy as np
import cvxpy as cp

A = np.array([
    [-0.50, -0.82,  0.31],
    [-0.23,  0.61,  0.57],
    [ 0.31, -0.11, -0.36],
])
n = A.shape[0]
I = np.eye(n)

P = cp.Variable((n, n), symmetric=True)
constraints = [
    P >> I,
    A.T @ P @ A - P << -I,
]
# Feasibility problem; minimizing trace(P) just picks a well-scaled solution.
prob = cp.Problem(cp.Minimize(cp.trace(P)), constraints)
prob.solve(solver=cp.SCS if "MOSEK" not in cp.installed_solvers() else cp.MOSEK)

print("Solver status:", prob.status)
if prob.status not in ("optimal", "optimal_inaccurate"):
    raise SystemExit("LMIs infeasible -> no quadratic Lyapunov function; system not certified stable.")

P_val = P.value
P_val = (P_val + P_val.T) / 2  # symmetrize against solver round-off
np.set_printoptions(precision=4, suppress=True)
print("\nP =\n", P_val)

eig_P = np.linalg.eigvalsh(P_val)
eig_L = np.linalg.eigvalsh(A.T @ P_val @ A - P_val)
print("\neig(P)             =", eig_P, "-> P > 0:", bool(np.all(eig_P > 0)))
print("eig(A^T P A - P)   =", eig_L, "-> A^T P A - P < 0:", bool(np.all(eig_L < 0)))

# Sanity check: discrete-time stability <=> spectral radius of A < 1.
eig_A = np.linalg.eigvals(A)
print("\neig(A)             =", eig_A)
print("spectral radius    =", np.max(np.abs(eig_A)))
