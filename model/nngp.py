import numpy as np
from scipy.optimize import minimize
from typing import Optional, Tuple


def exponential_cov(dist: np.ndarray, sigma2: float, ell: float) -> np.ndarray:
    # Guard for extreme/invalid ell
    ell = float(max(ell, 1e-12))
    return sigma2 * np.exp(-dist / ell)


class SpatioTemporalNNGP:
    """
    Closest-to-paper NNGP-style predictor (Vecchia conditional) with two critical fixes:
      1) Internal coordinate normalization so ell has a meaningful scale.
      2) Better default ell when user doesn't provide one (ell is not identifiable from target-only fit).

    Interface is unchanged:
      - fit(X: (L,4), Y: (L,))
      - predict(X_nb: (L,K,4), Y_nb: (L,K,1), X_tg: (L,4))
    """

    def __init__(self, jitter: float = 1e-8, m: int = 10):
        self.jitter = float(jitter)
        self.m = int(m)

        # learned params
        self.beta: Optional[np.ndarray] = None  # (4,)
        self.phi: Optional[float] = None
        self.sigma2: Optional[float] = None
        self.ell: Optional[float] = None
        self.tau2: Optional[float] = None

    # ---------- fit with Y: (L,), X: (L,4) ----------

    def loglik_target(
        self,
        Y: np.ndarray,   # (L,)
        X: np.ndarray,   # (L,4)
        beta: np.ndarray,
        phi: float,
        sigma2: float,
        tau2: float,
    ) -> float:
        Y = np.asarray(Y, float).reshape(-1)
        X = np.asarray(X, float)
        L = Y.shape[0]
        assert X.shape == (L, 4)

        beta = np.asarray(beta, float).reshape(4)
        var = max(sigma2 + tau2, 1e-12)

        ll = 0.0
        mu0 = float(X[0] @ beta)
        e0 = Y[0] - mu0
        ll += -0.5 * (np.log(2 * np.pi * var) + (e0 * e0) / var)

        for t in range(1, L):
            mu_t = float(X[t] @ beta + phi * (Y[t-1] - (X[t-1] @ beta)))
            e = Y[t] - mu_t
            ll += -0.5 * (np.log(2 * np.pi * var) + (e * e) / var)

        return float(ll)

    def fit(
        self,
        X: np.ndarray,         # (L,4)
        Y: np.ndarray,         # (L,)
        init: Optional[dict] = None,
        verbose: bool = True,
    ):
        Y = np.asarray(Y, float).reshape(-1)
        X = np.asarray(X, float)
        L = Y.shape[0]
        assert X.shape == (L, 4)

        init = init or {}

        beta0, *_ = np.linalg.lstsq(X, Y, rcond=None)

        phi0 = float(init.get("phi", 0.5))

        # ell is not identifiable from target-only likelihood.
        # Use a sensible default for *normalized coordinates* (see _coords_from_features).
        ell0 = float(init.get("ell", 1.0))

        varY = float(np.var(Y) + 1e-6)
        sigma20 = float(init.get("sigma2", 0.8 * varY))
        tau20 = float(init.get("tau2", 0.2 * varY))

        def pack(beta, phi, sigma2, tau2, ell):
            z_phi = np.arctanh(np.clip(phi, -0.999, 0.999))
            return np.concatenate([
                beta.reshape(-1),
                np.array([z_phi, np.log(max(sigma2, 1e-12)), np.log(max(tau2, 1e-12)), np.log(max(ell, 1e-12))], float)
            ])

        def unpack(z):
            beta = z[:4]
            z_phi, z_sig, z_tau, z_ell = z[4:]
            phi = float(np.tanh(z_phi))
            sigma2 = float(np.exp(z_sig))
            tau2 = float(np.exp(z_tau))
            ell = float(np.exp(z_ell))
            return beta, phi, sigma2, tau2, ell

        z0 = pack(beta0, phi0, sigma20, tau20, ell0)

        def objective(z):
            beta, phi, sigma2, tau2, ell = unpack(z)
            ll = self.loglik_target(Y, X, beta, phi, sigma2, tau2)
            return -ll  # ell not used here

        res = minimize(objective, z0, method="L-BFGS-B")
        beta_hat, phi_hat, sigma2_hat, tau2_hat, ell_hat = unpack(res.x)

        self.beta = beta_hat
        self.phi = phi_hat
        self.sigma2 = sigma2_hat
        self.tau2 = tau2_hat
        self.ell = ell_hat

        if verbose:
            print("Optimization success:", res.success, res.message)
            print(f"beta={self.beta}")
            print(f"phi={self.phi:.4f}, sigma2={self.sigma2:.6g}, tau2={self.tau2:.6g}, ell={self.ell:.6g}")

        return {
            "phi": self.phi,
            "sigma2": self.sigma2,
            "tau2": self.tau2,
            "ell": self.ell,
            "neg_loglik": float(res.fun),
        }

    # ---------- internal distance computation from features ----------

    @staticmethod
    def _coords_from_features(X_tg: np.ndarray, X_nb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract coords from first 2 dims and normalize to make distances O(1).

        X_tg: (L,4) -> coords_tg: (1,2) normalized
        X_nb: (L,K,4) -> coords_nb: (K,2) normalized

        Normalization:
          - subtract neighbor centroid
          - divide by median neighbor radius
        """
        coords_tg = np.asarray(X_tg[0, :2], float).reshape(1, 2)  # (1,2)
        coords_nb = np.asarray(X_nb[0, :, :2], float)            # (K,2)

        center = coords_nb.mean(axis=0, keepdims=True)
        coords_nb = coords_nb - center
        coords_tg = coords_tg - center

        scale = np.median(np.linalg.norm(coords_nb, axis=1)) + 1e-12
        coords_nb = coords_nb / scale
        coords_tg = coords_tg / scale

        return coords_tg, coords_nb

    @staticmethod
    def _compute_distances(coords_tg: np.ndarray, coords_nb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dist_tn = np.linalg.norm(coords_nb - coords_tg, axis=1)  # (K,)
        diff = coords_nb[:, None, :] - coords_nb[None, :, :]     # (K,K,2)
        dist_nn = np.linalg.norm(diff, axis=2)                   # (K,K)
        return dist_tn, dist_nn

    # ---------- NNGP / Vecchia conditional coefficients (target | N_m) ----------

    def _compute_vecchia_target_coeffs(
        self,
        dist_tn: np.ndarray,
        dist_nn: np.ndarray,
        sigma2: float,
        ell: float,
        tau2: float,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        K = dist_tn.shape[0]
        m = min(self.m, K)
        if m <= 0:
            return np.array([], dtype=int), float(sigma2 + tau2), np.array([], dtype=float)

        idx = np.argsort(dist_tn)[:m]

        dist_mm = dist_nn[np.ix_(idx, idx)]
        dist_0m = dist_tn[idx].reshape(-1)

        Sigma_mm = exponential_cov(dist_mm, sigma2, ell)
        Sigma_m0 = exponential_cov(dist_0m, sigma2, ell).reshape(-1, 1)  # (m,1)

        C_mm = Sigma_mm + (tau2 + self.jitter) * np.eye(m)
        C_00 = sigma2 + tau2

        ones = np.ones((m, 1), float)

        # Build augmented OK system (m+1, m+1)
        A = np.block([
            [C_mm,  ones],
            [ones.T, np.zeros((1, 1), float)]
        ])
        b = np.vstack([Sigma_m0, np.ones((1, 1), float)])

        # Solve (indefinite); use solve with small jitter if needed
        try:
            sol = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            A[:m, :m] += 1e-6 * np.eye(m)
            sol = np.linalg.solve(A, b)

        w = sol[:m, 0]  # OK weights
        lam = sol[m, 0]

        # OK variance for residual field:
        # d = C00 - w^T C_m0 + lambda*1
        d = float(C_00 - (w @ Sigma_m0[:, 0]) + lam)
        d = max(d, 1e-12)

        return idx, d, w

    # ---------- predict all values given neighbors (NNGP style) ----------

    def predict(
        self,
        X_nb: np.ndarray,    # (L,K,4)
        Y_nb: np.ndarray,    # (L,K,1)
        X_tg: np.ndarray,    # (L,4)
        return_var: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        NNGP/Vecchia-style prediction:
          y_t(target) | y_t(N_m) uses only m nearest neighbors of target.

        Mean model:
          mu_nb[t,k] = X_nb[t,k] beta + phi (Y_nb[t-1,k] - X_nb[t-1,k] beta)
          mu_tg[t]   = X_tg[t] beta + phi (y_tg[t-1] - X_tg[t-1] beta)

        Since y_tg[t-1] is not available in the interface, we use previous predicted mean[t-1].
        """
        assert self.beta is not None and self.phi is not None
        assert self.sigma2 is not None and self.tau2 is not None and self.ell is not None

        X_nb = np.asarray(X_nb, float)
        Y_nb = np.asarray(Y_nb, float)
        X_tg = np.asarray(X_tg, float)

        L, K, p = X_nb.shape
        assert p == 4
        assert Y_nb.shape == (L, K, 1)
        assert X_tg.shape == (L, 4)

        beta, phi = self.beta, self.phi
        sigma2, tau2, ell = self.sigma2, self.tau2, self.ell

        # distances from embedded coordinates (normalized internally)
        coords_tg, coords_nb = self._coords_from_features(X_tg, X_nb)
        dist_tn, dist_nn = self._compute_distances(coords_tg, coords_nb)

        # Vecchia / NNGP coefficients for target | N_m (time-invariant)
        idx, d, a_vec = self._compute_vecchia_target_coeffs(dist_tn, dist_nn, sigma2, ell, tau2)
        m = len(idx)

        # neighbor mean uses their own previous observed neighbor y (available)
        mu_nb = np.zeros((L, K), float)
        mu_nb[0] = X_nb[0] @ beta
        for t in range(1, L):
            mu_nb[t] = (X_nb[t] @ beta) + phi * (Y_nb[t-1, :, 0] - (X_nb[t-1] @ beta))

        mean = np.zeros(L, float)
        var = np.zeros(L, float) if return_var else None

        # t=0
        mu_tg0 = float(X_tg[0] @ beta)
        if m > 0:
            rNm0 = (Y_nb[0, idx, 0] - mu_nb[0, idx])  # (m,)
            mean[0] = mu_tg0 + float(a_vec @ rNm0)
        else:
            mean[0] = mu_tg0

        if return_var:
            var[0] = d

        # t>=1 (recursive AR term uses previous predicted target value)
        for t in range(1, L):
            mu_tg = float(X_tg[t] @ beta + phi * (mean[t-1] - float(X_tg[t-1] @ beta)))
            if m > 0:
                rNm = (Y_nb[t, idx, 0] - mu_nb[t, idx])  # (m,)
                mean[t] = mu_tg + float(a_vec @ rNm)
            else:
                mean[t] = mu_tg

            if return_var:
                var[t] = d

        return mean, var
