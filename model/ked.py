import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve
from typing import Tuple, Optional, Callable


def _solve_ked_block(C, F, c, f0, eps=1e-8):
    """
    Solve:
        [C  F] [w] = [c]
        [F' 0] [λ]   [f0]
    using Schur complement; needs C SPD.

    C: (n,n) SPD
    F: (n,q)
    c: (n,)   covariance between pred and training
    f0:(q,)   drift basis at pred
    returns w (n,), lam (q,)
    """
    n = C.shape[0]
    q = F.shape[1]

    # Cholesky on C (SPD). Add diagonal eps if needed.
    C_reg = C + eps * np.eye(n)
    cf = cho_factor(C_reg, lower=True, check_finite=False)

    # Compute C^{-1}F and C^{-1}c
    CiF = cho_solve(cf, F, check_finite=False)      # (n,q)
    Cic = cho_solve(cf, c.reshape(-1,1), check_finite=False).reshape(-1)  # (n,)

    # Schur complement: S = F' C^{-1} F (q,q)
    S = F.T @ CiF

    # rhs for lambda: F' C^{-1} c - f0
    rhs = F.T @ Cic - f0

    # Solve S λ = rhs (S should be SPD if F has full column rank)
    # Add tiny ridge for numerical stability
    S_reg = S + eps * np.eye(q)
    lam = np.linalg.solve(S_reg, rhs)

    # w = C^{-1}(c - F λ)
    w = Cic - CiF @ lam
    return w, lam


def _ked_variance(C00, w, c, lam, f0):
    # σ^2 = C00 - w'c + λ'f0
    return float(C00 - (w @ c) + (lam @ f0))


class KrigingExternalDrift:
    """
    Kriging with External Drift (KED) / Universal Kriging with external covariates.

    Uses covariance formulation:
        [ C   F ] [ w ] = [ c ]
        [ F'  0 ] [ λ ]   [ f0]
    Prediction:
        yhat = w' y
    Variance:
        s2 = C(0) - w' c + λ' f0
    """

    def __init__(
        self,
        variogram_model: str = "spherical",
        variogram_params: Optional[dict] = None,
        eps: float = 1e-8,
    ):
        self.variogram_model = variogram_model
        self.variogram_params = variogram_params or {"sill": 1.0, "range": 10.0, "nugget": 0.0}
        self.eps = float(eps)

        self.X_train = None
        self.y_train = None
        self.drift_train = None
        self.n_samples = None
        self.n_drift = None

    # -----------------------
    # Variogram & Covariance
    # -----------------------

    def _variogram(self, h: np.ndarray) -> np.ndarray:
        sill = float(self.variogram_params.get("sill", 1.0))
        range_ = float(self.variogram_params.get("range", 10.0))
        nugget = float(self.variogram_params.get("nugget", 0.0))

        if callable(self.variogram_model):
            return self.variogram_model(h, **self.variogram_params)

        if self.variogram_model == "spherical":
            # semivariogram; often gamma(0)=0 and nugget is a discontinuity;
            # here we keep gamma(0)=0 and represent nugget in covariance diagonal.
            hr = np.clip(h / range_, 0.0, 1.0)
            core = 1.5 * hr - 0.5 * hr**3
            gamma = (sill - nugget) * np.where(h <= range_, core, 1.0)
            # Do NOT add nugget here; handle nugget in covariance diagonal.
            return gamma

        if self.variogram_model == "exponential":
            return (sill - nugget) * (1.0 - np.exp(-3.0 * h / range_))

        if self.variogram_model == "gaussian":
            return (sill - nugget) * (1.0 - np.exp(-3.0 * (h / range_) ** 2))

        raise ValueError(f"Unknown variogram model: {self.variogram_model}")

    def _covariance(self, h: np.ndarray, add_nugget_on_diag: bool = False) -> np.ndarray:
        """
        Convert variogram model to covariance:
            C(h) = (sill - nugget) - gamma(h)   for h>0
            C(0) = sill
        Nugget is handled as diagonal-only term if add_nugget_on_diag=True.
        """
        sill = float(self.variogram_params.get("sill", 1.0))
        nugget = float(self.variogram_params.get("nugget", 0.0))

        gamma = self._variogram(h)
        C = (sill - nugget) - gamma

        # enforce C(0)=sill on exact zeros (numerical)
        if np.ndim(h) > 0:
            C = np.array(C, dtype=float, copy=True)
            C[h == 0] = sill - nugget  # base covariance without nugget
        else:
            C = float(C)
            if h == 0:
                C = sill - nugget

        if add_nugget_on_diag:
            # nugget as measurement error / microscale on diagonal
            if C.ndim == 2:
                C = C + nugget * np.eye(C.shape[0])
            else:
                # scalar variance at a point includes nugget
                C = C + nugget

        return C

    # -----------------------
    # Fit
    # -----------------------

    def fit(self, X: np.ndarray, y: np.ndarray, drift: np.ndarray) -> "KrigingExternalDrift":
        self.X_train = np.asarray(X, float)
        self.y_train = np.asarray(y, float).reshape(-1)

        drift_array = np.asarray(drift, float)
        if drift_array.ndim == 1:
            drift_array = drift_array.reshape(-1, 1)
        self.drift_train = drift_array

        self.n_samples = self.y_train.shape[0]
        self.n_drift = self.drift_train.shape[1]

        if self.X_train.shape[0] != self.n_samples:
            raise ValueError("X and y must have same number of samples")
        if self.drift_train.shape[0] != self.n_samples:
            raise ValueError("drift and y must have same number of samples")

        return self

    # -----------------------
    # Build system (cov form)
    # -----------------------

    def _build_system(
        self,
        X_pred: np.ndarray,
        drift_pred: np.ndarray,
        nmax: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          A: (M,M) system matrix
          rhs: (M, n_pred) RHS matrix (each column is rhs for a pred point)
          idx_sets: (n_pred, n_used) indices used per pred (or None if global)
          f0_all: (n_pred, q) drift basis at pred points
        """
        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction")

        X_pred = np.asarray(X_pred, float)
        if X_pred.ndim == 1:
            X_pred = X_pred.reshape(1, -1)

        drift_pred = np.asarray(drift_pred, float)
        if drift_pred.ndim == 1:
            drift_pred = drift_pred.reshape(-1, 1)
        if drift_pred.shape[0] != X_pred.shape[0]:
            # allow transpose if user passed (q, n_pred)
            if drift_pred.shape[1] == X_pred.shape[0]:
                drift_pred = drift_pred.T
            else:
                raise ValueError("drift_pred must align with X_pred rows")

        n_pred = X_pred.shape[0]

        # Build drift basis: [1, drift_vars...]
        F_train = np.column_stack([np.ones(self.n_samples), self.drift_train])  # (n, q)
        f0_all = np.column_stack([np.ones(n_pred), drift_pred])                 # (n_pred, q)
        q = F_train.shape[1]

        # Global system (all points) or local per pred
        if nmax is None or nmax >= self.n_samples:
            # distances among training points
            D = cdist(self.X_train, self.X_train)
            C = self._covariance(D, add_nugget_on_diag=True)  # include nugget on diag
            C = C + self.eps * np.eye(self.n_samples)

            # system A
            M = self.n_samples + q
            A = np.zeros((M, M), float)
            A[:self.n_samples, :self.n_samples] = C
            A[:self.n_samples, self.n_samples:] = F_train
            A[self.n_samples:, :self.n_samples] = F_train.T
            # bottom-right is zeros

            # RHS for all pred: [c; f0]
            Dp = cdist(X_pred, self.X_train)                         # (n_pred, n)
            c = self._covariance(Dp, add_nugget_on_diag=False)        # (n_pred, n)
            rhs = np.zeros((M, n_pred), float)
            rhs[:self.n_samples, :] = c.T
            rhs[self.n_samples:, :] = f0_all.T

            return A, rhs, None, f0_all

        # Local KED: build per prediction point (nearest nmax)
        # We return a dummy A/rhs; prediction will handle per-point
        return None, None, None, f0_all

    # -----------------------
    # Predict
    # -----------------------

    def predict(
        self,
        X_pred: np.ndarray,
        drift_pred: np.ndarray,
        return_std: bool = False,
        nmax: Optional[int] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        nmax: if set, use only the nearest nmax training points for each prediction point.
        """
        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction")

        X_pred = np.asarray(X_pred, float)
        if X_pred.ndim == 1:
            X_pred = X_pred.reshape(1, -1)

        drift_pred = np.asarray(drift_pred, float)
        if drift_pred.ndim == 1:
            drift_pred = drift_pred.reshape(-1, 1)
        if drift_pred.shape[0] != X_pred.shape[0]:
            if drift_pred.shape[1] == X_pred.shape[0]:
                drift_pred = drift_pred.T
            else:
                raise ValueError("drift_pred must align with X_pred rows")

        n_pred = X_pred.shape[0]
        yhat = np.zeros(n_pred, float)
        sigma = np.zeros(n_pred, float) if return_std else None

        # Drift basis
        F_train = np.column_stack([np.ones(self.n_samples), self.drift_train])  # (n,q)
        # q = F_train.shape[1]

        # Covariance among training
        D = cdist(self.X_train, self.X_train)
        C = self._covariance(D, add_nugget_on_diag=True)
        C = C + self.eps * np.eye(self.n_samples)

        # For each pred point:
        Dp = cdist(X_pred, self.X_train)
        c_all = self._covariance(Dp, add_nugget_on_diag=False)  # (n_pred, n)
        f0_all = np.column_stack([np.ones(n_pred), drift_pred]) # (n_pred, q)

        yhat = np.zeros(n_pred)
        sigma = np.zeros(n_pred) if return_std else None

        sill = float(self.variogram_params.get("sill", 1.0))
        C00 = sill  # process variance; if you want observation variance add nugget here

        for i in range(n_pred):
            c = c_all[i]       # (n,)
            f0 = f0_all[i]     # (q,)

            w, lam = _solve_ked_block(C, F_train, c, f0, eps=self.eps)

            yhat[i] = float(w @ self.y_train)

            if return_std:
                s2 = _ked_variance(C00, w, c, lam, f0)
                sigma[i] = np.sqrt(max(s2, 0.0))

        return yhat, sigma
