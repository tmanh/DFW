import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve


def _corr(h, model="gaussian", range_=4.0):
    h = np.asarray(h, float)
    range_ = float(max(range_, 1e-12))
    if model == "gaussian":
        return np.exp(-3.0 * (h / range_) ** 2)
    if model == "exponential":
        return np.exp(-3.0 * h / range_)
    if model == "spherical":
        hr = h / range_
        return np.where(h <= range_, 1.0 - (1.5 * hr - 0.5 * hr**3), 0.0)
    raise ValueError(f"Unknown corr model: {model}")


class CachedOKCKIntrinsic:
    """
    Ordinary cokriging (primary + secondary) using an Intrinsic Coregionalization Model (ICM):
        C11(h) = sill1 * rho(h)
        C22(h) = sill2 * rho(h)
        C12(h) = r12 * sqrt(sill1*sill2) * rho(h)
    (guaranteed PSD if |r12|<=1 and sills>=0)

    OKCK constraints for unknown means per variable:
        sum weights (primary data)   = 1
        sum weights (secondary data) = 0
    i.e. f0 = [1, 0]^T

    This class caches weights for ONE prediction location x0, for repeated timesteps.
    """

    def __init__(
        self,
        corr_model="gaussian",
        range_=4.0,
        sill1=2.0,
        sill2=1.0,
        r12=0.5,
        nugget1=0.0,
        nugget2=0.0,
        eps=1e-10,
        use_collocated_secondary=True,
    ):
        self.corr_model = corr_model
        self.range_ = float(range_)
        self.sill1 = float(sill1)
        self.sill2 = float(sill2)
        self.r12 = float(r12)
        self.nugget1 = float(nugget1)
        self.nugget2 = float(nugget2)
        self.eps = float(eps)
        self.use_collocated_secondary = bool(use_collocated_secondary)

        self.w1 = None   # weights on primary obs
        self.w2 = None   # weights on secondary obs
        self.n1 = None
        self.n2 = None

    def fit_geometry(self, X1, X2, x0):
        """
        X1: (n1,D) primary sample coords
        X2: (n2,D) secondary sample coords (can include x0 if collocated is enabled)
        x0: (D,) target coord
        """
        X1 = np.asarray(X1, float)
        X2 = np.asarray(X2, float)
        x0 = np.asarray(x0, float).reshape(1, -1)

        if X1.ndim == 1:
            X1 = X1.reshape(-1, 1)
        if X2.ndim == 1:
            X2 = X2.reshape(-1, 1)

        n1, D = X1.shape
        n2, D2 = X2.shape
        if D2 != D:
            raise ValueError("X1 and X2 must have same dimensionality")

        if not (-1.0 <= self.r12 <= 1.0):
            raise ValueError("r12 must be within [-1, 1] for PSD ICM")

        self.n1, self.n2 = n1, n2

        # Distances
        D11 = cdist(X1, X1)
        D22 = cdist(X2, X2)
        D12 = cdist(X1, X2)

        rho11 = _corr(D11, self.corr_model, self.range_)
        rho22 = _corr(D22, self.corr_model, self.range_)
        rho12 = _corr(D12, self.corr_model, self.range_)

        # Cov blocks
        C11 = self.sill1 * rho11
        C22 = self.sill2 * rho22
        C12 = (self.r12 * np.sqrt(self.sill1 * self.sill2)) * rho12  # (n1,n2)
        C21 = C12.T

        # Add nuggets on diagonals
        if self.nugget1 > 0:
            C11 = C11.copy()
            C11[np.diag_indices_from(C11)] += self.nugget1
        if self.nugget2 > 0:
            C22 = C22.copy()
            C22[np.diag_indices_from(C22)] += self.nugget2

        # Big C
        N = n1 + n2
        C = np.zeros((N, N), float)
        C[:n1, :n1] = C11
        C[:n1, n1:] = C12
        C[n1:, :n1] = C21
        C[n1:, n1:] = C22

        # Constraints: one per variable (OKCK)
        # F = [1 on primary obs, 0 on secondary obs; 0 on primary obs, 1 on secondary obs]
        F = np.zeros((N, 2), float)
        F[:n1, 0] = 1.0
        F[n1:, 1] = 1.0
        f0 = np.array([1.0, 0.0], float)  # predict primary mean only

        # c vector: cov between all obs and primary at x0
        d10 = cdist(X1, x0)  # (n1,1)
        d20 = cdist(X2, x0)  # (n2,1)

        c1 = self.sill1 * _corr(d10, self.corr_model, self.range_).reshape(-1)  # (n1,)
        c2 = (self.r12 * np.sqrt(self.sill1 * self.sill2)) * _corr(d20, self.corr_model, self.range_).reshape(-1)  # (n2,)
        c = np.concatenate([c1, c2], axis=0)  # (N,)

        # Solve [C F; F' 0][w;lam]=[c;f0] via Schur on C (SPD)
        C = C + self.eps * np.eye(N)
        cf = cho_factor(C, lower=True, check_finite=False)

        CiF = cho_solve(cf, F, check_finite=False)          # (N,2)
        Cic = cho_solve(cf, c.reshape(-1, 1), check_finite=False).reshape(-1)  # (N,)

        S = F.T @ CiF                                       # (2,2)
        rhs = F.T @ Cic - f0                                # (2,)
        S = S + self.eps * np.eye(2)
        lam = np.linalg.solve(S, rhs)                       # (2,)

        w = Cic - CiF @ lam                                 # (N,)
        self.w1 = w[:n1].copy()
        self.w2 = w[n1:].copy()
        return self

    def predict_timeseries(self, y1_all, y2_all):
        """
        y1_all: (n1,T) primary values
        y2_all: (n2,T) secondary values (must match X2 used in fit_geometry)
        returns: (T,)
        """
        if self.w1 is None:
            raise ValueError("Call fit_geometry first")

        y1_all = np.asarray(y1_all, float)
        y2_all = np.asarray(y2_all, float)
        if y1_all.shape[0] != self.n1 or y2_all.shape[0] != self.n2:
            raise ValueError("y shapes must match fitted geometry")

        # yhat[t] = w1@y1[:,t] + w2@y2[:,t]
        return (self.w1 @ y1_all) + (self.w2 @ y2_all)


def fast_okck_time_series(
    nxs,            # (K,D) neighbor coords
    xs,             # (D,)  target coord
    nby,            # (K,T) neighbor water level
    nrain,          # (K,T) neighbor rainfall
    lrain,          # (T,)  target rainfall (optional collocated secondary)
    *,
    corr_model="gaussian",
    range_=4.0,
    sill1=2.0,
    sill2=1.0,
    r12=0.5,
    nugget1=0.0,
    nugget2=0.0,
    eps=1e-10,
    use_collocated_secondary=True,
):
    nxs = np.asarray(nxs, float)
    xs = np.asarray(xs, float).reshape(-1)
    nby = np.asarray(nby, float)
    nrain = np.asarray(nrain, float)
    lrain = np.asarray(lrain, float).reshape(-1)

    K, D = nxs.shape
    T = nby.shape[1]

    # X1 = primary coords (neighbors)
    X1 = nxs

    # X2 = secondary coords (neighbors + optional collocated at target)
    if use_collocated_secondary:
        X2 = np.vstack([nxs, xs.reshape(1, -1)])  # (K+1,D)
        y2 = np.vstack([nrain, lrain.reshape(1, -1)])  # (K+1,T)
    else:
        X2 = nxs
        y2 = nrain

    okck = CachedOKCKIntrinsic(
        corr_model=corr_model,
        range_=range_,
        sill1=sill1,
        sill2=sill2,
        r12=r12,
        nugget1=nugget1,
        nugget2=nugget2,
        eps=eps,
        use_collocated_secondary=use_collocated_secondary,
    ).fit_geometry(X1=X1, X2=X2, x0=xs)

    yhat = okck.predict_timeseries(y1_all=nby, y2_all=y2)  # (T,)
    return yhat
