import os, time, pickle
import numpy as np
import torch

from pykrige.uk import UniversalKriging

from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist

from gstools import Gaussian, krige


# ----------------------------
# Fast KED for ONE target point, MANY timesteps
# Equivalent system to GSTools ExtDrift:
# [ C   1   D^T ] [w]   [c]
# [ 1^T 0   0   ] [μ] = [1]
# [ D   0   0   ] [λ]   [d0]
#
# where:
#   C = model.covariance(dist(train,train)) + cond_err*I
#   c = model.covariance(dist(train,pred))
#   D = drift at training (q x N), d0 = drift at prediction (q x 1)
# ----------------------------
def fast_ked_time_series_gstools_best(
    gst_model,
    xs, x, lrain, nrain, valid=None,
    target_xy=None,
    cond_err="adaptive",        # "adaptive" or "nugget" or float
    cond_err_alpha=0.10,        # cond_err = alpha * median(var(y_t)) if adaptive
    eps=1e-10,

    drift_norm="zscore",        # "zscore" recommended
    drift_std_floor=1e-6,
    drift_clamp=5.0,

    rain_lag=0,
    rain_smooth=0,

    fallback="idw",
    idw_power=1.0,
):
    """
    Fast, stabilized KED that stays consistent with GSTools covariance model.

    xs:    (1,K,M) with x,y in cols 1,2 (like your OK helper)
    x:     (1,K,T) neighbor values
    lrain: (1,T)   drift at target
    nrain: (1,K,T) drift at neighbors
    """
    assert xs.shape[0] == 1, "batch size 1 only"

    lx_all = xs[0, :, 1].detach().cpu().numpy().astype(np.float64)
    ly_all = xs[0, :, 2].detach().cpu().numpy().astype(np.float64)

    Y_all = x[0].detach().cpu().numpy().astype(np.float64)        # (K,T)
    Dn_all = nrain[0].detach().cpu().numpy().astype(np.float64)   # (K,T)
    D0_all = lrain[0].detach().cpu().numpy().astype(np.float64)   # (T,)

    K, T = Y_all.shape

    if target_xy is None:
        tx, ty = 0.0, 0.0
    else:
        tx, ty = float(target_xy[0]), float(target_xy[1])

    # valid mask helper
    def mask_at(t):
        if valid is None:
            return np.ones(K, dtype=bool)
        v = valid.detach().cpu().numpy()
        if v.ndim == 3 and v.shape[-1] == 1:
            return v[0, :, 0].astype(bool)
        if v.ndim == 3 and v.shape[-1] == T:
            return v[0, :, t].astype(bool)
        return np.ones(K, dtype=bool)

    # optional smoothing/lagging of drift
    def smooth_1d(a, w):
        if w <= 1:
            return a
        out = np.empty_like(a)
        csum = np.cumsum(np.insert(a, 0, 0.0))
        for i in range(len(a)):
            j0 = max(0, i - w + 1)
            out[i] = (csum[i+1] - csum[j0]) / (i - j0 + 1)
        return out

    if rain_smooth and rain_smooth > 1:
        D0_all = smooth_1d(D0_all, int(rain_smooth))
        for k in range(K):
            Dn_all[k, :] = smooth_1d(Dn_all[k, :], int(rain_smooth))

    if rain_lag and rain_lag > 0:
        lag = int(rain_lag)
        D0_all = np.concatenate([np.full(lag, D0_all[0]), D0_all[:-lag]])
        Dn_all = np.concatenate([np.repeat(Dn_all[:, :1], lag, axis=1), Dn_all[:, :-lag]], axis=1)

    # IDW distances (fallback)
    d_to_target_all = np.sqrt((lx_all - tx) ** 2 + (ly_all - ty) ** 2)
    d_to_target_all = np.maximum(d_to_target_all, 1e-12)

    def idw_predict(y_vec, use_mask):
        d = d_to_target_all[use_mask]
        w = 1.0 / (d ** idw_power)
        w = w / (np.sum(w) + 1e-12)
        return float(np.sum(w * y_vec[use_mask]))

    pts_all = np.column_stack([lx_all, ly_all])

    # adaptive cond_err
    if cond_err == "adaptive":
        vts = np.var(Y_all, axis=0)
        cond_err_use = float(cond_err_alpha * max(np.median(vts), 1e-6))
    elif cond_err == "nugget":
        cond_err_use = float(getattr(gst_model, "nugget", 0.0))
    else:
        cond_err_use = float(cond_err)

    # build cache for given mask
    def build_cache(use_mask):
        idxs = np.where(use_mask)[0]
        N = idxs.size
        pts = pts_all[idxs]

        Dnn = cdist(pts, pts)
        C = np.asarray(gst_model.covariance(Dnn), float)
        C[np.diag_indices(N)] += (cond_err_use + eps)

        cf = cho_factor(C, lower=True, check_finite=False)

        Dpn = cdist(pts, np.array([[tx, ty]], dtype=np.float64)).reshape(-1)
        cvec = np.asarray(gst_model.covariance(Dpn), float)

        b = cho_solve(cf, cvec.reshape(-1, 1), check_finite=False).reshape(-1)  # C^{-1}c
        ones = np.ones(N, dtype=np.float64)
        a = cho_solve(cf, ones.reshape(-1, 1), check_finite=False).reshape(-1)  # C^{-1}1

        s11 = float(ones @ a)
        uTb = float(ones @ b)
        rhs1 = uTb - 1.0
        return idxs, cf, b, a, s11, rhs1

    use0 = mask_at(0)
    if use0.sum() < 2:
        return torch.full((1, T), float("nan"), device=x.device)

    idxs, cf, b, a, s11, rhs1 = build_cache(use0)
    use_prev = use0

    yhat = np.empty(T, dtype=np.float64)

    for t in range(T):
        use = mask_at(t)
        if use.sum() < 2:
            yhat[t] = idw_predict(Y_all[:, t], np.ones(K, dtype=bool)) if fallback == "idw" else float(np.nanmean(Y_all[:, t]))
            continue

        if use.sum() != use_prev.sum() or not np.all(use == use_prev):
            idxs, cf, b, a, s11, rhs1 = build_cache(use)
            use_prev = use

        yt = Y_all[idxs, t]
        dt = Dn_all[idxs, t]
        d0 = float(D0_all[t])

        # drift normalization
        if drift_norm == "zscore":
            m = float(np.mean(dt))
            s = float(np.std(dt))
            if s < drift_std_floor:
                mask_tmp = np.zeros(K, dtype=bool); mask_tmp[idxs] = True
                yhat[t] = idw_predict(Y_all[:, t], mask_tmp) if fallback == "idw" else float(np.mean(yt))
                continue
            dt = (dt - m) / (s + 1e-12)
            d0 = (d0 - m) / (s + 1e-12)
            if drift_clamp is not None:
                cval = float(drift_clamp)
                dt = np.clip(dt, -cval, cval)
                d0 = float(np.clip(d0, -cval, cval))
        else:
            if np.std(dt) < drift_std_floor:
                mask_tmp = np.zeros(K, dtype=bool); mask_tmp[idxs] = True
                yhat[t] = idw_predict(Y_all[:, t], mask_tmp) if fallback == "idw" else float(np.mean(yt))
                continue

        # E = C^{-1} d
        E = cho_solve(cf, dt.reshape(-1, 1), check_finite=False).reshape(-1)

        s12 = float(np.sum(E))
        s22 = float(np.dot(dt, E))
        rhs2 = float(np.dot(dt, b) - d0)

        det = s11 * s22 - s12 * s12
        if abs(det) < eps:
            mask_tmp = np.zeros(K, dtype=bool); mask_tmp[idxs] = True
            yhat[t] = idw_predict(Y_all[:, t], mask_tmp) if fallback == "idw" else float(np.mean(yt))
            continue

        mu = (rhs1 * s22 - rhs2 * s12) / det
        lam = (-rhs1 * s12 + rhs2 * s11) / det

        yhat[t] = float(np.dot(b, yt) - mu * np.dot(a, yt) - lam * np.dot(E, yt))

    return torch.tensor(yhat, device=x.device).unsqueeze(0)


def compare_fast_ked_vs_gstools(
    neigh_xy,          # (N,2) or (N,D)
    target_xy,         # (2,)  or (D,)
    Y,                 # (N,T)   neighbor values over time
    Dtrain,            # (N,T)   neighbor drift over time
    Dpred,             # (T,)    target drift over time
    var=2.0,
    len_scale=4.0,
    nugget=0.0,
    cond_err="nugget",
    eps=1e-10,
):
    """
    Compare vectorized fast_ked_time_series_gstools (cached-Cholesky KED)
    against gstools.krige.ExtDrift evaluated per-timestep.

    Returns:
        errs: (T,) absolute errors per timestep
        z_fast: (T,) fast predictions
        z_gs: (T,) gstools predictions
    """
    neigh_xy = np.asarray(neigh_xy, float)
    target_xy = np.asarray(target_xy, float).reshape(-1)

    if neigh_xy.ndim == 1:
        neigh_xy = neigh_xy.reshape(-1, 1)
    D = neigh_xy.shape[1]
    assert target_xy.shape[0] == D

    Y = np.asarray(Y, float)
    Dtrain = np.asarray(Dtrain, float)
    Dpred = np.asarray(Dpred, float).reshape(-1)

    N, T = Y.shape
    assert Dtrain.shape == (N, T)
    assert Dpred.shape == (T,)

    # GSTools model
    model = Gaussian(dim=D, var=var, len_scale=len_scale, nugget=nugget)

    # --- fast (vectorized over time) ---
    z_fast = fast_ked_time_series_gstools(
        gst_model=model,
        X_train=neigh_xy,
        y_train_ts=Y,
        drift_train_ts=Dtrain,
        X_pred=target_xy,
        drift_pred_ts=Dpred,
        cond_err=cond_err,
        eps=eps,
    )
    z_fast = np.asarray(z_fast, float).reshape(-1)

    # --- gstools per timestep ---
    pos_train = [neigh_xy[:, j] for j in range(D)]
    pos_pred = [np.array([target_xy[j]]) for j in range(D)]

    z_gs = np.empty(T, dtype=float)
    for t in range(T):
        krig_t = krige.ExtDrift(
            model,
            pos_train,
            Y[:, t],
            Dtrain[:, t],
            cond_err=cond_err,   # match diagonal handling
        )
        pred_t = krig_t(pos_pred, ext_drift=np.array([Dpred[t]]))
        z_gs[t] = float(np.asarray(pred_t).reshape(-1)[0])

    errs = np.abs(z_gs - z_fast)
    return errs, z_fast, z_gs


# ---------------------------
# Quick runnable test
# ---------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 20
    T = 50

    neigh_xy = rng.uniform(-5, 5, size=(N, 2))
    target_xy = np.array([0.5, -1.0])

    Y = rng.normal(size=(N, T))
    Dtrain = rng.normal(size=(N, T))
    Dpred = rng.normal(size=(T,))

    errs, z_fast, z_gs = compare_fast_ked_vs_gstools(
        neigh_xy, target_xy, Y, Dtrain, Dpred,
        var=2.0, len_scale=4.0, nugget=0.0,
        cond_err="nugget",
        eps=1e-10,
    )

    print("max abs error:", float(errs.max()))
    print("mean abs error:", float(errs.mean()))