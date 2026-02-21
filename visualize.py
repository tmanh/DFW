import os
import logging
import pickle
import numpy as np

from scipy.signal import find_peaks
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt

import folium
from statsmodels.stats.multitest import multipletests


if os.path.exists('log-test-all.txt'):
    os.remove('log-test-all.txt')
logging.basicConfig(filename="log-test-all.txt", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def station_bin_rmse(station_result, lo=0.6, hi=1.2):
    """
    station_result: result[k] for ONE method, ONE station id.
    Uses abs(tgts) bin [lo, hi) and computes RMSE over that subset.
    Returns (rmse, n_points). If no data -> (np.nan, 0).
    """
    if not station_result.get('corr'):
        return np.nan, 0

    tgts = np.asarray(station_result['tgts'][0], dtype=float).reshape(-1)
    outs = np.asarray(station_result['outs'][0], dtype=float).reshape(-1)

    # filter extremes similar to your collect_bin_means
    mask0 = (tgts > -4) & (tgts < 4)
    tgts = tgts[mask0]
    outs = outs[mask0]

    abs_t = np.abs(tgts)
    mask = (abs_t >= lo) & (abs_t < hi)
    if mask.sum() == 0:
        return np.nan, 0

    se = (outs[mask] - tgts[mask]) ** 2
    rmse = float(np.sqrt(np.mean(se)))
    return rmse, int(mask.sum())


def _extract_latlon(loc):
    """
    loc can be (2,), (1,2), etc. Returns (lat, lon).
    NOTE: if your loc is (lon, lat) swap below.
    """
    loc = np.asarray(loc).reshape(-1)
    if loc.size < 2:
        return None
    # Most common: [lat, lon]. If yours is [lon, lat], swap these two lines.
    lat, lon = float(loc[0]), float(loc[1])
    return lat, lon


def _add_halo_circle(m, lat, lon, color, text, r=8):
    # colored inner dot with black outline
    folium.CircleMarker(
        location=[lat, lon],
        radius=r,
        color="#111111",
        weight=2,
        fill=True,
        fill_color=color,
        fill_opacity=1.0,
        opacity=1.0,
        tooltip=folium.Tooltip(text, sticky=True),
    ).add_to(m)


def add_3bin_legend(m, title, bins, colors):
    items = "".join(
        f"""
        <div style="display:flex;align-items:center;margin:4px 0;">
          <div style="width:14px;height:14px;background:{c};border:1px solid #111;margin-right:8px;"></div>
          <div>{lab}</div>
        </div>
        """
        for lab, c in zip(bins, colors)
    )

    html = f"""
    <div style="
      position: fixed;
      top: 20px; right: 20px;
      z-index: 9999;
      background: rgba(255,255,255,0.95);
      padding: 10px 12px;
      border: 2px solid #333;
      border-radius: 6px;
      font-size: 13px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    ">
      <div style="font-weight:700;margin-bottom:6px;">{title}</div>
      {items}
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))


def station_bin_rmse_and_relative(station_result, lo=0.6, hi=1.2, eps=1e-8):
    """
    Returns:
      rmse_abs (float), rmse_rel (float), npts (int), mean_abs_tgt (float)

    rmse_rel = rmse_abs / mean(|tgt|) within the selected bin.
    """
    # collect like you already do in collect_bin_means:
    # station_result['tgts'][0], station_result['outs'][0], station_result['loss'][0] etc.
    if not station_result.get("tgts") or not station_result.get("outs"):
        return np.nan, np.nan, 0, np.nan

    tgts = np.asarray(station_result["tgts"][0], float)
    outs = np.asarray(station_result["outs"][0], float)

    # sanity
    n = min(len(tgts), len(outs))
    tgts = tgts[:n]
    outs = outs[:n]

    # filter extreme if you want (match your pipeline)
    mask_ok = (tgts > -4) & (tgts < 4)
    tgts = tgts[mask_ok]
    outs = outs[mask_ok]

    # bin by ABS target change
    at = np.abs(tgts)
    mask_bin = (at >= lo) & (at < hi)
    if mask_bin.sum() == 0:
        return np.nan, np.nan, 0, np.nan

    e2 = (outs[mask_bin] - tgts[mask_bin]) ** 2
    rmse_abs = float(np.sqrt(np.mean(e2)))

    mean_abs_tgt = float(np.mean(at[mask_bin]))
    rmse_rel = float(rmse_abs / (mean_abs_tgt + eps))

    return rmse_abs, rmse_rel, int(mask_bin.sum()), mean_abs_tgt


def print_metric_stats(rows, idx, name):
    v = np.array([r[idx] for r in rows], float)
    mask = np.isfinite(v)
    if mask.sum() == 0:
        print(f"{name}: no finite values")
        return
    x = v[mask]
    print(f"\n{name} over stations (finite: {mask.sum()} / {len(rows)})")
    print(f"  mean: {np.mean(x):.4f}, std: {np.std(x):.4f}, min: {np.min(x):.4f}, median: {np.median(x):.4f}, max: {np.max(x):.4f}")


def print_rmse_rel_counts(rows, lo, hi, bins=(0.2, 0.4, 0.6, 0.8, 1.0), show_ids=False, max_ids=20):
    """
    rows entries like: (k, lat, lon, rmse_abs, rmse_rel, npts, mean_abs_tgt)

    bins define intervals:
      <bins[0], [bins[0],bins[1]), ..., [bins[-2],bins[-1}), >=bins[-1]
    """
    # extract
    ids   = np.array([r[0] for r in rows], object)
    rel   = np.array([r[4] for r in rows], float)
    npts  = np.array([r[5] for r in rows], int)

    mask = np.isfinite(rel) & (npts > 0)
    ids, rel = ids[mask], rel[mask]

    if rel.size == 0:
        print(f"No stations have samples in |target| bin [{lo},{hi}) (RMSE_rel finite)")
        return

    bins = np.asarray(bins, float)

    def _print_bucket(name, m):
        c = int(np.sum(m))
        print(f"  {name}: {c}")
        if show_ids and c > 0:
            listed = ids[m][:max_ids]
            more = c - len(listed)
            print("    ids:", ", ".join(map(str, listed)) + (f" ... (+{more} more)" if more > 0 else ""))

    print(f"\nCounts of stations by RMSE_rel (RMSE / mean|Δ|) within |target| bin [{lo},{hi})  [N={rel.size}]")

    _print_bucket(f"< {bins[0]:.2f}", rel < bins[0])

    for a, b in zip(bins[:-1], bins[1:]):
        _print_bucket(f"[{a:.2f}, {b:.2f})", (rel >= a) & (rel < b))

    _print_bucket(f"≥ {bins[-1]:.2f}", rel >= bins[-1])


def make_station_error_map(
    results_dict,
    method_name=None,
    lo=0.6, hi=1.2,
    out_html="station_rmse_map_0p6_1p2.html",
    tiles="CartoDB positron",
    color_by="abs",   # "abs" or "rel"
):
    if method_name is None:
        method_name = list(results_dict.keys())[0]
    result = results_dict[method_name]

    rows = []
    for k in result.keys():
        if 'loc' not in result[k]:
            continue
        ll = _extract_latlon(result[k]['loc'])
        if ll is None:
            continue
        lat, lon = ll

        rmse_abs, rmse_rel, npts, mean_abs_tgt = station_bin_rmse_and_relative(result[k], lo=lo, hi=hi)
        rows.append((k, lat, lon, rmse_abs, rmse_rel, npts, mean_abs_tgt))

    if len(rows) == 0:
        raise RuntimeError("No stations with loc found in results for this method.")

    lats = np.array([r[1] for r in rows], float)
    lons = np.array([r[2] for r in rows], float)

    m = folium.Map(location=[float(np.mean(lats)), float(np.mean(lons))], zoom_start=9, tiles=tiles)

    # ---- pick metric ----
    if color_by not in ["abs", "rel"]:
        raise ValueError("color_by must be 'abs' or 'rel'")
    metric_idx = 3 if color_by == "abs" else 4
    metric_name = "RMSE" if color_by == "abs" else "Relative RMSE (RMSE / mean|Δ|)"

    vals = np.array([r[metric_idx] for r in rows], float)

    # ---- thresholds ----
    # For abs you wanted: 0-0.2-0.4+
    # For relative, typical bins are e.g. <0.25, <0.5, >=0.5 (change if you want)
    if color_by == "abs":
        t1, t2 = 0.2, 0.4
        labels = ["0–0.2 (low)", "0.2–0.4 (medium)", "≥0.4 (high)"]
    else:
        t1, t2 = 0.25, 0.50
        labels = [f"0–{t1:.2f} (low)", f"{t1:.2f}–{t2:.2f} (medium)", f"≥{t2:.2f} (high)"]

    LOW, MID, HIGH, NAN = "#2171b5", "#fec44f", "#d73027", "#9e9e9e"

    def val_to_color(v):
        if not np.isfinite(v):
            return NAN
        if v < t1:
            return LOW
        elif v < t2:
            return MID
        else:
            return HIGH

    add_3bin_legend(
        m,
        title=f"{method_name} — {metric_name} in |target| bin [{lo}, {hi})",
        bins=labels,
        colors=[LOW, MID, HIGH],
    )

    # ---- print station counts per bracket ----
    finite = np.isfinite(vals)
    n_nan = int(np.sum(~finite))
    n_low = int(np.sum(finite & (vals < t1)))
    n_mid = int(np.sum(finite & (vals >= t1) & (vals < t2)))
    n_high = int(np.sum(finite & (vals >= t2)))

    print(f"[{method_name}] Color-by={color_by} for |target| in [{lo}, {hi}):")
    print(f"  low:   {n_low}")
    print(f"  mid:   {n_mid}")
    print(f"  high:  {n_high}")
    print(f"  NaN:   {n_nan}")
    print(f"  total: {len(rows)}")

    # ---- plot ----
    for (k, lat, lon, rmse_abs, rmse_rel, npts, mean_abs_tgt) in rows:
        v = rmse_abs if color_by == "abs" else rmse_rel
        color = val_to_color(v)

        if np.isfinite(rmse_abs):
            txt = (
                f"Station: {k}"
                f"<br>RMSE_abs[{lo},{hi}): {rmse_abs:.4f}"
                f"<br>mean|Δ|[{lo},{hi}): {mean_abs_tgt:.4f}"
                f"<br>RMSE_rel: {rmse_rel:.3f}"
                f"<br>N points: {npts}"
            )
        else:
            txt = f"Station: {k}<br>No samples in bin [{lo},{hi})"

        _add_halo_circle(m, lat, lon, color, txt, r=4)

    print_rmse_rel_counts(rows, lo, hi, bins=(0.1, 0.2, 0.3, 0.5, 1.0), show_ids=False)

    m.save(out_html)
    print(f"Saved: {out_html}")
    return out_html


def collect_bin_means(result):
    all_gain = []
    all_outs = []
    all_tgts = []
    all_loss = []
    all_corr = []
    for k in result.keys():
        if result[k]['corr']:
            all_corr.append(result[k]['corr'][0])
            all_gain.extend(result[k]['gain'][0])
            all_tgts.extend(result[k]['tgts'][0])
            all_outs.extend(result[k]['outs'][0])
            all_loss.extend(result[k]['loss'][0].tolist())

    all_gain = np.array(all_gain)
    all_outs = np.array(all_outs)
    all_tgts = np.array(all_tgts)
    all_loss = np.array(all_loss)
    all_corr = np.array(all_corr)

    mask = (all_tgts > -4) & (all_tgts < 4)  # Filter out extreme values

    all_gain = all_gain[mask]
    all_loss = all_loss[mask]
    all_tgts = all_tgts[mask]
    all_outs = all_outs[mask]
    all_tgts = np.abs(all_tgts)
    all_loss = np.abs(all_loss)

    mask4 = (all_tgts >= 0) & (all_tgts <= 0.3)
    mask5 = (all_tgts > 0.3) & (all_tgts < 0.6)
    mask6 = (all_tgts >= 0.6) & (all_tgts < 1.2)
    mask7 = (all_tgts >= 1.2)
    maskx = (all_tgts < 1.2) & (all_tgts >= 0)
    bin_masks = [mask4, mask5, mask6] # mask7

    bin_loss = []
    bin_outs = []
    bin_means = []
    bin_stds = []
    for idx, mask in enumerate(bin_masks):
        loss_subset = all_loss[mask]
        loss_subset = np.sqrt(loss_subset)
        # tgts_subset = all_tgts[mask]

        # loss_subset = np.sqrt(loss_subset) / tgts_subset
        bin_loss.append(np.array(loss_subset).reshape(-1))
        bin_means.append(np.mean(loss_subset) if len(loss_subset) > 0 else np.nan)
        bin_stds.append(np.std(loss_subset) if len(loss_subset) > 0 else 0)
        bin_outs.append(all_outs[mask])

    bin_masks_x = [m[maskx] for m in bin_masks]

    return bin_means, bin_stds, bin_loss, all_corr, all_loss[maskx], bin_outs, all_tgts, bin_masks_x


import numpy as np

def collect_bin_means_conditioned_on_best_hit(
    result,
    best_result,
    hit_tol=0.1,   # meters, define "Best hit" as |Best_pred - GT| <= hit_tol
    use_loss_if_available=True
):
    """
    Same as collect_bin_means(), but only keeps samples where BEST is a "hit".
    "Hit" is defined per-sample from Best's error.

    Returns:
        bin_means, bin_stds, bin_loss, all_corr, all_loss_maskx, bin_outs,
        all_tgts_abs, bin_masks_x
    """
    all_gain = []
    all_outs = []
    all_tgts = []
    all_loss = []
    all_corr = []

    # iterate stations
    for k in result.keys():
        # require both to exist + be valid
        if (k not in best_result) or (not result[k].get("corr")) or (not best_result[k].get("corr")):
            continue

        tgts = np.asarray(result[k]["tgts"][0])
        outs = np.asarray(result[k]["outs"][0])

        # --- get Best error mask (per-sample) ---
        best_tgts = np.asarray(best_result[k]["tgts"][0])
        best_outs = np.asarray(best_result[k]["outs"][0])

        # sanity: must align
        if len(tgts) != len(best_tgts) or len(outs) != len(best_outs):
            # if this happens, your pipeline isn't aligned; skip station to avoid mis-indexing
            continue

        if use_loss_if_available and ("loss" in best_result[k]) and (best_result[k]["loss"] is not None):
            # loss is usually squared error; you later do sqrt(loss) to get RMSE
            best_loss = np.asarray(best_result[k]["loss"][0]).reshape(-1)
            best_abs_err = np.sqrt(np.abs(best_loss))
        else:
            best_abs_err = np.abs(best_outs - best_tgts)

        hit_mask = best_abs_err <= hit_tol

        if hit_mask.sum() == 0:
            continue

        # --- apply the same mask to THIS method ---
        tgts = tgts[hit_mask]
        outs = outs[hit_mask]

        loss = np.asarray(result[k]["loss"][0]).reshape(-1)
        if len(loss) != len(best_abs_err):
            # if loss isn't aligned, fallback to squared error
            loss = (outs - tgts) ** 2
        else:
            loss = loss[hit_mask]

        gain = np.asarray(result[k]["gain"][0]).reshape(-1)
        if len(gain) == len(best_abs_err):
            gain = gain[hit_mask]
        else:
            gain = np.zeros_like(tgts)

        all_corr.append(result[k]["corr"][0])
        all_gain.extend(gain.tolist())
        all_tgts.extend(tgts.tolist())
        all_outs.extend(outs.tolist())
        all_loss.extend(loss.tolist())

    # to numpy
    all_gain = np.asarray(all_gain)
    all_outs = np.asarray(all_outs)
    all_tgts = np.asarray(all_tgts)
    all_loss = np.asarray(all_loss)
    all_corr = np.asarray(all_corr)

    # your original extreme filter (keep if you still want it)
    mask = (all_tgts > -4) & (all_tgts < 4)
    all_gain = all_gain[mask]
    all_loss = all_loss[mask]
    all_tgts = all_tgts[mask]
    all_outs = all_outs[mask]

    # binning uses abs targets
    all_tgts_abs = np.abs(all_tgts)
    all_loss_abs = np.abs(all_loss)

    mask4 = (all_tgts_abs >= 0) & (all_tgts_abs <= 0.3)
    mask5 = (all_tgts_abs > 0.3) & (all_tgts_abs < 0.6)
    mask6 = (all_tgts_abs >= 0.6) & (all_tgts_abs < 1.2)
    maskx = (all_tgts_abs < 1.2) & (all_tgts_abs >= 0)

    bin_masks = [mask4, mask5, mask6]

    bin_loss = []
    bin_outs = []
    bin_means = []
    bin_stds = []

    for mask_b in bin_masks:
        loss_subset = all_loss_abs[mask_b]
        rmse_vals = np.sqrt(loss_subset)  # per-sample abs error in meters
        bin_loss.append(rmse_vals.reshape(-1))
        bin_means.append(np.mean(rmse_vals) if len(rmse_vals) > 0 else np.nan)
        bin_stds.append(np.std(rmse_vals) if len(rmse_vals) > 0 else 0)
        bin_outs.append(all_outs[mask_b])

    # masks restricted to maskx region (as in your original code)
    bin_masks_x = [m[maskx] for m in bin_masks]

    return bin_means, bin_stds, bin_loss, all_corr, all_loss_abs[maskx], bin_outs, all_tgts_abs, bin_masks_x


def ttest_collect_bin_means(
    all_outs,
    method_names,
    ref_method=None,
    correction="holm",   # "holm", "bonferroni", "fdr_bh", ...
    alpha=0.05,
    nan_policy="omit",
    verbose=True,
):
    """
    Paired t-tests per bin comparing each method vs a reference method, with
    multiple-comparisons correction over ALL (method x bin) tests.

    Args:
        all_outs: list of methods, each is a list of bins, each bin is a list/array of scores
                 shape conceptually: [M][B][N]
        method_names: list of method names (len M)
        ref_method: reference method name (default: last method in method_names)
        correction: multiple-testing correction method for statsmodels.multipletests
        alpha: significance level
        nan_policy: passed to scipy.stats.ttest_rel ("omit" recommended)
        verbose: print a readable report

    Returns:
        results: list of dict rows with bin, method, t, p_raw, p_adj, reject
    """
    if len(all_outs) != len(method_names):
        raise ValueError("len(all_outs) must match len(method_names).")

    M = len(all_outs)
    B = len(all_outs[0])
    if any(len(m) != B for m in all_outs):
        raise ValueError("Each method in all_outs must have the same number of bins.")

    # pick reference method
    if ref_method is None:
        ref_idx = M - 1
    else:
        if ref_method not in method_names:
            raise ValueError(f"ref_method '{ref_method}' not found in method_names.")
        ref_idx = method_names.index(ref_method)

    ref_name = method_names[ref_idx]

    tests = []  # (bin, method_name, t, p_raw)
    for b in range(B):
        ref = np.asarray(all_outs[ref_idx][b], dtype=float)

        for m in range(M):
            name = method_names[m]
            if m == ref_idx:
                continue  # don't compare reference to itself

            x = np.asarray(all_outs[m][b], dtype=float)

            if ref.shape[0] != x.shape[0]:
                continue
                raise ValueError(
                    f"Paired test requires equal lengths. Bin {b}, '{name}' has {x.shape[0]} "
                    f"but reference '{ref_name}' has {ref.shape[0]}."
                )

            t_stat, p_raw = ttest_rel(ref, x, nan_policy=nan_policy)
            tests.append((b, name, float(t_stat), float(p_raw)))

    # multiple-comparisons correction over all tests (method x bin)
    pvals = np.array([t[3] for t in tests], dtype=float)
    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method=correction)

    results = []
    for (b, name, t_stat, p_raw), rj, padj in zip(tests, reject, p_adj):
        results.append(
            {
                "bin": b,
                "method": name,
                "ref_method": ref_name,
                "t": t_stat,
                "p_raw": p_raw,
                "p_adj": float(padj),
                "reject": bool(rj),
            }
        )

    if verbose:
        print(f"Reference: {ref_name}")
        print(f"Multiple-comparisons correction: {correction} (alpha={alpha})")
        print(f"Total tests corrected: {len(results)}\n")
        for row in results:
            sig = "SIGNIFICANT" if row["reject"] else "n.s."
            print(
                f"bin-{row['bin']} | {row['method']} vs {row['ref_method']} "
                f"| t={row['t']:.3f} | p_raw={row['p_raw']:.3e} | p_adj={row['p_adj']:.3e} | {sig}"
            )

    return results


def compare_methods_by_bin(results_dict, save_dir='./plots'):
    os.makedirs(save_dir, exist_ok=True)
    bin_labels = ["[0, 0.3]", "(0.3, 0.6)", "[0.6, 1.2)"]  # , ">= 1.2"

    method_names = []
    all_bin_means = []
    all_bin_stds = []
    all_corrs = []
    all_loss = []
    all_outs = []

    for method, result in results_dict.items():
        bin_means, bin_stds, bin_loss, corr, loss, outs, all_gain, bin_masks = collect_bin_means(result)
        all_bin_means.append(bin_means)
        all_bin_stds.append(bin_stds)
        all_corrs.append(corr)
        all_loss.append(loss)
        all_outs.append(outs)
        method_names.append(method)

    ttest_collect_bin_means(all_outs, method_names, ref_method="HIGNN")

    all_bin_means = np.array(all_bin_means)  # (n_methods, n_bins)
    all_bin_stds = np.array(all_bin_stds)    # (n_methods, n_bins)

    # After all_bin_means and all_bin_stds are ready
    x = np.arange(len(method_names))
    
    for b in range(len(bin_labels)):
        plt.figure(figsize=(14, 6))
        means = all_bin_means[:, b]

        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.bar(
            x, 
            means, 
            #yerr=stds, 
            capsize=8, 
            tick_label=method_names, 
            alpha=0.85, 
            color=plt.cm.tab10.colors[:len(method_names)]
        )
        plt.ylabel("RMSE", fontsize=25)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/mean_error_bin_{b+1}_{bin_labels[b].replace(' ', '').replace('[','').replace(']','').replace(',','_').replace('<','lt').replace('>=','gte').replace('-','m')}.png")
        plt.close()

    # Print or log correlation for each method
    for method, corr, loss, bin_mean in zip(method_names, all_corrs, all_loss, all_bin_means):
        corr = np.array(corr)
        mean_corr = np.nanmean(corr)
        std_corr = np.nanstd(corr)
        mean_loss = np.nanmean(np.sqrt(loss))
        std_loss = np.nanstd(np.sqrt(loss))
        print(f"{method}: Mean RMSE = {bin_mean}")
        print(f"{method}: Overall correlation = {mean_corr:.4f} - {std_corr:.4f}")
        print(f"{method}: Overall RMSE = {mean_loss:.4f} - {std_loss:.4f}")


def compare_methods_by_bin_conditioned_on_best_hit(
    results_dict,
    best_name="Best",
    hit_tol=0.1,
    save_dir="./plots_best_hit",
):
    os.makedirs(save_dir, exist_ok=True)
    bin_labels = ["[0, 0.3]", "(0.3, 0.6)", "[0.6, 1.2)"]

    assert best_name in results_dict, f"'{best_name}' not found in results_dict keys."

    best_result = results_dict[best_name]

    method_names = []
    all_bin_means = []
    all_bin_stds = []
    all_corrs = []
    all_loss = []
    all_outs = []

    # Keep methods in the same order as dict
    for method, result in results_dict.items():
        bin_means, bin_stds, bin_loss, corr, loss, outs, all_tgts, bin_masks = \
            collect_bin_means_conditioned_on_best_hit(
                result,
                best_result,
                hit_tol=hit_tol,
                use_loss_if_available=True
            )

        all_bin_means.append(bin_means)
        all_bin_stds.append(bin_stds)
        all_corrs.append(corr)
        all_loss.append(loss)
        all_outs.append(outs)
        method_names.append(method)

    # paired t-test using the SAME conditioning
    ttest_collect_bin_means(all_outs, method_names, ref_method="HIGNN")

    all_bin_means = np.asarray(all_bin_means)
    all_bin_stds = np.asarray(all_bin_stds)

    x = np.arange(len(method_names))

    for b in range(len(bin_labels)):
        plt.figure(figsize=(14, 6))
        means = all_bin_means[:, b]

        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.bar(
            x,
            means,
            capsize=8,
            tick_label=method_names,
            alpha=0.85,
            color=plt.cm.tab10.colors[:len(method_names)]
        )
        plt.ylabel("RMSE (conditioned on Best-hit)", fontsize=25)
        plt.title(f"RMSE in bin {bin_labels[b]} | Best-hit tol = {hit_tol} m", fontsize=20)
        plt.tight_layout()

        fname = bin_labels[b].replace(' ', '').replace('[','').replace(']','').replace(',','_').replace('<','lt').replace('>=','gte').replace('-','m')
        plt.savefig(f"{save_dir}/mean_error_best_hit_bin_{b+1}_{fname}.png")
        plt.close()

    # Log overall stats (also conditioned)
    for method, corr, loss, bin_mean in zip(method_names, all_corrs, all_loss, all_bin_means):
        corr = np.asarray(corr)
        mean_corr = np.nanmean(corr) if len(corr) else np.nan
        std_corr  = np.nanstd(corr)  if len(corr) else np.nan

        rmse_vals = np.sqrt(np.asarray(loss)) if len(loss) else np.array([])
        mean_rmse = np.nanmean(rmse_vals) if len(rmse_vals) else np.nan
        std_rmse  = np.nanstd(rmse_vals)  if len(rmse_vals) else np.nan

        print(f"{method}: Mean RMSE per bin = {bin_mean}")
        print(f"{method}: Overall correlation = {mean_corr:.4f} ± {std_corr:.4f}")
        print(f"{method}: Overall RMSE (Best-hit conditioned) = {mean_rmse:.4f} ± {std_rmse:.4f}")


def compare_methods_by_bin_new(results_dict, save_dir='./plots', baseline_method=None, my_method=None):
    os.makedirs(save_dir, exist_ok=True)
    bin_labels = ["[0, 0.3]", "(0.3, 0.6)", "[0.6, 1.2)"]

    method_names, all_bin_means, all_bin_stds, all_corrs, all_loss, all_outs = [], [], [], [], [], []
    first_bin_masks = None  # assume binning based on GT; should be same across methods

    all_bin_loss = []
    for method, result in results_dict.items():
        bin_means, bin_stds, bin_loss, corr, loss, outs, all_gain, bin_masks = collect_bin_means(result)
        if first_bin_masks is None:
            first_bin_masks = bin_masks
        all_bin_means.append(bin_means)
        all_bin_stds.append(bin_stds)
        all_corrs.append(corr)
        all_loss.append(loss)
        all_bin_loss.append(bin_loss)
        all_outs.append(outs)
        method_names.append(method)

    # Optional stats test you already have
    ttest_collect_bin_means(all_outs, method_names)

    all_bin_means = np.array(all_bin_means)  # (n_methods, n_bins)
    all_bin_stds  = np.array(all_bin_stds)   # (n_methods, n_bins)

    # ---------- Gain/Loss vs baseline ----------
    if baseline_method is None:
        baseline_method = method_names[0]
    if baseline_method not in method_names:
        raise ValueError(f"baseline_method='{baseline_method}' not in {method_names}")

    # Bin counts for weighting
    baseline_idx = method_names.index(baseline_method)
    baseline_outs = all_bin_loss[baseline_idx]  # (n_bins,)

    mymethod_idx = method_names.index(my_method)
    mymethod_outs = all_bin_loss[mymethod_idx]  # (n_bins,)

    # Per-method, per-bin gain (positive = better than baseline)
    gains = [
        np.nanmean(baseline_outs[b] - mymethod_outs[b]) for b in range(len(bin_labels))
    ]
    # ---------- Optional: plot gains for a chosen method ----------
    xx = np.arange(len(bin_labels))
    plt.figure(figsize=(8.5, 6))
    plt.bar(xx, gains, tick_label=bin_labels, alpha=0.9, color=plt.cm.tab10.colors[:len(bin_labels)])
    plt.axhline(0, linewidth=1)

    plt.ylabel(f"Gain (meter)", fontsize=19)
    plt.xticks(fontsize=19, rotation=15)
    plt.yticks(fontsize=19)

    plt.tight_layout()
    out_png = os.path.join(save_dir, f"gains_{my_method}_vs_{baseline_method}.png")
    plt.savefig(out_png)
    plt.close()


def compare_methods_by_bin_best_ref(
    results_dict,
    best_key="Best",
    best_rmse_threshold=0.2,
    save_dir="./plots_best_ref",
    min_samples_per_bin=1,
):
    os.makedirs(save_dir, exist_ok=True)
    bin_labels = ["[0, 0.3]", "(0.3, 0.6)", "[0.6, 1.2)"]  # , ">= 1.2"

    # --- collect Best arrays ---
    (best_bin_means, best_bin_stds, best_bin_loss,
     best_corr, best_loss_mse, best_outs, best_tgts, best_bin_masks) = collect_bin_means(results_dict[best_key])

    best_loss_mse = np.asarray(best_loss_mse)
    best_rmse = np.sqrt(best_loss_mse)   # IMPORTANT: loss_mse -> rmse

    good_mask = np.isfinite(best_rmse) & (best_rmse <= best_rmse_threshold)
    print(f"[Best-ref filter] Using {good_mask.sum()}/{len(best_rmse)} samples where '{best_key}' RMSE <= {best_rmse_threshold}")

    # sanity check: this MUST hold
    print("Sanity: max Best RMSE among kept =", np.nanmax(best_rmse[good_mask]))

    method_names = []
    all_bin_means = []
    all_bin_stds = []
    all_corrs = []
    all_loss_rmse = []
    all_outs = []

    for method, result in results_dict.items():
        (bin_means, bin_stds, bin_loss,
         corr, loss_mse, outs, tgts, bin_masks) = collect_bin_means(result)

        loss_mse = np.asarray(loss_mse)
        rmse = np.sqrt(loss_mse)

        # ensure alignment with Best (same length after collect_bin_means)
        if len(rmse) != len(best_rmse):
            raise ValueError(
                f"Length mismatch after collect_bin_means: method '{method}' has {len(rmse)} "
                f"but Best has {len(best_rmse)}. This means your sample concatenation order differs."
            )

        # --- recompute per-bin means/stds on FILTERED samples only ---
        bin_means_f, bin_stds_f = [], []
        for b in range(len(bin_labels)):
            m = bin_masks[b] & good_mask & np.isfinite(rmse)
            if m.sum() < min_samples_per_bin:
                bin_means_f.append(np.nan)
                bin_stds_f.append(np.nan)
            else:
                bin_means_f.append(float(np.mean(rmse[m])))
                bin_stds_f.append(float(np.std(rmse[m])))

        method_names.append(method)
        all_bin_means.append(bin_means_f)
        all_bin_stds.append(bin_stds_f)
        all_corrs.append(corr)
        all_loss_rmse.append(rmse[good_mask])   # store filtered rmse for overall stats
        all_outs.append(outs)

    # --- plot bars ---
    all_bin_means = np.array(all_bin_means)
    x = np.arange(len(method_names))

    for b in range(len(bin_labels)):
        plt.figure(figsize=(14, 6))
        means = all_bin_means[:, b]
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.bar(
            x,
            means,
            capsize=8,
            tick_label=method_names,
            alpha=0.85,
            color=plt.cm.tab10.colors[:len(method_names)]
        )
        plt.ylabel(f"RMSE (Best RMSE <= {best_rmse_threshold})", fontsize=25)
        plt.tight_layout()
        plt.savefig(
            f"{save_dir}/mean_error_bin_{b+1}_{bin_labels[b].replace(' ', '').replace('[','').replace(']','').replace(',','_').replace('<','lt').replace('>=','gte').replace('-','m')}_bestlte{best_rmse_threshold}.png"
        )
        plt.close()

    # --- overall stats on filtered samples ---
    for method, corr, rmse_f in zip(method_names, all_corrs, all_loss_rmse):
        # corr in your code is per-k (length ~ number of stations), not per-sample -> keep as you already do
        corr = np.array(corr)
        mean_corr = np.nanmean(corr)
        std_corr = np.nanstd(corr)
        mean_rmse = np.nanmean(rmse_f)
        std_rmse = np.nanstd(rmse_f)
        print(f"{method}: Overall correlation = {mean_corr:.4f} - {std_corr:.4f}")
        print(f"{method}: Overall RMSE (filtered by Best) = {mean_rmse:.4f} - {std_rmse:.4f}")


def drawing(results_dict, chunk_len=168):
    os.makedirs('./plot-results', exist_ok=True)
    
    fm = list(results_dict.keys())[0]
    for k in results_dict[fm].keys():
        all_outs = []
        method_names = ['GT']
        if results_dict[fm][k]['corr']:
            all_outs.append(results_dict[fm][k]['tgts'][0])
            for method, result in results_dict.items():
                all_outs.append(result[k]['outs'][0])
                method_names.append(method)

            # Convert all_outs to numpy arrays if needed
            all_outs = [out.cpu().numpy() if hasattr(out, "cpu") else out for out in all_outs]
            
            # Get the length of the time series
            n = len(all_outs[0])
            n_chunks = (n + chunk_len - 1) // chunk_len  # ceil division

            for i in range(n_chunks):
                start = i * chunk_len
                end = min((i + 1) * chunk_len, n)
                time_steps = range(start, end)

                # if np.max(all_outs[0][start:end]) <= 0.6:
                #     continue

                plt.figure(figsize=(12, 6))
                for outs, name in zip(all_outs, method_names):
                    plt.plot(time_steps, outs[start:end], label=name, marker='o')
                plt.title(f'Comparison of Methods for {k} (Chunk {i+1})')
                plt.xlabel('Time Step')
                plt.ylabel('Output Value')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                
                # Save each chunk separately
                output_path = f'./plot-results/{k}_chunk{i+1}.png'
                plt.savefig(output_path)
                plt.close()


def collect_station_rmse_in_bracket(
    results,
    low=0.6,
    high=1.2,
    exclude_stations=None
):
    """
    Returns: dict {station_id: rmse}
    """
    station_rmse = {}

    for k in results.keys():
        if exclude_stations and k in exclude_stations:
            continue
        if not results[k]['corr']:
            continue

        tgts = np.array(results[k]['tgts'][0])
        outs = np.array(results[k]['outs'][0])

        mask = (np.abs(tgts) >= low) & (np.abs(tgts) < high)
        if mask.sum() == 0:
            continue

        err = outs[mask] - tgts[mask]
        rmse = np.sqrt(np.mean(err ** 2))
        station_rmse[k] = rmse

    return station_rmse


def collect_station_relative_rmse_per_sample(
    results,
    low=0.6,
    high=1.2,
    exclude_stations=None,
    eps=1e-6
):
    station_rel_rmse = {}

    for k in results.keys():
        if exclude_stations and k in exclude_stations:
            continue
        if not results[k]['corr']:
            continue

        tgts = np.array(results[k]['tgts'][0])
        outs = np.array(results[k]['outs'][0])

        mask = (np.abs(tgts) >= low) & (np.abs(tgts) < high)
        if mask.sum() == 0:
            continue

        rel_err = (outs[mask] - tgts[mask]) / (np.abs(tgts[mask]) + eps)
        station_rel_rmse[k] = np.sqrt(np.mean(rel_err ** 2))

    return station_rel_rmse


def reletive_percentage_under_thresholds(values_dict, thresholds):
    vals = np.array(list(values_dict.values()))
    return [100.0 * np.mean(vals <= t) for t in thresholds]


def percentage_under_thresholds(station_rmse, thresholds):
    rmses = np.array(list(station_rmse.values()))
    percentages = []

    for t in thresholds:
        percentages.append(100.0 * np.mean(rmses <= t))

    return np.array(percentages)


def plot_relative_rmse_percentage(
    results_dict,
    thresholds=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    low=0.6,
    high=1.2,
    abs_err_thresh=1.5,
    reference_method="HIGNN",
    save_path="./plots/relative_rmse_percentage_0.6_1.2.png"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 🔹 detect outliers ONCE
    ref_results = results_dict[reference_method]
    outliers = find_outlier_stations_by_absolute_error(
        ref_results, low, high, abs_err_thresh
    )

    print(f"Excluded {len(outliers)} outlier stations (|err| > {abs_err_thresh} m)")

    plt.figure(figsize=(12, 6))

    for method, results in results_dict.items():
        station_rel_rmse = collect_station_relative_rmse_per_sample(
            results,
            low,
            high,
            exclude_stations=outliers
        )

        if len(station_rel_rmse) == 0:
            continue

        percentages = percentage_under_thresholds(
            station_rel_rmse, thresholds
        )

        plt.plot(
            thresholds,
            percentages,
            marker='o',
            linewidth=2,
            label=method
        )

    plt.xlabel("Relative RMSE threshold (fraction of GT)", fontsize=18)
    plt.ylabel("Percentage of stations (%)", fontsize=18)
    plt.xticks(thresholds, fontsize=14)
    plt.yticks(np.arange(0, 101, 10), fontsize=14)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_station_rmse_percentages(
    results_dict,
    thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
    low=0.6,
    high=1.2,
    abs_err_thresh=1.5,
    reference_method="HIGNN",
    save_path="./plots/station_rmse_percentage_0.6_1.2.png"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 🔹 detect outliers ONCE
    ref_results = results_dict[reference_method]
    outliers = find_outlier_stations_absolute_error(
        ref_results,
        low=low,
        high=high,
        abs_err_thresh=abs_err_thresh
    )

    print(
        f"[INFO] Excluded {len(outliers)} outlier stations "
        f"(median |err| > {abs_err_thresh} m)"
    )

    plt.figure(figsize=(12, 6))

    for method, results in results_dict.items():
        station_rmse = collect_station_rmse_in_bracket(
            results,
            low=low,
            high=high,
            exclude_stations=outliers
        )

        if len(station_rmse) == 0:
            continue

        percentages = percentage_under_thresholds(
            station_rmse,
            thresholds
        )

        plt.plot(
            thresholds,
            percentages,
            marker='o',
            linewidth=2,
            label=method
        )

    plt.xlabel("RMSE threshold (m)", fontsize=18)
    plt.ylabel("Percentage of stations (%)", fontsize=18)
    plt.title("Stations with RMSE ≤ threshold (|GT| ∈ [0.6, 1.2))", fontsize=18)

    plt.xticks(thresholds, fontsize=14)
    plt.yticks(np.arange(0, 101, 10), fontsize=14)
    plt.ylim(0, 100)

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def find_outlier_stations_by_absolute_error(
    results,
    low=0.6,
    high=1.2,
    abs_err_thresh=1.5
):
    """
    Returns:
        set of station_ids to exclude
    """
    outliers = set()

    for k in results.keys():
        if not results[k]['corr']:
            continue

        tgts = np.array(results[k]['tgts'][0])
        outs = np.array(results[k]['outs'][0])

        mask = (np.abs(tgts) >= low) & (np.abs(tgts) < high)
        if mask.sum() == 0:
            continue

        abs_err = np.abs(outs[mask] - tgts[mask])
        if np.median(abs_err) > abs_err_thresh:
            outliers.add(k)

    return outliers


def find_outlier_stations_absolute_error(
    results,
    low=0.6,
    high=1.2,
    abs_err_thresh=1.5
):
    """
    Returns:
        set of station_ids to exclude
    """
    outliers = set()

    for k in results.keys():
        if not results[k]['corr']:
            continue

        tgts = np.array(results[k]['tgts'][0])
        outs = np.array(results[k]['outs'][0])

        mask = (np.abs(tgts) >= low) & (np.abs(tgts) < high)
        if mask.sum() == 0:
            continue

        abs_err = np.abs(outs[mask] - tgts[mask])

        # robust criterion: consistently large error
        if np.median(abs_err) > abs_err_thresh:
            outliers.add(k)

    return outliers


def extract_abnormal_peaks(
    y,
    event_thresh=0.6,
    min_prominence=0.1,
    min_distance=6
):
    """
    Extract indices of abnormal GT peaks.

    Parameters
    ----------
    y : 1D array
        GT time series
    event_thresh : float
        Absolute threshold for abnormal event
    min_prominence : float
        Suppresses noisy spikes
    min_distance : int
        Minimum distance between peaks (timesteps)

    Returns
    -------
    peaks_idx : np.ndarray
        Indices of abnormal peaks
    """
    peaks, props = find_peaks(
        y,
        height=event_thresh,
        prominence=min_prominence,
        distance=min_distance
    )

    return peaks


def match_predicted_peak(
    tgts,
    outs,
    gt_idx,
    time_tol=6,
    peak_tol=0.2
):
    """
    Check whether prediction matches a GT peak.

    Returns:
        hit (bool), signed_error (float or None)
    """
    start = max(0, gt_idx - time_tol)
    end   = min(len(outs), gt_idx + time_tol + 1)

    local_pred = outs[start:end]
    pr_peak = np.max(local_pred)
    pr_idx = start + np.argmax(local_pred)

    gt_peak = tgts[gt_idx]

    signed_err = gt_peak - pr_peak
    hit = pr_peak >= (signed_err - peak_tol)

    return hit, signed_err, pr_idx - gt_idx


def collect_peak_metrics(
    results,
    event_thresh=0.6,
    peak_tol=0.2,
    time_tol=6,
    min_prominence=0.1,
    min_distance=6
):
    peak_errors = []
    event_hits = []
    false_positives = []

    for k in results.keys():
        if not results[k]['corr']:
            continue

        tgts = np.asarray(results[k]['tgts'][0])
        outs = np.asarray(results[k]['outs'][0])

        # ---------- Extract GT abnormal peaks ----------
        gt_peaks = extract_abnormal_peaks(
            tgts,
            event_thresh=event_thresh,
            min_prominence=min_prominence,
            min_distance=min_distance
        )

        # ---------- No GT abnormal event ----------
        if len(gt_peaks) == 0:
            # false positive if prediction shows abnormal peak anywhere
            pr_peaks = extract_abnormal_peaks(
                outs,
                event_thresh=event_thresh,
                min_prominence=min_prominence,
                min_distance=min_distance
            )
            false_positives.append(len(pr_peaks) > 0)
            continue

        # ---------- Match each GT peak ----------
        for gt_idx in gt_peaks:
            hit, signed_err, _ = match_predicted_peak(
                tgts,
                outs,
                gt_idx,
                time_tol=time_tol,
                peak_tol=peak_tol
            )

            event_hits.append(hit)
            if signed_err is not None:
                peak_errors.append(signed_err)

    return {
        "peak_errors": np.array(peak_errors),
        "event_recall": np.mean(event_hits) if event_hits else np.nan,
        "false_positive_rate": np.mean(false_positives) if false_positives else np.nan
    }


def plot_peak_reconstruction(
    results_dict,
    event_thresh=0.6,
    save_dir="./plots_peak_eval"
):
    os.makedirs(save_dir, exist_ok=True)

    peak_tol = 0.1  # tolerance for counting a hit (predicted peak close enough to GT peak)

    method_names = []
    peak_errs = []
    false_positive_rates = []
    recalls = []

    for method, results in results_dict.items():
        metrics = collect_peak_metrics(results, event_thresh, peak_tol=peak_tol)

        if len(metrics["peak_errors"]) == 0:
            continue

        method_names.append(method)
        peak_errs.append(metrics["peak_errors"])
        false_positive_rates.append(metrics["false_positive_rate"])
        recalls.append(metrics["event_recall"])

    # -------------------------
    # 1️⃣ Signed peak error (boxplot)
    # -------------------------
    plt.figure(figsize=(12, 6))
    plt.boxplot(
        peak_errs,
        tick_labels=method_names,
        showfliers=False
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Signed peak error (Pred − GT) [m]", fontsize=18)
    plt.xticks(rotation=20, fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("Peak amplitude error (signed)", fontsize=18)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/peak_error_signed.png")
    plt.close()

    # -------------------------
    # 2️⃣ False positive peak detection (bar plot)
    # -------------------------
    plt.figure(figsize=(10, 6))
    plt.bar(
        method_names,
        false_positive_rates,
        color=plt.cm.tab10.colors[:len(method_names)]
    )
    plt.ylim(0, 1)
    plt.ylabel("False positive rate", fontsize=18)
    plt.xticks(rotation=20, fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(
        f"False positive peak detection (Pred ≥ {event_thresh} m, GT < {event_thresh} m)",
        fontsize=18
    )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/false_positive_rate.png")
    plt.close()

    # -------------------------
    # 3️⃣ Event recall (bar plot)
    # -------------------------
    plt.figure(figsize=(10, 6))
    plt.bar(
        method_names,
        recalls,
        color=plt.cm.tab10.colors[:len(method_names)]
    )
    plt.ylim(0, 1)
    plt.ylabel("Event recall", fontsize=18)
    plt.xticks(rotation=20, fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(
        f"Event recall (GT ≥ {event_thresh} m, tol = {peak_tol} m)",
        fontsize=18
    )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/event_recall.png")
    plt.close()


def load_and_log():
    results_dict = {
        # "IDW": 'model.distance.InverseDistance-results.pkl',
        # "OK": 'model.kriging.OrdinaryKrigingInterpolation-results.pkl',
        # "MLP": 'model.mlp.MLPW-results.pkl',
        # "NNGP": 'model.nngp.SpatioTemporalNNGP.pkl-results.pkl',
        # "KED": 'model.nngp.FAST_GSTOOLS_KED.pkl-results.pkl',
        # "CoKriging": 'model.nngp.CoKriging.pkl-results.pkl',
        # "20%": 'g-model.gnn.GATWithEdgeAttrRain-results-all-t0.2.pkl',
        # "40%": 'g-model.gnn.GATWithEdgeAttrRain-results-all-0.4.pkl',
        # "60%": 'g-model.gnn.GATWithEdgeAttrRain-results-all-t0.6.pkl',
        # "80%": 'g-model.gnn.GATWithEdgeAttrRain-results-all-t0.8.pkl',
        # "100%": 'g-model.gnn.GATWithEdgeAttrRain-results-all-1.0.pkl',
        "gnn-0.2": 'g-model.gnn.GATWithEdgeAttrRain-results-all-gru-0.2.pkl',
        # "gnn-0.4": 'g-model.gnn.GATWithEdgeAttrRain-results-all-gru-0.4.pkl',
        # "gnn-0.6": 'g-model.gnn.GATWithEdgeAttrRain-results-all-gru-0.6.pkl',
        # "gnn-0.8": "g-model.gnn.GATWithEdgeAttrRain-results-all-gru-0.8.pkl",
        # "gnn-1.0": "g-model.gnn.GATWithEdgeAttrRain-results-all-gru-1.0.pkl",
        "mlp-0.2": "g-model.gnn.GATWithEdgeAttrRain-results-all-mlp-0.2.pkl",
        "mlp-0.4": "g-model.gnn.GATWithEdgeAttrRain-results-all-mlp-0.4.pkl",
        "mlp-0.8": "g-model.gnn.GATWithEdgeAttrRain-results-all-mlp-0.8.pkl",
        "mlp-1.0": "g-model.gnn.GATWithEdgeAttrRain-results-all-mlp-1.0.pkl",
        # "HIGNNx": 'g-model.gnn.GATWithEdgeAttrRain-results-all-d1.0.pkl',
        "HIGNN": 'g-model.gnn.GATWithEdgeAttrRain-results-all-mlp-0.6.pkl',
        # "HIGNN": 'g-model.gnn.GATWithEdgeAttrRain-results-all-mlp.pkl',
        # "HIGNN-idw": 'g-model.gnn.GATWithEdgeAttrRain-results-all-idw.pkl',
        # "HIGNN-only": 'g-model.gnn.GATWithEdgeAttrRain-results-all-only.pkl',
        "Best": 'model.best.possible.pkl-results.pkl',
    }
    for k in list(results_dict.keys()):
        with open(results_dict[k], 'rb') as f:
            results_dict[k] = pickle.load(f)

    compare_methods_by_bin(results_dict)
    # compare_methods_by_bin_best_ref(results_dict, best_rmse_threshold=0.1)
    # compare_methods_by_bin_new(results_dict, baseline_method="MLP", my_method="HIGNN")
    # compare_methods_by_bin_new(results_dict, baseline_method="KED", my_method="HIGNN")
    # compare_methods_by_bin_new(results_dict, baseline_method="OK", my_method="HIGNN")
    # compare_methods_by_bin_new(results_dict, baseline_method="IDW", my_method="HIGNN")
    # compare_methods_by_bin_new(results_dict, baseline_method="NNGP", my_method="HIGNN")
    # drawing(results_dict)

    # plot_station_rmse_percentages(
    #     results_dict,
    #     thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    # )
    # plot_relative_rmse_percentage(
    #     results_dict,
    #     thresholds=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    # )

    plot_peak_reconstruction(
        results_dict,
        event_thresh=0.6
    )

    compare_methods_by_bin_conditioned_on_best_hit(
        results_dict,
        best_name="Best",
        hit_tol=0.1,
        save_dir="./plots_best_hit"
    )

    # make_station_error_map(
    #     results_dict,
    #     method_name="HIGNN",
    #     lo=0.6, hi=1.2,
    #     out_html="map_rmse_bin_0p6_1p2.html",
    # )
    # make_station_error_map(
    #     results_dict,
    #     method_name="HIGNN",
    #     lo=0.0, hi=4,
    #     out_html="map_rmse_bin_all.html",
    # )


folder = './'

load_and_log()
