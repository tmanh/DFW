from functools import reduce
import os
import logging
import pickle
import numpy as np

from scipy.stats import ttest_rel
import matplotlib.pyplot as plt


if os.path.exists('log-test-all.txt'):
    os.remove('log-test-all.txt')
logging.basicConfig(filename="log-test-all.txt", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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
    maskx = (all_tgts < 1.2)
    bin_masks = [mask4, mask5, mask6, mask7] # mask7

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

    return bin_means, bin_stds, bin_loss, all_corr, all_loss[maskx], bin_outs, all_tgts, bin_masks


def ttest_collect_bin_means(all_outs, method_names):
    for b in range(len(all_outs[0])):
        for scores, name in zip(all_outs, method_names):
            # Use ttest_rel if your results are paired (e.g., same runs or folds)
            print(len(scores[b]), len(all_outs[-1][b]))

    for b in range(len(all_outs[0])):
        for scores, name in zip(all_outs, method_names):
            # Use ttest_rel if your results are paired (e.g., same runs or folds)
            print(len(scores[b]), len(all_outs[-1][b]))
            t_stat, p_value = ttest_rel(np.array(all_outs[-1][b]), np.array(scores[b]))
            print(f"t-test bin-{b}: t = {t_stat:.3f}, p = {p_value}")
            if p_value < 0.05:
                print(f"Result: Statistically significant difference with Method {name}.\n")
            else:
                print(f"Result: No statistically significant difference with Method {name}.\n")

    # for scores in all_outs:
    #     # Use ttest_rel if your results are paired (e.g., same runs or folds)
    #     t_stat, p_value = ttest_rel(scores, all_outs[1], nan_policy='omit')
    #     print(f"t-test: t = {t_stat:.3f}, p = {p_value:.4f}")
    #     if p_value < 0.05:
    #         print(f"Result: Statistically significant difference with Method.\n")
    #     else:
    #         print(f"Result: No statistically significant difference with Method.\n")



def compare_methods_by_bin(results_dict, save_dir='./plots'):
    os.makedirs(save_dir, exist_ok=True)
    bin_labels = ["[0, 0.3]", "(0.3, 0.6)", "[0.6, 1.2)", ">= 1.2"]  # , ">= 1.2"

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

    ttest_collect_bin_means(all_outs, method_names)

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
    for method, corr, loss in zip(method_names, all_corrs, all_loss):
        corr = np.array(corr)
        mean_corr = np.nanmean(corr)
        std_corr = np.nanstd(corr)
        mean_loss = np.nanmean(loss)
        std_loss = np.nanstd(loss)
        print(f"{method}: Overall correlation = {mean_corr:.4f} - {std_corr:.4f}")
        print(f"{method}: Overall RMSE = {mean_loss:.4f} - {std_loss:.4f}")


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

                if np.max(all_outs[0][start:end]) <= 0.6:
                    continue

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
    reference_method="MLP",
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


def load_and_log():
    results_dict = {
        # "no-std": 'g-model.gnn.GATWithEdgeAttrRain-results-no-std.pkl',
        # "no-cor": 'g-model.gnn.GATWithEdgeAttrRain-results-no-cor.pkl',
        # "mse-only": 'g-model.gnn.GATWithEdgeAttrRain-results-mse-only.pkl',

        "IDW": 'model.distance.InverseDistance-results.pkl',
        "OK": 'model.kriging.OrdinaryKrigingInterpolation-results.pkl',
        "MLP": 'model.mlp.MLPW-results.pkl',
        "NNGP": 'model.nngp.SpatioTemporalNNGP.pkl-results.pkl',
        "KED": 'model.nngp.KED.pkl-results.pkl',
        "CoKriging": 'model.nngp.CoKriging.pkl-results.pkl',
        # "PGNN-idw": 'g-model.gnn.GATWithEdgeAttrRain-results-gru-idw.pkl',
        # "LOK": 'g-model.gnn.GATWithEdgeAttrRain-results-lok.pkl',
        # "PGNN": 'g-model.gnn.GATWithEdgeAttrRain-results-mlp.pkl',
        # "GNN": 'g-model.gnn.GATWithEdgeAttr-results.pkl',
        # "new": 'g-model.gnn.GATWithEdgeAttrRain-results-all.pkl',
        "HIGNN": 'g-model.gnn.GATWithEdgeAttrRain-results-gru-mlp.pkl',
        # "20%": 'g-model.gnn.GATWithEdgeAttrRain-results-all-t0.2.pkl',
        # "40%": 'g-model.gnn.GATWithEdgeAttrRain-results-all-0.4.pkl',
        # "60%": 'g-model.gnn.GATWithEdgeAttrRain-results-all-t0.6.pkl',
        # "80%": 'g-model.gnn.GATWithEdgeAttrRain-results-all-t0.8.pkl',
        # "100%": 'g-model.gnn.GATWithEdgeAttrRain-results-all-1.0.pkl',
        # "PGNN-min": 'g-model.gnn.GATWithEdgeAttrRain-results-mingru-mlp.pkl',
        # "PGNN-mlp": 'g-model.gnn.GATWithEdgeAttrRain-results-mlp-mlp.pkl',
        # "PGNN-no": "g-model.gnn.GATWithEdgeAttrRain-results-no-mlp.pkl",
        # "D": "g-model.gnn.GATWithEdgeAttrRain-results-all-d0.2.pkl",
        # "NE": "g-model.gnn.GATWithEdgeAttrRain-results-all-ne0.2.pkl",
        # "NS": "g-model.gnn.GATWithEdgeAttrRain-results-all-ns0.2.pkl",
        # "Tidal": "g-model.gnn.GATWithEdgeAttrRain-results-all-td0.2.pkl",
        # "Tidal": "g-model.gnn.GATWithEdgeAttrRain-results-all-tidal0.2.pkl",
    }
    for k in list(results_dict.keys()):
        with open(results_dict[k], 'rb') as f:
            results_dict[k] = pickle.load(f)

    compare_methods_by_bin(results_dict)
    # compare_methods_by_bin_new(results_dict, baseline_method="MLP", my_method="HIGNN")
    # compare_methods_by_bin_new(results_dict, baseline_method="KED", my_method="HIGNN")
    # compare_methods_by_bin_new(results_dict, baseline_method="OK", my_method="HIGNN")
    # compare_methods_by_bin_new(results_dict, baseline_method="IDW", my_method="HIGNN")
    # compare_methods_by_bin_new(results_dict, baseline_method="NNGP", my_method="HIGNN")
    drawing(results_dict)

    plot_station_rmse_percentages(
        results_dict,
        thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    )
    # plot_relative_rmse_percentage(
    #     results_dict,
    #     thresholds=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    # )


folder = './'

load_and_log()
