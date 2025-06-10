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

    mask4 = (all_tgts >= 0) & (all_tgts <= 0.30)
    mask5 = (all_tgts > 0.30) & (all_tgts < 0.6)
    mask6 = (all_tgts >= 0.6) & (all_tgts < 1.2)
    # mask7 = (all_gain >= 1.2)
    maskx = (all_tgts < 1.2)
    bin_masks = [mask4, mask5, mask6]

    bin_outs = []
    bin_means = []
    bin_stds = []
    for idx, mask in enumerate(bin_masks):
        loss_subset = all_loss[mask]
        loss_subset = np.sqrt(loss_subset)
        # tgts_subset = all_tgts[mask]

        # loss_subset = np.sqrt(loss_subset) / tgts_subset
        bin_means.append(np.mean(loss_subset) if len(loss_subset) > 0 else np.nan)
        bin_stds.append(np.std(loss_subset) if len(loss_subset) > 0 else 0)
        bin_outs.append(all_outs[mask])

    return bin_means, bin_stds, all_corr, all_loss[maskx], bin_outs, all_tgts, bin_masks


def ttest_collect_bin_means(all_outs, method_names):
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
    bin_labels = ["[0, 0.3]", "(0.3, 0.6)", "[0.6, 1.2)"]  # , ">= 1.2"

    method_names = []
    all_bin_means = []
    all_bin_stds = []
    all_corrs = []
    all_loss = []
    all_outs = []

    for method, result in results_dict.items():
        bin_means, bin_stds, corr, loss, outs, all_gain, bin_masks = collect_bin_means(result)
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
        plt.figure(figsize=(8.5, 6))
        means = all_bin_means[:, b]
        stds = all_bin_stds[:, b]

        # Get GT stats for this bin
        gt_mask = bin_masks[b]
        gt_vals = all_gain[gt_mask]
        gt_min = np.nanmin(gt_vals) if len(gt_vals) > 0 else float('nan')
        gt_mean = np.nanmean(gt_vals) if len(gt_vals) > 0 else float('nan')
        gt_max = np.nanmax(gt_vals) if len(gt_vals) > 0 else float('nan')
        
        print(bin_labels[b], means, stds)
        
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.bar(
            x, 
            means, 
            yerr=stds, 
            capsize=8, 
            tick_label=method_names, 
            alpha=0.85, 
            color=plt.cm.tab10.colors[:len(method_names)]
        )
        plt.ylabel("RMSE", fontsize=25)
        plt.title(f"Mean Error ± Std for Bin: {bin_labels[b]}", fontsize=25)
        plt.suptitle(
            f"Difference - Min: {gt_min:.2f} | Mean: {gt_mean:.2f} | Max: {gt_max:.2f}",
            x=0.5, y=0.97, ha='center', fontsize=25, color='black'
        )

        # plt.ylim([
        #     np.nanmin(means - stds) - 0.1 * np.nanmax(np.abs(means - stds)),
        #     np.nanmax(means + stds) + 0.1 * np.nanmax(np.abs(means + stds))
        # ])
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

def load_and_log():
    results_dict = {
        "IDW": 'backup/model.distance.InverseDistance-results.pkl',
        "OK": 'backup/model.kriging.OrdinaryKrigingInterpolation-results.pkl',
        "MLP": 'backup/model.mlp.MLPW-results.pkl',
        "POK": 'backup/model.nngp.SpatioTemporalNNGP.pkl-results.pkl',
        "PGNN": 'backup/g-model.gnn.GATWithEdgeAttrRain-results-all.pkl',
        # "no-std": 'backup/g-model.gnn.GATWithEdgeAttrRain-results-no-std.pkl',
        # "no-cor": 'backup/g-model.gnn.GATWithEdgeAttrRain-results-no-cor.pkl',
        # "mse-only": 'backup/g-model.gnn.GATWithEdgeAttrRain-results-mse-only.pkl',
    }
    for k in list(results_dict.keys()):
        with open(results_dict[k], 'rb') as f:
            results_dict[k] = pickle.load(f)

    compare_methods_by_bin(results_dict)
    drawing(results_dict)


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


folder = './'

load_and_log()
