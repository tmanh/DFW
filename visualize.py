from functools import reduce
import os
import logging
import pickle
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


if os.path.exists('log-test-all.txt'):
    os.remove('log-test-all.txt')
logging.basicConfig(filename="log-test-all.txt", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def collect_bin_means(result):
    all_mean = []
    all_loss = []
    all_corr = []
    for k in result.keys():
        if result[k]['corr']:
            all_corr.append(result[k]['corr'])
            all_mean.extend(result[k]['mean'][0].tolist())
            all_loss.extend(result[k]['loss'][0].tolist())

    all_mean = np.array(all_mean)
    all_loss = np.array(all_loss)
    all_corr = np.array(all_corr)
    corr = np.mean(all_corr) if len(all_corr) > 0 else np.nan

    mask = ((all_mean < -0.2) | (all_mean > 0.2)) & (all_mean > -5)
    all_mean = all_mean[mask]
    all_loss = all_loss[mask]

    mask1 = all_mean < -1
    mask2 = (all_mean < -0.5) & (all_mean >= -1)
    mask3 = (all_mean < 0) & (all_mean >= -0.5)
    mask4 = (all_mean >= 0) & (all_mean < 0.5)
    mask5 = (all_mean >= 0.5) & (all_mean < 1.0)
    mask6 = (all_mean >= 1)
    bin_masks = [mask1, mask2, mask3, mask4, mask5, mask6]

    bin_means = []
    bin_stds = []
    for mask in bin_masks:
        loss_subset = all_loss[mask]
        bin_means.append(np.mean(loss_subset) if len(loss_subset) > 0 else np.nan)
        bin_stds.append(np.std(loss_subset) if len(loss_subset) > 0 else 0)
    return bin_means, bin_stds, corr, all_mean, bin_masks

def compare_methods_by_bin(results_dict, save_dir='./plots'):
    os.makedirs(save_dir, exist_ok=True)
    bin_labels = ["< -1", "[-1, -0.5)", "[-0.5, 0)", "[0, 0.5)", "[0.5, 1)", ">= 1"]

    method_names = []
    all_bin_means = []
    all_bin_stds = []
    all_corrs = []

    for method, result in results_dict.items():
        bin_means, bin_stds, corr, all_mean, bin_masks = collect_bin_means(result)
        all_bin_means.append(bin_means)
        all_bin_stds.append(bin_stds)
        all_corrs.append(corr)
        method_names.append(method)

    all_bin_means = np.array(all_bin_means)  # (n_methods, n_bins)
    all_bin_stds = np.array(all_bin_stds)    # (n_methods, n_bins)

    # After all_bin_means and all_bin_stds are ready
    x = np.arange(len(method_names))
    
    for b in range(len(bin_labels)):
        plt.figure(figsize=(7, 6))
        means = all_bin_means[:, b]
        stds = all_bin_stds[:, b]

        # Get GT stats for this bin
        gt_mask = bin_masks[b]
        gt_vals = all_mean[gt_mask]
        gt_min = np.nanmin(gt_vals) if len(gt_vals) > 0 else float('nan')
        gt_mean = np.nanmean(gt_vals) if len(gt_vals) > 0 else float('nan')
        gt_max = np.nanmax(gt_vals) if len(gt_vals) > 0 else float('nan')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.bar(
            x, 
            means, 
            yerr=stds, 
            capsize=8, 
            tick_label=method_names, 
            alpha=0.85, 
            color=plt.cm.tab10.colors[:len(method_names)]
        )
        plt.ylabel("Mean Error", fontsize=15)
        plt.title(f"Mean Error ± Std for Bin: {bin_labels[b]}", fontsize=15)
        plt.suptitle(
            f"GT min: {gt_min:.2f} | GT mean: {gt_mean:.2f} | GT max: {gt_max:.2f}",
            x=0.5, y=0.97, ha='center', fontsize=15, color='black'
        )

        plt.ylim([
            np.nanmin(means - stds) - 0.1 * np.nanmax(np.abs(means - stds)),
            np.nanmax(means + stds) + 0.1 * np.nanmax(np.abs(means + stds))
        ])
        plt.tight_layout()
        plt.savefig(f"{save_dir}/mean_error_bin_{b+1}_{bin_labels[b].replace(' ', '').replace('[','').replace(']','').replace(',','_').replace('<','lt').replace('>=','gte').replace('-','m')}.png")
        plt.close()

    # Print or log correlation for each method
    for method, corr in zip(method_names, all_corrs):
        print(f"{method}: Overall correlation = {corr:.4f}")

def load_and_log():
    results_dict = {
        "IDW2": 'model.distance.InverseDistance2-results.pkl',
        "IDW": 'model.distance.InverseDistance-results.pkl',
        "POK": 'rainfall-OK-results.pkl',
        "GNN": 'g-model.gnn.GATWithEdgeAttr-results.pkl',
        "OK": 'model.kriging.OrdinaryKrigingInterpolation-results.pkl',
        "MLP": 'model.mlp.MLPW-results.pkl',
    }
    for k in list(results_dict.keys()):
        with open(results_dict[k], 'rb') as f:
            results_dict[k] = pickle.load(f)

    compare_methods_by_bin(results_dict)


folder = './'

load_and_log()
# load_and_log(os.path.join(folder, 'model.mlp.MLP-results.pkl'), 'mlp')

# load_and_log(os.path.join(folder, 'model.kriging.OrdinaryKrigingInterpolation-results.pkl'), 'ok')
# load_and_log(os.path.join(folder, 'model.mlp.MLPW-results.pkl'), 'mlpw')
# # load_and_log(os.path.join(folder, 'model.mlp.MLPRW-results.pkl'), 'mlprw')
# # load_and_log(os.path.join(folder, 'model.mlp.MLPR-results.pkl'), 'mlpr')
# load_and_log(os.path.join(folder, 'g-model.gnn.GATWithEdgeAttr-results.pkl'), 'gnn')
# load_and_log(os.path.join(folder, 'model.distance.InverseDistance-results.pkl'), 'idw')
# # load_and_log(os.path.join(folder, 'model.kriging.UniversalKrigingInterpolation-results.pkl'), 'uk')
# load_and_log(os.path.join(folder, 'model.nngp.SpatioTemporalNNGP.pkl-results.pkl'), 'nngp')
