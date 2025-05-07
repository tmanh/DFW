import os
import logging
import pickle
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


if os.path.exists('log-test-all.txt'):
    os.remove('log-test-all.txt')
logging.basicConfig(filename="log-test-all.txt", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def plot_error_distribution(errors, label=None, kind='hist', save_path=None):
    """
    Save the distribution plot of a list of errors.

    Parameters:
    - errors (list or array): List of error values.
    - label (str): Optional label for plot title.
    - kind (str): 'hist', 'box', or 'violin'.
    - save_path (str): File path to save the plot (e.g., 'plots/kc_025.png').
                       If None, nothing is saved.
    """
    if save_path is None:
        return  # silently skip if no save path provided

    counts, bins = np.histogram(errors, bins=100)
    probs = counts / counts.sum()  # normalize to sum = 1
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    plt.figure(figsize=(10, 4))
    plt.bar(bin_centers, probs, width=(bins[1] - bins[0]), edgecolor='black', alpha=0.7)
    
    # plt.xlim(0, 1.0)  # Zoom in to range where most errors lie
    # plt.xticks(np.arange(0, 100, 0.1))  # Tick marks at 0.0, 0.1, ..., 1.0
    
    plt.xlabel("Error Value")
    plt.ylabel("Relative Frequency")
    plt.title(f"Normalized Histogram for {label}")
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def analyze_error_brackets(result, method="default", save_dir="./plots"):
    all_corr = []
    all_mean = []
    all_best = []
    all_loss = []

    # Flatten values across keys
    for k in result.keys():
        all_corr.extend(result[k]['corr'])
        all_mean.extend(result[k]['mean'])  # GT values
        all_best.extend(result[k]['best'])  # Nearest-station prediction
        all_loss.extend(result[k]['loss'])  # Interpolated errors
        if len(result[k]['best']) > 0:
            print(len(result[k]['best']), np.min(result[k]['best']), np.max(result[k]['best']))
        else:
            print(0, 0, 0)
    all_corr = np.array(all_corr)
    all_mean = np.array(all_mean)
    all_best = np.array(all_best)
    all_loss = np.array(all_loss)

    # Bin 1: mean < 0.2
    mask_bin1 = all_mean < 0.2
    bin_masks = [mask_bin1]

    # Remaining values: mean ≥ 0.2
    remaining_mean = all_mean[~mask_bin1]
    if len(remaining_mean) > 0:
        mean_qs = np.quantile(remaining_mean, [0.25, 0.5, 0.75])
        prev = 0.2
        for q in mean_qs:
            mask = (all_mean >= prev) & (all_mean < q)
            bin_masks.append(mask)
            prev = q
        # Final bin: >= last quantile
        mask = (all_mean >= mean_qs[-1])
        bin_masks.append(mask)

    logging.info(f'{method} — Total samples: {len(all_mean)}')

    for i, mask in enumerate(bin_masks):
        mean_subset = all_mean[mask]
        best_subset = all_best[mask]
        loss_subset = all_loss[mask]
        corr_subset = all_corr[mask]

        bin_range = f"mean_bin{i+1}"
        mean_gt = mean_subset.mean() if len(mean_subset) > 0 else np.nan
        logging.info(f"{bin_range}, count = {len(mean_subset)}, mean of GT = {mean_gt:.4f}")

        # Compute quantiles for best values in this mean bin
        if len(best_subset) == 0:
            continue

        best_qs = np.quantile(best_subset, [0.25, 0.5, 0.75])
        for j, (min_b, max_b) in enumerate(zip([ -np.inf] + list(best_qs), list(best_qs) + [np.inf])):
            bmask = (best_subset > min_b) & (best_subset <= max_b)
            losses = loss_subset[bmask]
            corres = corr_subset[bmask]
            gt_vals = mean_subset[bmask]

            bin_tag = f"{method}_mean{i+1}_best{j+1}"
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/{bin_tag}.png"
            plot_error_distribution(errors=losses, label=bin_tag, save_path=save_path)

            if len(losses) > 0:
                mean_cor = np.mean(corres)
                mean_err = np.mean(losses)
                std_err = np.std(losses)
                mean_gt_bin = np.mean(gt_vals)
                logging.info(
                    f"  Best bin {j+1}: ({min_b:.2f}, {max_b:.2f}] — count = {len(losses)}, "
                    f"mean_cor = {mean_cor}, mean err = {mean_err:.4f}, std = {std_err:.4f}, mean GT = {mean_gt_bin:.4f}"
                )
            else:
                logging.info(f"  Best bin {j+1}: ({min_b:.2f}, {max_b:.2f}] — count = 0")

    logging.info('\n')


def load_and_log(path, method):
    with open(path, 'rb') as f:
        result = pickle.load(f)

    logging.info(f'{path}')
    analyze_error_brackets(result, method)
    print('----')


folder = './'

# load_and_log(os.path.join(folder, 'bk_checkpoint_all/model.mlp.MLPW-wo-results.pkl'))
# load_and_log(os.path.join(folder, 'bk_checkpoint_all/g-model.gru.GRU-io-results.pkl'))
# load_and_log(os.path.join(folder, 'bk_checkpoint_all/model.distance.InverseDistance-o-results.pkl'))
# load_and_log(os.path.join(folder, 'bk_checkpoint_all/model.kriging.OrdinaryKrigingInterpolation-o-results.pkl'))

load_and_log(os.path.join(folder, 'model.mlp.MLPW-wo-results.pkl'), 'mlp')
load_and_log(os.path.join(folder, 'g-model.gru.GRU-io-results.pkl'), 'gnn')
load_and_log(os.path.join(folder, 'model.distance.InverseDistance-o-results.pkl'), 'idw')
load_and_log(os.path.join(folder, 'model.kriging.OrdinaryKrigingInterpolation-o-results.pkl'), 'ok')
load_and_log(os.path.join(folder, 'model.rbf.RBF-o-results.pkl'), 'rbf')