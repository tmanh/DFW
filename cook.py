import logging
import pickle
import argparse
import time

import torch

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

from model.mlp import *
from sklearn.cluster import KMeans

from dataloader import *
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from torchmetrics.functional import pearson_corrcoef

from model.cokriging import *


logging.basicConfig(filename="log-test-all.txt", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def plot_graph(o, y, x, idx):
    # Move tensors to CPU and convert to NumPy
    o_np = o
    y_np = y
    # x_np = x

    # Create a time axis
    time_steps = range(len(y_np))

    # Create and save plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, y_np, label='Ground Truth (y)', marker='o')
    # for i in range(x_np.shape[1]):
    #     plt.plot(time_steps, x_np[:, i], label=f'Source ({i})', marker='o')
    plt.plot(time_steps, o_np, label='Interpolated (o)', marker='x')
    plt.title('Comparison of Ground Truth and Interpolated Time Series')
    plt.xlabel('Time Step')
    plt.ylabel('Water Level')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save to file (you can change path/filename as needed)
    output_path = f'results/{idx}_ked.png'
    plt.savefig(output_path)
    plt.close()

    # print(f"Plot saved to {output_path}")
    # input('Enter to continue: ')


def get_checkpoint_name(cfg):
    rn = str(cfg.model['target'])
    if rn == 'model.gru.GRU':
        mn = 'gru'
    elif rn == 'model.mlp.MLP':
        mn = 'mlp'
    elif rn == 'model.gnn.GNN':
        mn = 'gnn'
    elif rn == 'model.mlp.MLPW':
        mn = 'mlpw'
    elif rn == 'model.mlp.MLPRW':
        mn = 'mlprw'
    elif rn == 'model.mlp.MLPR':
        mn = 'mlpr'
    else:
        return None
    return f'model_{mn}.pth'


def split_stations_by_clusters(data, n_clusters=25, n_train_clusters=13, random_seed=42):
    """
    Splits coordinate-based station data into train and test sets using spatial clustering.

    Args:
        data (dict): Dictionary where keys are coordinates (lon, lat).
                     Each value must have a 'neighbor' dict with coordinate keys.
        n_clusters (int): Number of spatial clusters for KMeans.
        n_train_clusters (int): Number of clusters to include in training set.
        random_seed (int): Seed for reproducibility.

    Returns:
        good_nb_extended (set): Training coordinates (stations + neighbors).
        test_nb (list): Testing coordinates (non-overlapping with training and their neighbors).
    """

    # Extract coordinates
    station_coords = list(data.keys())
    coord_array = np.array(station_coords)

    # Step 1: Cluster all coordinates
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed).fit(coord_array)
    cluster_labels = kmeans.labels_

    # Step 2: Select clusters to be training
    chosen_clusters = random.sample(range(n_clusters), n_train_clusters)
    good_nb = [station_coords[i] for i in range(len(station_coords)) if cluster_labels[i] in chosen_clusters]

    # Step 3: Extend training set with neighbors
    good_nb_extended = set(good_nb)
    for p in good_nb:
        neighbors = data[p]['graph'].keys()
        for nb in neighbors:
            if nb not in good_nb_extended:
                good_nb_extended.add(nb)

    # Step 4: Define test set as all coords not in training+neighbors
    test_nb = [p for p in station_coords if p not in good_nb_extended]

    return good_nb_extended, test_nb


def neighbor_degradation_penalty_loss(o, x, y, alpha=1.0):
    """
    o: (B, N, T) - model outputs per neighbor
    x: (B, N, T) - raw neighbor inputs
    y: (B, T)    - ground truth sequence
    alpha: penalty weight
    """
    B, N, T = x.shape

    # Expand y to (B, N, T)
    y_un = y.unsqueeze(1)
    y_exp = y_un.expand(-1, N, -1)

    # Main MSE loss: how close each o is to y
    main_loss = F.mse_loss(o, y_un)

    # Per-sample, per-neighbor degradation penalty
    model_error = F.mse_loss(o, y_un, reduction='none')
    input_error = F.mse_loss(x, y_exp, reduction='none')

    # Only penalize where model is worse than raw input
    penalty = torch.relu(model_error - input_error).mean()  # (B, N)

    # Final loss = base + penalty
    return main_loss + alpha * penalty.mean()


def pearson_corrcoef(x, y, dim=-1, eps=1e-8):
    """
    Compute Pearson correlation between x and y along given dimension.
    Assumes x and y are of the same shape.
    """
    x_centered = x - x.mean(axis=dim, keepdims=True)
    y_centered = y - y.mean(axis=dim, keepdims=True)

    cov = (x_centered * y_centered).mean(axis=dim)
    std_x = x_centered.std(axis=dim)
    std_y = y_centered.std(axis=dim)

    corr = cov / (std_x * std_y + eps)
    return corr


def neighbor_degradation_penalty_loss(o, x, y, alpha=1.0):
    """
    o: (B, N, T) - model outputs per neighbor
    x: (B, N, T) - raw neighbor inputs
    y: (B, T)    - ground truth sequence
    alpha: penalty weight
    """
    B, N, T = x.shape

    # Expand y to (B, N, T)
    y_un = y.unsqueeze(1)
    y_exp = y_un.expand(-1, N, -1)

    # Main MSE loss: how close each o is to y
    main_loss = F.mse_loss(o, y_un)

    # Per-sample, per-neighbor degradation penalty
    model_error = F.mse_loss(o, y_un, reduction='none')
    input_error = F.mse_loss(x, y_exp, reduction='none')

    # Only penalize where model is worse than raw input
    penalty = torch.relu(model_error - input_error).mean()  # (B, N)

    # Final loss = base + penalty
    return main_loss + alpha * penalty.mean()


def has_significant_slope(ts, window_size=7, range_thresh=0.2, outlier_threshold=4.0):
    ts = pd.Series(ts)
    filtered = medfilt(ts, kernel_size=window_size)
    # Outlier detection
    diff = ts - filtered
    std = np.std(diff)
    z_scores = diff / (std + 1e-8)
    outlier_mask = np.abs(z_scores) > outlier_threshold
    cleaned_ts = ts.copy()
    cleaned_ts[outlier_mask] = filtered[outlier_mask]
    gain = cleaned_ts.max() - cleaned_ts.min()

    if gain > range_thresh:
        return True  # Found significant local change
    return False  # No significant local change found


def test(cfg, train=True):
    if not os.path.exists('data/split.pkl'):
        with open('data/selected_stats_rainfall_segment.pkl', 'rb') as f:
            data = pickle.load(f)

        good_nb_extended, test_nb = split_stations_by_clusters(data)
        print('Train length: ', len(good_nb_extended))
        print('Test length: ', len(test_nb))
        with open('data/split.pkl', 'wb') as f:
            pickle.dump({'train': good_nb_extended, 'test': test_nb}, f)
    else:
        with open('data/split.pkl', 'rb') as f:
            split = pickle.load(f)
            good_nb_extended = split['train']
            test_nb = split['test']

    test_dataset = WaterDatasetYAllStations(
        path='data/selected_stats_rainfall_segment.pkl',
        train=False,
        selected_stations=test_nb,
        input_type=cfg.dataset.inputs
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    results = {
        k: {'corr': [], 'loss': [], 'gain': [], 'tgts': [], 'outs': []}
        for k in range(len(list(test_dataset.data.keys())))
    }

    seg_len = 168
    total_elapsed = 0.0
    total_n = 0

    # --------- OKCK hyperparams (tune!) ----------
    okck_params = dict(
        corr_model="gaussian",
        range_=4.0,
        sill1=2.0,        # primary sill
        sill2=1.0,        # secondary sill
        r12=0.5,          # cross-corr (-1..1)  IMPORTANT
        nugget1=0.0,
        nugget2=0.0,
        eps=1e-10,
        use_collocated_secondary=True,  # uses lrain at target as an extra secondary point
    )

    with torch.no_grad():
        for idx, (mxs, my, mlrain, mnbxs, mnby, mnrain, loc) in enumerate(test_loader):
            outs = []
            tgts = []
            cors = []
            gain = []

            for i in range(my.shape[-1] // seg_len):
                start_time = time.time()

                # segment slices (torch)
                y = my[:, i*seg_len:(i+1)*seg_len]                    # (1,Tseg)
                nby = mnby[:, :, i*seg_len:(i+1)*seg_len]            # (1,K,Tseg)
                nxs = mnbxs                                           # (1,K,D)
                xs = mxs                                              # (1,D)
                lrain = mlrain[:, i*seg_len:(i+1)*seg_len]            # (1,Tseg)
                nrain = mnrain[:, :, i*seg_len:(i+1)*seg_len]         # (1,K,Tseg)

                y_np = y.detach().cpu().numpy()  # (1,Tseg)

                loc_vals = np.expand_dims(y_np[0], axis=-1)[:, 0]
                if not has_significant_slope(loc_vals):
                    continue
                delta = abs(np.max(loc_vals) - np.min(loc_vals))
                if delta <= 0.3:
                    continue

                # --------- coords (EDIT if your coord columns differ) ----------
                nxs_np = nxs.detach().cpu().numpy()   # (1,K,Dfeat)
                xs_np = xs.detach().cpu().numpy()     # (1,Dfeat)

                K, Dfeat = nxs_np[0].shape
                if Dfeat < 2:
                    xcoord = nxs_np[0][:, 0]
                    ycoord = np.zeros_like(xcoord)
                    target_xy = np.array([float(xs_np[0, 0]), 0.0], dtype=float)
                else:
                    xcoord = nxs_np[0][:, 0]
                    ycoord = nxs_np[0][:, 1]
                    target_xy = np.array([float(xs_np[0, 0]), float(xs_np[0, 1])], dtype=float)

                # neighbor coords (K,2)
                nxy = np.column_stack([xcoord, ycoord]).astype(np.float64)

                # values (numpy)
                nby_np = nby[0].detach().cpu().numpy().astype(np.float64)    # (K,Tseg)
                nrain_np = nrain[0].detach().cpu().numpy().astype(np.float64)# (K,Tseg)
                lrain_np = lrain[0].detach().cpu().numpy().astype(np.float64)# (Tseg,)

                # --------- OKCK prediction for this segment ----------
                try:
                    pred_ts = fast_okck_time_series(
                        nxs=nxy,
                        xs=target_xy,
                        nby=nby_np,
                        nrain=nrain_np,
                        lrain=lrain_np,
                        **okck_params
                    )
                except Exception:
                    # fallback: IDW (same as your typical fallback)
                    # weights ~ 1 / dist
                    tx, ty = float(target_xy[0]), float(target_xy[1])
                    d = np.sqrt((nxy[:, 0] - tx)**2 + (nxy[:, 1] - ty)**2)
                    d = np.maximum(d, 1e-12)
                    w = 1.0 / d
                    w = w / (np.sum(w) + 1e-12)
                    pred_ts = w @ nby_np  # (Tseg,)

                o = pred_ts.reshape(1, -1).astype(float)  # (1,Tseg)
                y_seg = y_np.astype(float)

                total_elapsed += time.time() - start_time
                total_n += 1

                # post-process
                o1 = suppress_spike_segments(o.flatten())
                y1 = y_seg.flatten()

                outs.extend(o1.tolist())
                tgts.extend(y1.tolist())
                cors.append(pearson_corrcoef(o1, y1))
                gain.extend((y1 - o1).tolist())

                print(idx)

            outs = np.array(outs)
            tgts = np.array(tgts)
            if outs.shape[0] > 0:
                se = (outs - tgts) ** 2
                cor = pearson_corrcoef(outs, tgts)
                results[idx]['corr'].append(cor)
                results[idx]['loss'].append(se)
                results[idx]['gain'].append(gain)
                results[idx]['tgts'].append(tgts)
                results[idx]['outs'].append(outs)

    print(f"Total Elapsed: {total_elapsed:.6f} seconds, Average time per segment: {total_elapsed / max(total_n,1):.6f} seconds")

    ckpt_name = 'model.nngp.CoKriging.pkl'
    with open(f'{ckpt_name}-results.pkl', 'wb') as f:
        pickle.dump(results, f)


def suppress_spike_segments(
    signal,
    z_thresh=6.0,
    max_segment_len=8,
    context=5
):
    """
    Suppress short contiguous spike segments using local median.

    Parameters
    ----------
    signal : 1D np.ndarray
    z_thresh : float
        MAD-based outlier threshold
    max_segment_len : int
        Maximum length of spike segment to correct
    context : int
        Number of neighbors on each side for median replacement

    Returns
    -------
    filtered : np.ndarray
    """
    x = signal.copy()
    n = len(x)

    # Robust stats
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-8
    z = np.abs(x - med) / mad

    is_outlier = z > z_thresh

    i = 0
    while i < n:
        if not is_outlier[i]:
            i += 1
            continue

        # Find contiguous segment
        start = i
        while i < n and is_outlier[i]:
            i += 1
        end = i  # [start, end)

        seg_len = end - start

        # Only fix *short* pathological bursts
        if seg_len <= max_segment_len:
            left = max(0, start - context)
            right = min(n, end + context)

            neighborhood = np.concatenate([
                x[left:start],
                x[end:right]
            ])

            if len(neighborhood) > 0:
                replacement = np.median(neighborhood)
                x[start:end] = replacement

    return x


def main():
    torch.manual_seed(42)  # For reproducibility
    np.random.seed(42)  # For reproducibility
    random.seed(42)  # For reproducibility

    parser = argparse.ArgumentParser(description="Run the test function with configurable parameters.")
    parser.add_argument("--cfg", type=str, help="Config file path", default="config/idw.yaml")
    parser.add_argument("--training", action="store_true", help="Enable training mode (default: False)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    print('-----Testing-----')
    test(cfg, train=False)


if __name__ == "__main__":
    main()
