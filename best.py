import logging
import pickle
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

from scipy.optimize import minimize

from model.mlp import *
from sklearn.cluster import KMeans

from dataloader import *
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from torchmetrics.functional import pearson_corrcoef



logging.basicConfig(filename="log-test-all.txt", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def plot_graph(o, y, x, idx):
    # Move tensors to CPU and convert to NumPy
    o_np = o
    y_np = y
    x_np = x

    # Create a time axis
    time_steps = range(len(y_np))

    # Create and save plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, y_np, label='Ground Truth (y)', marker='o')
    for i in range(x_np.shape[1]):
        plt.plot(time_steps, x_np[:, i], label=f'Source ({i})', marker='o')
    plt.plot(time_steps, o_np, label='Interpolated (o)', marker='x')
    plt.title('Comparison of Ground Truth and Interpolated Time Series')
    plt.xlabel('Time Step')
    plt.ylabel('Water Level')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save to file (you can change path/filename as needed)
    output_path = f'results/{idx}_nngp.png'
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


def oracle_best_interp_weights(
    y: np.ndarray,          # (T,)
    nby: np.ndarray,        # (K,T) or (T,K)
    ridge: float = 1e-8,
    max_iter: int = 500,
) -> np.ndarray:
    y = np.asarray(y, float).reshape(-1)

    N = np.asarray(nby, float)
    if N.ndim == 3:
        N = N.squeeze(-1)
    if N.shape[0] != y.shape[0] and N.shape[1] == y.shape[0]:
        N = N.T
    if N.shape[0] != y.shape[0]:
        raise ValueError(f"nby shape {N.shape} not compatible with y shape {y.shape}")

    T, K = N.shape
    if K == 0:
        return np.zeros((0,), float)

    # Precompute quadratic form: ||y - Nw||^2 + ridge||w||^2
    A = N.T @ N + ridge * np.eye(K)     # (K,K) PSD
    b = N.T @ y                         # (K,)

    def fun(w):
        # 0.5 * w^T A w - b^T w + const
        return 0.5 * (w @ (A @ w)) - (b @ w)

    def jac(w):
        return (A @ w) - b

    # simplex constraints: w>=0 and sum(w)=1
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0,
             'jac': lambda w: np.ones_like(w)})
    bounds = [(0.0, 1.0) for _ in range(K)]

    w0 = np.full(K, 1.0 / K, dtype=float)

    res = minimize(fun, w0, method='SLSQP', jac=jac,
                   bounds=bounds, constraints=cons,
                   options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False})

    w = res.x
    # numerical cleanup
    w[w < 0] = 0.0
    s = w.sum()
    if s > 1e-12:
        w /= s
    else:
        w = np.full(K, 1.0 / K, dtype=float)
    return w


def oracle_best_interp_predict(
    y: np.ndarray,          # (T,)
    nby: np.ndarray,        # (K,T) or (T,K)
    ridge: float = 1e-8,
    sum_to_one: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      o: (T,) oracle best interpolated series
      w: (K,) constant weights
    """
    y = np.asarray(y, float).reshape(-1)
    N = np.asarray(nby, float)

    if N.ndim == 3:
        N = N.squeeze(-1)
    if N.shape[0] != y.shape[0] and N.shape[1] == y.shape[0]:
        N = N.T
    # now N is (T,K)
    w = oracle_best_interp_weights(y, N, ridge=ridge)
    o = N @ w
    return o, w


def test(cfg, train=True):
    if not os.path.exists('data/split.pkl'):
        with open('data/selected_stats_rainfall_segment.pkl', 'rb') as f:
            data = pickle.load(f)

        good_nb_extended, test_nb = split_stations_by_clusters(data)
        print('Train length: ', len(good_nb_extended))
        print('Test length: ', len(test_nb))
        with open('data/split.pkl', 'wb') as f:
            pickle.dump(
                {'train': good_nb_extended, 'test': test_nb},
                f
            )
    else:
        with open('data/split.pkl', 'rb') as f:
            split = pickle.load(f)
            good_nb_extended = split['train']
            test_nb = split['test']

    # Training loop
    ckpt_name = 'model.best.possible.pkl'
    
    test_dataset = WaterDatasetY(
        path='data/selected_stats_rainfall_segment.pkl', train=False,
        selected_stations=test_nb, input_type=cfg.dataset.inputs
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    results = {
        k:{
            'corr': [],
            'loss': [],
            'gain': [],
            'tgts': [],
            'outs': []
        } for k in range(len(list(test_dataset.data.keys())))
    }

    seg_len = 168
    total_elapsed = 0
    total_n = 0
    with torch.no_grad():
        for idx, (mxs, my, mlrain, mnbxs, mnby, mnrain, loc) in enumerate(test_loader):
            outs = []
            tgts = []
            cors = []
            gain = []
            for i in range(my.shape[-1] // seg_len):
                start_time = time.time()
                y = my[:, i*seg_len:(i+1)*seg_len].detach().cpu().numpy()
                nby = mnby[:, :, i*seg_len:(i+1)*seg_len].detach().cpu().numpy()
                nxs = mnbxs.detach().cpu().numpy()
                xs = mxs.detach().cpu().numpy()
                lrain = mlrain[:, i*seg_len:(i+1)*seg_len].detach().cpu().numpy()
                nrain = mnrain[:, :, i*seg_len:(i+1)*seg_len].detach().cpu().numpy()
                
                y = np.expand_dims(y[0], axis=-1)
                xs = xs.repeat(y.shape[0], axis=0)
                lrain = np.expand_dims(lrain[0], axis=-1)
                all_xs = np.concatenate([xs, lrain], axis=-1)

                loc_vals = y[:, 0]
                if not has_significant_slope(loc_vals):
                    continue
                delta = abs(np.max(loc_vals) - np.min(loc_vals))
                if delta <= 0.3:
                    continue

                nby = np.expand_dims(nby[0], axis=-1).transpose(1, 0, 2)
                nxs = nxs.repeat(y.shape[0], axis=0)
                nrain = np.expand_dims(nrain[0], -1).transpose(1, 0, 2)
                all_nxs = np.concatenate([nxs, nrain], axis=-1)

                # y: (T,1) currently
                y_vec = y[:, 0]  # (T,)

                # nby currently after your transform is something like (T,K,1) or (K,T,1)
                N = nby  # keep name aligned
                o, w = oracle_best_interp_predict(
                    y=y_vec,
                    nby=N,
                    ridge=1e-6,       # small stabilizer
                    sum_to_one=True,  # interpolation constraint
                )
                print(w)
                rnb = np.squeeze(nrain, axis=-1).transpose(1, 0)
                rvt = np.transpose(lrain, (1, 0))
                print(pearson_corrcoef(rnb, rvt))
                total_elapsed += time.time() - start_time
                total_n += 1

                o = o.flatten()
                y = y.flatten()

                outs.extend(
                    o
                )
                tgts.extend(
                    y
                )
                cors.append(
                    pearson_corrcoef(
                        o,
                        y
                    )
                )
                gain.extend(
                    (y - o).tolist()
                )

                if np.abs(y).max() > 0.5 and np.abs(y).max() < 1.0:
                    plot_graph(o, y, nby, f'{loc.numpy()}-{idx}-{i}')

            outs = np.array(outs)
            tgts = np.array(tgts)
            if outs.shape[0] > 0:
                se = (outs - tgts) ** 2
                cor = pearson_corrcoef(
                    outs,
                    tgts
                )
                results[idx]['corr'].append(cor)
                results[idx]['loss'].append(se)
                results[idx]['gain'].append(gain)
                results[idx]['tgts'].append(tgts)
                results[idx]['outs'].append(outs)

    print(f'Total Elapsed: {total_elapsed:.6f} seconds, Average time per segment: {total_elapsed / total_n:.6f} seconds')

    with open(f'{ckpt_name}-results.pkl', 'wb') as f:
        pickle.dump(results, f)


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
