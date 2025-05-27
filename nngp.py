import logging
import pickle
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from sklearn.gaussian_process.kernels import RBF, Matern

from common.utils import instantiate_from_config
from model.mlp import *
from sklearn.cluster import KMeans

from dataloader import *
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from tqdm import tqdm
from torchmetrics.functional import pearson_corrcoef

from model.nngp import SpatioTemporalNNGP


logging.basicConfig(filename="log-test-all.txt", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def plot_graph(o, y, x):
    print(o.shape, y.shape, x.shape)
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
    output_path = './interpolation_vs_groundtruth.png'
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved to {output_path}")
    input('Enter to continue: ')


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


def has_significant_slope(ts, window_size=3, range_thresh=0.15):
    ts = np.array(ts)
    if len(ts) < window_size:
        return False
    
    for i in range(len(ts) - window_size + 1):
        window = ts[i:i+window_size]
        if (np.max(window) - np.min(window)) >= range_thresh:
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
            pickle.dump(
                {'train': good_nb_extended, 'test': test_nb},
                f
            )
    else:
        with open('data/split.pkl', 'rb') as f:
            split = pickle.load(f)
            good_nb_extended = split['train']
            test_nb = split['test']

    train_dataset = WaterDatasetX(
        path='data/selected_stats_rainfall_segment.pkl', train=True,
        selected_stations=good_nb_extended, testing_stations=test_nb
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Define spatial kernel (e.g., Matern)
    # kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-6, 1e3))
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
    model = SpatioTemporalNNGP(
        kernel=kernel,
        alpha=1e-6,  # jitter for numerical stability
        phi=0.8,     # autoregressive temporal parameter (typically between -1 and 1)
        tau2=0.1     # observation noise variance
    )

    # Training loop
    ckpt_name = 'model.nngp.SpatioTemporalNNGP.pkl'

    if not os.path.exists(ckpt_name):
        all_xs = []
        all_y = []
        for xs, y, lrain in tqdm(train_loader, desc=f"Training", leave=False):
            y = y.squeeze(0).detach().cpu().numpy()
            xs = xs.repeat(lrain.shape[1], 1).detach().cpu().numpy()
            lrain = lrain.squeeze(0).unsqueeze(-1).detach().cpu().numpy()

            all_xs.append(np.concatenate([xs, lrain], axis=-1))
            all_y.append(y)

        X_flat = np.concatenate(all_xs, axis=0)
        y_flat = np.concatenate(all_y, axis=0)
        model.fit(X_flat, y_flat.reshape(-1, 1))

        with open(ckpt_name, 'wb') as f:
            pickle.dump(model, f)
    else:
        with open(ckpt_name, 'rb') as f:
            model = pickle.load(f)
    
    test_dataset = WaterDatasetY(
        path='data/selected_stats_rainfall_segment.pkl', train=False,
        selected_stations=test_nb, input_type=cfg.dataset.inputs
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    results = {
        k:{
            'corr': [],
            'loss': [],
            'mean': [],
        } for k in range(len(list(test_dataset.data.keys())))
    }

    with torch.no_grad():
        for idx, (mxs, my, mlrain, mnbxs, mnby, mnrain) in enumerate(test_loader):
            outs = []
            tgts = []
            for i in range(my.shape[-1] // 168):
                y = my[:, i*168:(i+1)*168].detach().cpu().numpy()
                nby = mnby[:, :, i*168:(i+1)*168].detach().cpu().numpy()
                nxs = mnbxs.detach().cpu().numpy()
                xs = mxs.detach().cpu().numpy()
                lrain = mlrain[:, i*168:(i+1)*168].detach().cpu().numpy()
                nrain = mnrain[:, :, i*168:(i+1)*168].detach().cpu().numpy()
                
                y = np.expand_dims(y[0], axis=-1)
                xs = xs.repeat(y.shape[0], axis=0)
                lrain = np.expand_dims(lrain[0], axis=-1)
                all_xs = np.concatenate([xs, lrain], axis=-1)

                if not has_significant_slope(y):
                    continue

                nby = np.expand_dims(nby[0], axis=-1).transpose(1, 0, 2)
                nxs = nxs.repeat(y.shape[0], axis=0)
                nrain = np.expand_dims(nrain[0], -1).transpose(1, 0, 2)
                all_nxs = np.concatenate([nxs, nrain], axis=-1)
                print(all_nxs.shape, nby.shape)
                exit()
                o = model.predict(all_nxs, nby, all_xs)[:, 0]

                # print(o.shape, x.shape, y.shape)
                # print(o.shape, x.shape)
                # exit()
                # from calflops import calculate_flops
                # inputs = {}
                # inputs["xs"] = xs
                # inputs["x"] = x
                # # inputs["inputs"] = cfg.dataset.inputs
                # # inputs["train"] = False
                # # inputs["stage"] = -1
                # flops, macs, params = calculate_flops(
                #     model=model, 
                #     kwargs=inputs,
                #     output_as_string=True,
                #     output_precision=4
                # )
                # print("FLOPs:%s  MACs:%s  Params:%s \n" %(flops, macs, params))
                # # FLOPs:31.872 KFLOPS  MACs:15.744 KMACs  Params:5.442 K 
                # exit()
                # o1 = idw(xs, x, inputs=cfg.dataset.inputs, train=False, stage=-1)

                outs.extend(
                    o.flatten()
                )
                tgts.extend(
                    y.flatten()
                )

                # plot_graph(o, y, nby[:, :3])

            outs = np.array(outs)
            tgts = np.array(tgts)
            if outs.shape[0] > 0:
                se = (outs - tgts) ** 2
                print(np.mean(se))

                corr = pearson_corrcoef(
                    outs,
                    tgts
                )
                results[idx]['corr'].append(corr)
                results[idx]['loss'].append(se)
                results[idx]['mean'].append(tgts)


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

    if args.training:
        print('-----Training-----')
        test(cfg, train=True)
        
    print('-----Testing-----')
    test(cfg, train=False)


if __name__ == "__main__":
    main()
