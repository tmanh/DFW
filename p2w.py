import logging
import pickle
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from common.utils import instantiate_from_config
from model.mlp import *
from sklearn.cluster import KMeans

from dataloader import *
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from tqdm import tqdm
from torchmetrics.functional import pearson_corrcoef
from scipy.signal import medfilt

from rgwater import std_loss


logging.basicConfig(filename="log-test-all.txt", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def plot_graph(o, y, x, idx):
    # Move tensors to CPU and convert to NumPy
    o_np = o.detach().cpu().numpy().flatten()
    y_np = y.detach().cpu().numpy().flatten()
    if len(x.shape) == 3:
        x = x.squeeze(0)
    x_np = x.detach().cpu().numpy()

    # Create a time axis
    time_steps = range(len(y_np))

    # Create and save plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, y_np, label='Ground Truth (y)', marker='o')
    for i in range(x_np.shape[0]):
        plt.plot(time_steps, x_np[i], label=f'Source ({i})', marker='o')
    plt.plot(time_steps, o_np, label='Interpolated (o)', marker='x')
    plt.title('Comparison of Ground Truth and Interpolated Time Series')
    plt.xlabel('Time Step')
    plt.ylabel('Water Level')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save to file (you can change path/filename as needed)
    output_path = f'results/{idx}.png'
    plt.savefig(output_path)
    plt.close()

    # print(f"Plot saved to {output_path}")
    # input('Enter to continue: ')


def get_checkpoint_name(cfg):
    rn = str(cfg.model['target']).split('.')[-1].lower()
    return f'model_p2w_{rn}.pth'


def split_stations_by_clusters(data, n_clusters=25, n_train_clusters=7, random_seed=42):
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
        with open('data/selected_stats_segment.pkl', 'rb') as f:
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

    # 🔹 1️⃣ Define Device (Multi-GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    train_dataset = WaterDataset(
        path='data/selected_stats_rainfall_segment.pkl', train=True,
        selected_stations=good_nb_extended, input_type=cfg.dataset.inputs
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # 🔹 2️⃣ Initialize Model
    model = instantiate_from_config(cfg.model).to(device)
    model = model.to(device)

    mse_loss = nn.MSELoss()

    # Training loop
    num_epochs = 50

    ckpt_name = get_checkpoint_name(cfg=cfg)
    if train:
        list_loss = []
        epoch_bar = tqdm(range(num_epochs), desc="Epochs")  # Initialize tqdm for epochs
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in epoch_bar:
            for x, xs, y, kes, lrain, nrain, valid, loc in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
                x = x.to(device)
                nrain = nrain.to(device)

                # Forward pass
                o = model(x, nrain)

                # Compute L1 loss on all elements
                loss = mse_loss(o, x)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                    
                # Print loss for every epoch
                epoch_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
                list_loss.append(loss.item())

            # Save model checkpoint every 5 epochs
            torch.save(model.state_dict(), ckpt_name)
        
    if ckpt_name is not None:
        model.load_state_dict(torch.load(ckpt_name), strict=False)
        model.eval()

    rn = cfg.model['target']
    test_dataset = WaterDataset(
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
            'outs': [],
        } for k in range(len(list(test_dataset.data.keys())))
    }

    seg_len = 168
    total_elapsed = 0
    total_n = 0
    loss_all = []
    corr_all = []
    with torch.no_grad():
        for idx, (mx, _, my, _, _, mnrain, _, _) in enumerate(test_loader):
            start = time.time()
            outs = []
            tgts = []
            gain = []
            for i in range(mx.shape[2] // seg_len):
                start = time.time()
                x = mx[:, :, i*seg_len:(i+1)*seg_len].to(device)
                nrain = mnrain[:, :, i*seg_len:(i+1)*seg_len].to(device)
                y = my[:, i*seg_len:(i+1)*seg_len].to(device)

                if not has_significant_slope(y[0].detach().cpu().numpy()):
                    continue

                if y[0].max() < 0.5:
                    continue

                o = model(x, nrain)

                elapsed = time.time() - start
                total_elapsed += elapsed
                total_n += 1

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

                o_np = o.flatten().detach().cpu().numpy()
                x_np = x.flatten().detach().cpu().numpy()

                outs.extend(
                    o_np
                )
                tgts.extend(
                    x_np
                )
                gain.extend(
                    (x_np - o_np).tolist()
                )

                # if torch.abs(y).max() > 0.5 and torch.abs(y).max() < 1.0:
                #     plot_graph(o, y, x, f'{loc.numpy()}-{idx}-{i}_{rn}')

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
                corr_all.append(cor)
                loss_all.extend(se)

    print(f'Total Elapsed: {total_elapsed:.6f} seconds, Average time per segment: {total_elapsed / total_n:.6f} seconds')
    loss = np.array(loss_all)
    corr = np.array(corr_all)
    print(np.mean(loss), np.mean(corr))
    # stm: 0.34381017 0.354788613727146
    # cgru: 0.32168755 0.4462470519091705
    # gru: 0.33668485 0.4309533541763729
    # mlp: 0.33610648 0.41939733343606117
    # mlp2: 0.3301921 0.3744572715679498

    # mlp 0.5874386 0.34767955272802337
    # mlp2 0.5836547 0.3314928786040023
    # min_gru 0.5867267 0.36448541480162644
    # gru 0.58060926 0.39438870440330126
    # with open(f'{rn}-results.pkl', 'wb') as f:
    #     pickle.dump(results, f)


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
        
    # print('-----Testing-----')
    test(cfg, train=False)


if __name__ == "__main__":
    main()
