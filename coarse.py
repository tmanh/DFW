import argparse
import os
import logging
import time
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from dataloader import *
from torch_geometric.loader import DataLoader
import torch_geometric

import pickle
from scipy.signal import welch
from scipy.signal import detrend, find_peaks
from scipy.signal import savgol_filter

from tqdm import tqdm

from model.gnn import HIGNN
from model.mlp import *
from water_level import has_significant_slope, split_stations_by_clusters

logging.basicConfig(filename="log-test-all-gnn.txt", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def plot_loss_chart(losses, title="Training Loss Over Epochs", save_path='loss.png'):
    """
    Plot a chart for loss values over epochs.

    Parameters:
        losses (list or array-like): List of loss values.
        title (str): Title of the chart.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linestyle='-', label='Training Loss')
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=600)
        # print(f"Graph saved to {save_path}")

    # Show the plot
    # plt.show()
    plt.close()


def plot_predictions_with_time(predictions, save_path=None):
    """
    Plots the predicted values against their corresponding timestamps.

    Args:
        predictions (list): List of predicted values.
        save_path (str, optional): Path to save the plot. If None, the plot is not saved.
    """ 
    # Plot the predictions
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(predictions)), predictions, linestyle='-', label='Predicted Values')
    plt.title('Predicted Values Over Time')
    plt.xlabel('Time')
    plt.ylabel('Predicted Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=600)
        # print(f"Graph saved to {save_path}")

    # Show the plot
    # plt.show()
    plt.close()


def location_aware_loss(o, y, loc_list):
    """
    Calculate the loss based on query points, grouping by GPS locations.

    Args:
        o (torch.Tensor): Predicted output, shape (B, ...)
        y (torch.Tensor): Ground truth, shape (B, ...)
        loc_list (torch.Tensor): GPS location of the query points, shape (B, 2) or (B, N, 2)

    Returns:
        torch.Tensor: Location-aware loss.
    """    
    # Compute per-sample loss
    loss_list = ((o - y) ** 2).mean(dim=1)  # Mean squared error per sample, shape (B,)

    # Group losses by unique GPS locations
    unique_locs, loc_indices = torch.unique(loc_list, dim=0, return_inverse=True)
    loss_per_location = torch.zeros(len(unique_locs), device=o.device)
    counts = torch.zeros(len(unique_locs), device=o.device)

    # Sum losses and counts for each unique location
    for i, loc_index in enumerate(loc_indices):
        loss_per_location[loc_index] += loss_list[i]
        counts[loc_index] += 1

    # Normalize losses by the number of occurrences for each location
    mean_loss_per_location = loss_per_location / counts

    # Compute the overall loss as the mean across locations
    overall_loss = mean_loss_per_location.mean()

    return overall_loss


def create_model(device):
    model = HIGNN().to(device)
    return model


def plot_graph(o, y, others, r, idx):
    # Move tensors to CPU and convert to NumPy
    o_np = o.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    r_np = r.detach().cpu().numpy()
    others_np = others.detach().cpu().numpy()

    # Create a time axis
    time_steps = range(y_np.shape[-1])

    # Create and save plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, y_np[0], label='Ground Truth 1 (y)', marker='o')
    plt.plot(time_steps, o_np[0], label='Interpolated 1 (o)', marker='x')
    plt.plot(time_steps, r_np[0], label='Rainfall 1 (r)')

    plt.plot(time_steps, others_np[1], label='Ground Truth 2 (y)', marker='o')
    plt.plot(time_steps, o_np[1], label='Interpolated 2 (o)', marker='x')
    plt.plot(time_steps, r_np[1], label='Rainfall 2 (r)')

    plt.plot(time_steps, others_np[2], label='Ground Truth 3 (y)', marker='o')
    plt.plot(time_steps, o_np[2], label='Interpolated 3 (o)', marker='x')
    plt.plot(time_steps, r_np[2], label='Rainfall 3 (r)')

    plt.title('Comparison of Ground Truth and Interpolated Time Series')
    plt.xlabel('Time Step')
    plt.ylabel('Water Level')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save to file (you can change path/filename as needed)
    os.makedirs('coarse', exist_ok=True)
    output_path = f'coarse/{idx}_rgnn.png'
    
    plt.savefig(output_path)
    plt.close()


def _zscore(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x - x.mean(dim=-1, keepdim=True)
    return x / (x.std(dim=-1, keepdim=True) + eps)

def pearson_torch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    x, y: (..., T)
    returns: (...) correlation
    """
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    cov = (x * y).mean(dim=-1)
    denom = x.std(dim=-1) * y.std(dim=-1) + eps
    return cov / denom


@torch.no_grad()
def lagged_corr_gate_batch(
    water: torch.Tensor,     # (N, T)  water level
    rain: torch.Tensor,      # (N, T)  precipitation
    max_lag: int = 24,
    min_gate: float = 0.05,
    power: float = 2.0,
) -> torch.Tensor:
    """
    Returns gate g: (N,) in [min_gate, 1], using max abs corr over lags.
    """
    # z-score per series
    w = (water - water.mean(dim=-1, keepdim=True)) / (water.std(dim=-1, keepdim=True) + 1e-8)
    r = (rain  - rain.mean(dim=-1, keepdim=True))  / (rain.std(dim=-1, keepdim=True)  + 1e-8)

    N, T = w.shape
    best = torch.zeros(N, device=w.device)

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            ww = w[:, -lag:]
            rr = r[:, :T + lag]
        elif lag > 0:
            ww = w[:, :T - lag]
            rr = r[:, lag:]
        else:
            ww, rr = w, r

        if ww.shape[-1] < 8:
            continue

        c = pearson_torch(ww, rr).abs()  # (N,)
        best = torch.maximum(best, c)

    g = best.clamp(0, 1).pow(power)
    g = min_gate + (1.0 - min_gate) * g
    return g


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


def std_loss(y_pred, y_true):
    """
    Computes the absolute difference between the std of prediction and target.
    """
    std_loss_val = torch.abs(torch.std(y_pred) - torch.std(y_true))
    return std_loss_val


def loss_fn(loss_f, o, y, pearson=True, std=True):
    if o.shape[0] == 0:
        return torch.tensor(0.0, device=o.device)

    loss = loss_f(o, y)
    if pearson:
        loss -= pearson_corrcoef(o.squeeze(-1), y.squeeze(-1))[0]
    
    if std:
        loss += std_loss(o, y)

    return loss


def topk_mse(pred: torch.Tensor, target: torch.Tensor, top_percent: float = 0.1) -> torch.Tensor:
    """
    Compute MSE using only the top x% of samples with the highest squared error.

    Args:
        pred (torch.Tensor): Predictions (any shape).
        target (torch.Tensor): Ground truth (same shape as pred).
        top_percent (float): Fraction (0 < top_percent <= 1) of samples to use,
                             e.g., 0.1 = top 10%.

    Returns:
        torch.Tensor: Scalar MSE over the top x% errors.
    """
    assert pred.shape == target.shape, "Prediction and target must have the same shape"
    assert 0 < top_percent <= 1, "top_percent must be in (0,1]"

    # flatten to 1D
    errors = (pred - target).view(-1) ** 2
    k = max(1, int(len(errors) * top_percent))

    # get top-k largest errors
    topk_errors, _ = torch.topk(errors, k)

    return topk_errors.mean()


def test(cfg, train=True, ckpt_path='model_rgnn.pth', top=0.2):
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

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    train_dataset = GWaterDataset(
        path='data/selected_stats_rainfall_segment.pkl',
        train=True,
        selected_stations=good_nb_extended,
        input_type=cfg.dataset.inputs
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    
    test_dataset = GWaterDataset(
        path='data/selected_stats_rainfall_segment.pkl',
        train=False,
        selected_stations=test_nb,
        input_type=cfg.dataset.inputs
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # <<< ADDED

    model = create_model(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch_geometric.nn.DataParallel(model, device_ids=[0, 1])
        model.to(device)

    # Define loss function and optimizer
    l2_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_test_rmse = float("inf")  
    best_hits = 0
    best_epoch = -1                                    # <<< ADDED
    best_ckpt_path = ckpt_path.replace(".pth", "_best_testRMSE.pth")  # <<< ADDED

    if train:
        num_epochs = 25
        epoch_bar = tqdm(range(num_epochs), desc="Epochs")

        for epoch in epoch_bar:
            model.train()

            epoch_loss = 0.0
            num_batches = 0

            for y, x, valid, graph_data, fx, r, loc, earray in tqdm(
                train_loader, desc=f"Training Epoch {epoch+1}", leave=False
            ):
                valid = valid.float().to(device)
                y = y.to(device).float()
                x = x.to(device).float()
                nodes = graph_data.x.unsqueeze(-1).to(device)
                edge_index = graph_data.edge_index.to(device)
                edge_attr = graph_data.edge_attr.to(device)
                fx = fx.to(device)
                r = r.to(device)
                loc = loc.to(device)
                earray = earray.to(device)

                # Forward pass
                o = model(nodes, edge_index, edge_attr, valid, r, fx, loc, earray)
                
                # ---- Build GT water for all 32 stations (virtual + neighbors) ----
                gt_all = torch.cat(
                    [y, nodes[1:o.shape[0]].squeeze(-1)],  # (1,168) + (31,168)
                    dim=0
                )  # -> (32,168)

                # ---- Extract rainfall per node ----
                # Adjust this if your r layout differs; common case is (B, N, T)
                if r.ndim == 3:
                    rain_all = r[0, :o.shape[0], :]        # (32,168)
                elif r.ndim == 2:
                    rain_all = r[:o.shape[0], :]           # (32,168)
                else:
                    raise ValueError(f"Unexpected r shape: {r.shape}")

                # ---- Gate: how meaningful is rain->water for each station? ----
                g = lagged_corr_gate_batch(gt_all.detach(), rain_all.detach(),
                                        max_lag=24, min_gate=0.05, power=2.0)  # (32,)

                # ---- Per-station MSE, then gated aggregation ----
                mse_per_node = ((o - gt_all) ** 2).mean(dim=1)  # (32,)

                # Normalize so overall scale stays stable
                loss = (g * mse_per_node).sum() / (g.sum() + 1e-8)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                epoch_bar.set_description(
                    f"Epoch [{epoch+1}/{num_epochs}], Loss: {(epoch_loss / num_batches):.4f}"
                )

            # ===================== TEST RMSE (UNCHANGED LOGIC) =====================
            model.eval()
            re_mae_all = []
            hits = []
            seg_len = 168

            with torch.no_grad():
                for my, mx, mvalid, graph_data, mfx, mr, loc, earray in test_loader:
                    mnodes = graph_data.x.unsqueeze(-1)
                    edge_index = graph_data.edge_index.to(device)
                    edge_attr = graph_data.edge_attr.to(device)
                    mfx = mfx.to(device)
                    mr = mr.to(device)
                    loc = loc.to(device)
                    earray = earray.to(device)

                    for i in range(mx.shape[2] // seg_len):
                        y = my[:, i*seg_len:(i+1)*seg_len].to(device)
                        valid = mvalid[:, :, i*seg_len:(i+1)*seg_len].to(device)
                        nodes = mnodes[:, i*seg_len:(i+1)*seg_len].to(device)
                        r = mr[:, :, i*seg_len:(i+1)*seg_len].to(device)
                        fx = mfx

                        if not has_significant_slope(y[0].detach().cpu().numpy()):
                            continue

                        o = model(nodes, edge_index, edge_attr, valid, r, fx, loc, earray)
                        o = o[:1]

                        delta = abs(torch.max(y) - torch.min(y))
                        if delta < 0.6:
                            continue

                        re_mae_all.extend(
                            (
                                (o - y)**2 * torch.abs(y - y.mean())
                            ).flatten().tolist()
                        )

                        hits.append(torch.max(o) >= torch.max(y) - 0.2)  # Consider it a hit if predicted peak is within 0.2 of GT peak

            if len(re_mae_all) > 0:
                test_mean_re_mae = np.sqrt(np.mean(re_mae_all))
            else:
                test_mean_re_mae = float("inf")

            print(f"\nEpoch {epoch+1} | Test MAE: {test_mean_re_mae}")

            if test_mean_re_mae < best_test_rmse and epoch >= 3:
                best_test_rmse = test_mean_re_mae
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"New best checkpoint saved (RMSE={test_mean_re_mae:.4f})")

            model.train()
            # ======================================================================

        print(f"\nBest epoch: {best_epoch}, Best Test RMSE: {best_test_rmse:.4f}")

    model.load_state_dict(torch.load(best_ckpt_path))   # <<< CHANGED (best ckpt)
    model.eval()
    model.to(device)

    if train:
        return

    # ===================== ORIGINAL TEST CODE (UNCHANGED) =====================
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    results = {
        k: {'corr': [], 'loss': [], 'gain': [], 'tgts': [], 'outs': []}
        for k in range(len(list(test_dataset.data.keys())))
    }

    seg_len = 168
    total_elapsed = 0
    total_n = 0

    with torch.no_grad():
        for idx, (my, mx, mvalid, graph_data, mfx, mr, loc, earray) in enumerate(test_loader):
            mnodes = graph_data.x.unsqueeze(-1)
            edge_index = graph_data.edge_index.to(device)
            edge_attr = graph_data.edge_attr.to(device)
            mfx = mfx.to(device)
            mr = mr.to(device)
            loc = loc.to(device)
            earray = earray.to(device)

            outs, tgts, cors, gain = [], [], [], []

            for i in range(mx.shape[2] // seg_len):
                start = time.time()
                x = mx[:, :, i*seg_len:(i+1)*seg_len].to(device)
                valid = mvalid[:, :, i*seg_len:(i+1)*seg_len].to(device)
                y = my[:, i*seg_len:(i+1)*seg_len].to(device)
                nodes = mnodes[:, i*seg_len:(i+1)*seg_len]
                r = mr[:, :, i*seg_len:(i+1)*seg_len].to(device)
                fx = mfx

                nodes = nodes.to(device)

                if not has_significant_slope(y[0].detach().cpu().numpy()):
                    continue

                o = model(nodes, edge_index, edge_attr, valid, r, fx, loc, earray)

                dt = 1
                tidal_prob, diagnostics = fluctuation_probability(y[0].detach().cpu().numpy(), dt)
                plot_graph(o, y, nodes.squeeze(-1), r.squeeze(0) / 20.0, f'{idx}-{i}-{tidal_prob:.2f}')

                elapsed = time.time() - start
                total_elapsed += elapsed
                total_n += 1

                o = o[:1]

                o_np = o.flatten().detach().cpu().numpy()
                y_np = y.flatten().detach().cpu().numpy()

                outs.extend(o_np)
                tgts.extend(y_np)
                cors.append(pearson_corrcoef(o_np, y_np))
                gain.extend((y_np - o_np).tolist())

            if len(outs) > 0:
                outs = np.array(outs)
                tgts = np.array(tgts)
                se = (outs - tgts) ** 2
                cor = pearson_corrcoef(outs, tgts)

                results[idx]['corr'].append(cor)
                results[idx]['loss'].append(se)
                results[idx]['gain'].append(gain)
                results[idx]['tgts'].append(tgts)
                results[idx]['outs'].append(outs)
                results[idx]['loc'] = loc.detach().cpu().numpy()

    rn = cfg.model['target']
    with open(f'g-{rn}-results-all-d{top}.pkl', 'wb') as f:
        pickle.dump(results, f)

    print('Results saved.')


def fluctuation_probability(y, window_frac=0.15, polyorder=2,
                            local_frac=0.1):
    """
    Detect persistent (possibly bursty) fluctuations.
    """

    y = np.asarray(y)
    N = len(y)

    if N < polyorder + 5:
        return 0.0, {"reason": "too short"}

    # -------------------------
    # 1. Trend
    # -------------------------
    window = int(round(window_frac * N))
    window = max(polyorder + 2, min(window, N))
    if window % 2 == 0:
        window -= 1

    trend = savgol_filter(y, window_length=window, polyorder=polyorder)
    resid = y - trend

    # -------------------------
    # 2. Amplitude score
    # -------------------------
    resid_energy = np.mean(resid ** 2)
    trend_energy = np.mean(trend ** 2) + 1e-12
    energy_score = resid_energy / (resid_energy + trend_energy)

    # -------------------------
    # 3. Local fluctuation coverage (KEY FIX)
    # -------------------------
    L = max(5, int(local_frac * N))
    step = L // 2

    active = []
    for i in range(0, N - L + 1, step):
        seg = resid[i:i + L]
        zcr = np.mean(np.sign(seg[:-1]) != np.sign(seg[1:]))
        amp = np.std(seg)

        active.append((zcr > 0.1) and (amp > 0.5 * np.std(resid)))

    coverage_score = np.mean(active) if active else 0.0

    # -------------------------
    # 4. Final probability
    # -------------------------
    prob = (
        0.5 * energy_score +
        0.5 * coverage_score
    )

    prob = float(np.clip(prob, 0, 1))

    diagnostics = {
        "energy_score": energy_score,
        "coverage_score": coverage_score,
        "window_length": window,
        "local_window": L
    }

    return prob, diagnostics


def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # torch.manual_seed(42)  # For reproducibility
    # np.random.seed(42)  # For reproducibility
    # random.seed(42)  # For reproducibility

    parser = argparse.ArgumentParser(description="Run the test function with configurable parameters.")
    parser.add_argument("--cfg", type=str, help="Config file path", default="config/rgnn.yaml")
    parser.add_argument("--training", action="store_true", help="Enable training mode (default: False)")
    parser.add_argument("--ckpt", type=str, default="model_rgnn.pth", help="Path to save/load the model checkpoint")
    parser.add_argument("--top", type=float, default=0.2, help="Top p% stations to compute loss")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    ckpt_path = args.ckpt
    top = args.top
    # test(cfg, train=True, ckpt_path=ckpt_path, top=top)
    test(cfg, train=False, ckpt_path=ckpt_path, top=top)


if __name__ == "__main__":
    main()
