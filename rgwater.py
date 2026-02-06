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

from tqdm import tqdm

from model.gnn import GATWithEdgeAttrRain
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
    model = GATWithEdgeAttrRain().to(device)
    return model


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
    output_path = f'results/{idx}_rgnn.png'
    plt.savefig(output_path)
    plt.close()

    # print(f"Plot saved to {output_path}")
    # input()


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


class QuarticLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        # Compute element-wise error
        error = pred - target
        # Raise to the 4th power
        loss = error ** 4
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        

class MixLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        # Compute element-wise error
        error = pred - target
        # Raise to the 4th power
        loss = (error ** 2 + error ** 4) / 2
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def spike_aware_loss(pred, target, base_loss_fn=torch.nn.MSELoss(), eps=1e-6):
    # Compute absolute difference between consecutive time steps
    target_diff = torch.abs(target[:, 1:] - target[:, :-1])  # (B, T-1)
    spike_weight = torch.nn.functional.pad(target_diff, (1, 0)) + eps  # Pad to align shape (B, T)
    
    # Normalize weights
    spike_weight = spike_weight / spike_weight.mean(dim=1, keepdim=True)

    # Base loss
    base_loss = base_loss_fn(pred, target)

    # Weight the loss
    weighted_loss = (base_loss * spike_weight).mean()
    return weighted_loss


def loss_fn(loss_f, o, y, pearson=True, std=True):
    if o.shape[0] == 0:
        return torch.tensor(0.0, device=o.device)

    loss = loss_f(o, y)
    if pearson:
        loss -= pearson_corrcoef(o.squeeze(-1), y.squeeze(-1))[0]
    
    if std:
        loss += std_loss(o, y)

    return loss


def top_p_mask_global(y, p=0.2):
    flat = y.reshape(-1)
    k = max(1, int(math.ceil(flat.numel() * p)))
    _, idx = torch.topk(flat, k)
    m = torch.zeros_like(flat, dtype=torch.bool)
    m[idx] = True
    return m.view_as(y)


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

    # Define your training and testing dataset
    train_dataset = GWaterDataset(path='data/selected_stats_rainfall_segment.pkl', train=True,
        selected_stations=good_nb_extended, input_type=cfg.dataset.inputs
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)  # Adjust batch_size as needed
    
    test_dataset = GWaterDataset(
        path='data/selected_stats_rainfall_segment.pkl', train=False,
        selected_stations=test_nb, input_type=cfg.dataset.inputs
    )

    # Initialize model and wrap it with DataParallel
    model = create_model(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch_geometric.nn.DataParallel(model, device_ids=[0, 1])
        model.to(device)

    # Define loss function and optimizer
    l2_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if train:
        num_epochs = 20
        list_loss = []
        epoch_bar = tqdm(range(num_epochs), desc="Epochs")  # tqdm for epochs

        for epoch in epoch_bar:
            model.train()

            epoch_loss = 0.0
            num_batches = 0
            for y, x, valid, graph_data, fx, r, loc, earray in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
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
                o, rf, rff, valid = model(nodes, edge_index, edge_attr, valid, r, fx, loc, earray)
                
                # Compute loss
                pearson_flag = False
                std_flag = False

                mask = top_p_mask_global(y, p=top) if top < 1.0 else 1.0
                loss = loss_fn(l2_loss, o[:1][mask], y[mask], pearson=pearson_flag, std=std_flag)
                loss += loss_fn(l2_loss, rf[:1][mask], y[mask], pearson=pearson_flag, std=std_flag)
                loss += loss_fn(l2_loss, rf[:1], y, pearson=pearson_flag, std=std_flag)
                loss += loss_fn(
                    l2_loss,
                    rff[1:],
                    nodes[1:rf.shape[0]],
                    pearson=pearson_flag,
                    std=std_flag
                )
                loss += loss_fn(
                    l2_loss,
                    rf[1:],
                    nodes[1:rf.shape[0]].squeeze(-1),
                    pearson=pearson_flag,
                    std=std_flag
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                # Print loss for each epoch
                epoch_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Loss: {(epoch_loss / num_batches):.4f}")
                list_loss.append(loss.item())

        # Save model periodically
        torch.save(model.state_dict(), ckpt_path)

    # Load trained model for evaluation
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    model.to(device)

    if train:
        return

    test_dataset = GWaterDataset(
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
    with torch.no_grad():
        for idx, (my, mx, mvalid, graph_data, mfx, mr, loc, earray) in enumerate(test_loader):
            mnodes = graph_data.x.unsqueeze(-1)
            edge_index = graph_data.edge_index.to(device)
            edge_attr = graph_data.edge_attr.to(device)
            mfx = mfx.to(device)
            mr = mr.to(device)
            loc = loc.to(device)
            earray = earray.to(device)
            
            outs = []
            tgts = []
            cors = []
            gain = []
            for i in range(mx.shape[2] // seg_len):
                start = time.time()
                x = mx[:, :, i*seg_len:(i+1)*seg_len].to(device)
                valid = mvalid[:, :, i*seg_len:(i+1)*seg_len].to(device)
                y = my[:, i*seg_len:(i+1)*seg_len].to(device)
                nodes = mnodes[:, i*seg_len:(i+1)*seg_len]
                r = mr[:, :, i*seg_len:(i+1)*seg_len].to(device)
                fx = mfx

                nodes = nodes.to(device)
                y = y.to(device)
                valid = valid.to(device)

                if not has_significant_slope(y[0].detach().cpu().numpy()):
                    continue

                # Forward pass
                o, rf, rff, _ = model(nodes, edge_index, edge_attr, valid, r, fx, loc, earray)
                o = o[:1]

                elapsed = time.time() - start
                total_elapsed += elapsed
                total_n += 1
                
                # from calflops import calculate_flops
                # inputs = {}
                # inputs["nodes"] = nodes
                # inputs["edge_index"] = edge_index
                # inputs["edge_attr"] = edge_attr
                # inputs["valid"] = valid
                # flops, macs, params = calculate_flops(
                #     model=model, 
                #     kwargs=inputs,
                #     output_as_string=True,
                #     output_precision=4
                # )
                # print("FLOPs:%s  MACs:%s  Params:%s \n" %(flops, macs, params))
                # # FLOPs:93.456 KFLOPS  MACs:46.512 KMACs  Params:1.158 K
                # # FLOPs:3.552 KFLOPS  MACs:1.728 KMACs  Params:642
                # exit()
                
                o_np = o.flatten().detach().cpu().numpy()
                y_np = y.flatten().detach().cpu().numpy()

                outs.extend(
                    o_np
                )
                tgts.extend(
                    y_np
                )
                cors.append(
                    pearson_corrcoef(
                        o_np,
                        y_np
                    )
                )
                gain.extend(
                    (y_np - o_np).tolist()
                )

                if torch.abs(y).max() > 0.5 and torch.abs(y).max() < 1.0:
                    plot_graph(o, y, x, f'{loc.detach().cpu().numpy()}-{idx}-{i}')

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
                results[idx]['loc'] = loc.detach().cpu().numpy()

        for k in results.keys():
            print(f"Key: {k} - {results[k]['loc'] if 'loc' in results[k] else 'No Location'}, Output: {results[k]['outs'][0].shape if len(results[k]['outs']) > 0 else 0}")
    
    print(f'Total Elapsed: {total_elapsed:.6f} seconds, Average time per segment: {total_elapsed / total_n:.6f} seconds')

    rn = cfg.model['target']
    with open(f'g-{rn}-results-all-d{top}.pkl', 'wb') as f:
        pickle.dump(results, f)
    print('Results saved.')


def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    torch.manual_seed(42)  # For reproducibility
    np.random.seed(42)  # For reproducibility
    random.seed(42)  # For reproducibility

    parser = argparse.ArgumentParser(description="Run the test function with configurable parameters.")
    parser.add_argument("--cfg", type=str, help="Config file path", default="config/rgnn.yaml")
    parser.add_argument("--training", action="store_true", help="Enable training mode (default: False)")
    parser.add_argument("--ckpt", type=str, default="model_rgnn.pth", help="Path to save/load the model checkpoint")
    parser.add_argument("--top", type=float, default=0.2, help="Top p% stations to compute loss")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    ckpt_path = args.ckpt
    top = args.top
    test(cfg, train=True, ckpt_path=ckpt_path, top=top)
    test(cfg, train=False, ckpt_path=ckpt_path, top=top)


if __name__ == "__main__":
    main()
