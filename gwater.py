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

from model.gnn import GATWithEdgeAttr
from model.mlp import *

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
    model = GATWithEdgeAttr(2, 18, 1, 1).to(device)
    return model


def plot_graph(o, y, x):
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
    output_path = './interpolation_vs_groundtruth.png'
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved to {output_path}")
    input()


def pearson_corrcoef(x, y, dim=-1, eps=1e-8):
    """
    Compute Pearson correlation between x and y along given dimension.
    Assumes x and y are of the same shape.
    """
    x_centered = x - x.mean(dim=dim, keepdim=True)
    y_centered = y - y.mean(dim=dim, keepdim=True)

    cov = (x_centered * y_centered).mean(dim=dim)
    std_x = x_centered.std(dim=dim)
    std_y = y_centered.std(dim=dim)
    
    corr = cov / (std_x * std_y + eps)
    return corr


def test(cfg, out, n_points, model_size, inputs, train=True, testing_train=False):
    with open('data/split.pkl', 'rb') as f:
        split = pickle.load(f)
        good_nb_extended = split['train']
        test_nb = split['test']
    print('Train length: ', len(good_nb_extended))
    print('Test length: ', len(test_nb))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define your training and testing dataset
    train_dataset = GWaterDataset(path='data/selected_stats.pkl', train=True,
        selected_stations=good_nb_extended, input_type=cfg.dataset.inputs
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)  # Adjust batch_size as needed

    # Initialize model and wrap it with DataParallel
    model = create_model(device)
    model.n_points = n_points
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch_geometric.nn.DataParallel(model, device_ids=[0, 1])
        model.to(device)

    # Define loss function and optimizer
    l1_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if train:
        num_epochs = 5
        list_loss = []
        epoch_bar = tqdm(range(num_epochs), desc="Epochs")  # tqdm for epochs

        for epoch in epoch_bar:
            model.train()
            for y, x, valid, graph_data in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
                valid = valid.float().to(device)
                y = y.to(device).float()
                x = x.to(device).float()
                nodes = graph_data.x.unsqueeze(-1).to(device)
                edge_index = graph_data.edge_index.to(device)
                edge_attr = graph_data.edge_attr.to(device)

                # Forward pass
                o = model(nodes, edge_index, edge_attr, valid)

                # Compute loss
                loss = l1_loss(o, y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print loss for each epoch
                epoch_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
                list_loss.append(loss.item())

        # Save model periodically
        torch.save(model.state_dict(), 'model_gnn.pth')

    # Load trained model for evaluation
    model.load_state_dict(torch.load('model_gnn.pth'))
    model.eval()
    model.to(device)

    if train:
        return

    test_dataset = GWaterDataset(
        path='data/selected_stats.pkl', train=False,
        selected_stations=test_nb, input_type=cfg.dataset.inputs
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    results = {
        k:{
            'corr': [],
            'loss': [],
            'best': [],
            'mean': [],
        } for k in range(len(list(test_dataset.data.keys())))
    }

    with torch.no_grad():
        for idx, (my, mx, mvalid, graph_data) in enumerate(test_loader):
            mnodes = graph_data.x.unsqueeze(-1)
            edge_index = graph_data.edge_index.to(device)
            edge_attr = graph_data.edge_attr.to(device)
            start = time.time()
            for i in range(mx.shape[2] // 1024):
                x = mx[:, :, i*1024:(i+1)*1024].to(device)
                valid = mvalid[:, :, i*1024:(i+1)*1024].to(device)
                y = my[:, i*1024:(i+1)*1024].to(device)
                nodes = mnodes[:, i*1024:(i+1)*1024]

                nodes = nodes.to(device)
                y = y.to(device)
                valid = valid.to(device)

                # Forward pass
                o = model(nodes, edge_index, edge_attr, valid)
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
                
                l1_elements = (o - y) ** 2
                loss = torch.sqrt(torch.mean(l1_elements))
                corr = pearson_corrcoef(
                    o.flatten(),
                    y.flatten()
                )
                mean_corr = corr.mean()
                cvalid = torch.mean(valid.float(), dim=-1)
                if torch.sum(cvalid > 0) == 0:
                    continue

                min_d = torch.min(torch.mean(torch.abs(y.unsqueeze(1) - x), dim=-1), dim=-1)[0]
                results[idx]['corr'].append(mean_corr.item())
                results[idx]['loss'].append(loss.item())
                results[idx]['best'].append(min_d.item())
                results[idx]['mean'].append(torch.mean(torch.abs(y)).item())

                if x.shape[1] == 5:
                    plot_graph(o, y, x)

            print(f'Elapsed: {time.time() - start}')

    rn = cfg.model['target']
    with open(f'g-{rn}-{cfg.model.params.fmts}-results.pkl', 'wb') as f:
        pickle.dump(results, f)


def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    torch.manual_seed(42)  # For reproducibility
    np.random.seed(42)  # For reproducibility
    random.seed(42)  # For reproducibility

    parser = argparse.ArgumentParser(description="Run the test function with configurable parameters.")
    parser.add_argument("--cfg", type=str, help="Config file path", default="config/gru_32_1_io.yaml")
    parser.add_argument("--training", action="store_true", help="Enable training mode (default: False)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)

    model_size = 'gnn'
    inputs = 'pte'
    with open("log-test-all.txt", "w") as f:
        station_name = f'S32-{8}'
        # test(cfg, station_name, 16, model_size=model_size, inputs=inputs, train=True)
        test(cfg, station_name, 10, model_size=model_size, inputs=inputs, train=False)


def save_results(log, out, l1, list_l1, list_values):
    log.write(f'{out} {l1.item()}\n')
    print(f'{out} {l1.item()}')

    # Example usage
    # Assuming `times` and `list_values` are already populated from your code
    plot_predictions_with_time(list_l1, save_path=f'{out}-l1.png')
    plot_predictions_with_time(list_values, save_path=f'{out}-series.png')


main()
