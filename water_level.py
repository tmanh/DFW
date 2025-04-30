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
from model.attention import MLPAttention
from model.distance import InverseDistance
from model.mlp import *
from sklearn.cluster import KMeans

from dataloader import *
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from tqdm import tqdm

logging.basicConfig(filename="log-test-all.txt", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

random.seed(42)  # Replace 42 with any integer seed you want


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
# contribute, method, results

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
    else:
        return None
    return f'model_{mn}_wo.pth'


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
        neighbors = data[p]['neighbor'].keys()
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


def test(cfg, train=True):
    with open('data/selected_stats.pkl', 'rb') as f:
        data = pickle.load(f)

    good_nb_extended, test_nb = split_stations_by_clusters(data)
    # if not os.path.exists('test_nb.pkl'):
    #     with open('test_nb.pkl', 'wb') as f:
    #         pickle.dump(test_nb, f)
    # else:
    #     with open('test_nb.pkl', 'rb') as f:
    #         test_nb = pickle.load(f)
    print('Train length: ', len(good_nb_extended))
    print('Test length: ', len(test_nb))
    with open('split.pkl', 'wb') as f:
        pickle.dump(
            {'train': good_nb_extended, 'test': test_nb},
            f
        )
    # exit()

    # 🔹 1️⃣ Define Device (Multi-GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = WaterDataset(
        path='data/selected_stats.pkl', train=True,
        selected_stations=good_nb_extended, input_type=cfg.dataset.inputs
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # 🔹 2️⃣ Initialize Model
    model = instantiate_from_config(cfg.model).to(device)
    model = model.to(device)


    mse_loss = nn.MSELoss()

    print('Config:', cfg.model.params.fmts)

    # Training loop
    not_finish_training = True
    stage = 1
    num_epochs = 5
    freeze_base = False

    n_loops = 1
    # if 'w' in cfg.model.params.fmts:
    #     n_loops += 1

    while train and not_finish_training:
        if freeze_base:
            if torch.cuda.device_count() > 1:
                model.module.freeze()
            else:
                model.freeze()

        list_loss = []
        epoch_bar = tqdm(range(num_epochs), desc="Epochs")  # Initialize tqdm for epochs
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in epoch_bar:
            for x, xs, y, kes, valid in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
                x = x.to(device)
                xs = xs.to(device)
                y = y.to(device)
                kes = kes.to(device)
                valid = valid.to(device)

                # Forward pass
                o = model(xs, x, valid, inputs=cfg.dataset.inputs, train=True, stage=stage)

                # Compute L1 loss on all elements
                loss = mse_loss(o, y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Print loss for every epoch
                epoch_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
                list_loss.append(loss.item())

            # Save model checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), f"model_{epoch}.pth")

        n_loops -= 1

        if n_loops <= 0:
            not_finish_training = False
        
        if 'w' in cfg.model.params.fmts:
            stage = 2
            freeze_base = True

    ckpt_name = get_checkpoint_name(cfg=cfg)
    if ckpt_name is not None:
        if not os.path.exists(ckpt_name):
            if os.path.exists(f'model_{num_epochs - 1}.pth'):
                os.rename(f'model_{num_epochs - 1}.pth', ckpt_name)
            else:
                print('No trained model!')
                exit()

        # model.load_state_dict(torch.load(ckpt_name), strict=False)
        model.eval()

    test_dataset = WaterDataset(
        path='data/selected_stats.pkl', train=False,
        selected_stations=test_nb, input_type=cfg.dataset.inputs
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    results = {
        k:{'loss-p': 0, 'loss': 0, 'count': 0, 'name': list(test_dataset.data.keys())[k], 'range': 0} for k in range(len(list(test_dataset.data.keys())))
    }

    with torch.no_grad():
        for idx, (mx, mxs, my, _, mvalid) in enumerate(test_loader):
            start = time.time()
            for i in range(mx.shape[2] // 1024):
                x = mx[:, :, i*1024:(i+1)*1024].to(device)
                valid = mvalid[:, :, i*1024:(i+1)*1024].to(device)
                xs = mxs.to(device)
                y = my[:, i*1024:(i+1)*1024].to(device)

                if torch.abs(torch.mean(y)) < 0.3:
                    continue

                o = model(xs, x, valid, inputs=cfg.dataset.inputs, train=False, stage=-1)
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

                l1_elements = torch.abs(o - y.unsqueeze(1))
                ploss = torch.abs(o - y.unsqueeze(1)) / (y.unsqueeze(1) + 1e-8)
                ploss = torch.mean(ploss)
                loss = torch.mean(l1_elements)

                results[idx]['loss-p'] += ploss.item()
                results[idx]['loss'] += loss.item()
                results[idx]['count'] += 1 
                # plot_graph(o, y, x)
            print(f'Elapsed: {time.time() - start}')
    
    rn = cfg.model['target']
    with open(f'{rn}-{cfg.model.params.fmts}-results.pkl', 'wb') as f:
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
        
    print('-----Traning-----')
    # test(2, cfg, train=False, testing_train=True)
    # test(4, cfg, train=False, testing_train=True)
    # test(6, cfg, train=False, testing_train=True)
    # test(8, cfg, train=False, testing_train=True)
    # test(10, cfg, train=False, testing_train=True)
    
    print('-----Testing-----')
    # test(2, cfg, train=False)
    # test(4, cfg, train=False)
    # test(6, cfg, train=False)
    # test(8, cfg, train=False)
    test(cfg, train=False)


def save_results(out, l1, list_l1, list_values):
    logging.info(f'{out} {l1.item()}\n')
    print(f'{out} {l1.item()}')

    # Example usage
    # Assuming `times` and `list_values` are already populated from your code
    plot_predictions_with_time(list_l1, save_path=f'{out}-l1.png')
    plot_predictions_with_time(list_values, save_path=f'{out}-series.png')


if __name__ == "__main__":
    main()
