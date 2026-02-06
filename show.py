import argparse

from dataloader import *
from water_level import has_significant_slope, split_stations_by_clusters

from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

from tqdm import tqdm


def plot_graph(y, x, idx):
    # Move tensors to CPU and convert to NumPy
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
    plt.title('Comparison of Ground Truth and Interpolated Time Series')
    plt.xlabel('Time Step')
    plt.ylabel('Water Level')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save to file (you can change path/filename as needed)
    tmp_dir = './xxxxx'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    output_path = f'{tmp_dir}/{idx}_rgnn.png'
    plt.savefig(output_path)
    plt.close()


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

with open('data/selected_stats_rainfall_segment.pkl', 'rb') as f:
    data = pickle.load(f)
good_nb_extended, test_nb = split_stations_by_clusters(data)

device = 'cpu'

# Define your training and testing dataset
train_dataset = GWaterDataset(path='data/selected_stats_rainfall_segment.pkl', train=True,
    selected_stations=good_nb_extended, input_type=cfg.dataset.inputs
)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)  # Adjust batch_size as needed
    
test_dataset = GWaterDataset(
    path='data/selected_stats_rainfall_segment.pkl', train=False,
    selected_stations=test_nb, input_type=cfg.dataset.inputs
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

for idx, (y, x, valid, graph_data, fx, r, loc, earray) in enumerate(train_loader):
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

    plot_graph(y, x, f'{loc.detach().cpu().numpy()}-{idx}')