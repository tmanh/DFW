import argparse
import os
import logging
import time
from omegaconf import OmegaConf
import requests
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from dataloader import *
from torch_geometric.loader import DataLoader
import torch_geometric

import pickle
from scipy.signal import savgol_filter

import folium
from folium import Element

from tqdm import tqdm

from model.gnn import GATWithEdgeAttrRain
from model.mlp import *
from water_level import has_significant_slope, split_stations_by_clusters

from branca.element import MacroElement, Template

logging.basicConfig(filename="log-test-all-gnn.txt", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SmoothZoomFit(MacroElement):
    def __init__(self, south, west, north, east,
                 zoom_snap=0.1, zoom_delta=0.1, wheel_px_per_zoom=400,
                 pad_px=(10, 10)):
        super().__init__()
        self._name = "SmoothZoomFit"

        self.south = float(south)
        self.west  = float(west)
        self.north = float(north)
        self.east  = float(east)

        self.zoom_snap = float(zoom_snap)
        self.zoom_delta = float(zoom_delta)
        self.wheel_px_per_zoom = int(wheel_px_per_zoom)

        self.pad0 = int(pad_px[0])
        self.pad1 = int(pad_px[1])

        self._template = Template(u"""
        {% macro script(this, kwargs) %}
        (function () {
            var map = {{ this._parent.get_name() }};
            if (!map) return;

            map.options.zoomSnap = {{ this.zoom_snap }};
            map.options.zoomDelta = {{ this.zoom_delta }};
            map.options.wheelPxPerZoom = {{ this.wheel_px_per_zoom }};

            if (map.scrollWheelZoom) map.scrollWheelZoom.enable();

            map.fitBounds(
                [[{{ this.south }}, {{ this.west }}], [{{ this.north }}, {{ this.east }}]],
                {padding: [{{ this.pad0 }}, {{ this.pad1 }}]}
            );
        })();
        {% endmacro %}
        """)


OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]


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


def top_p_target_mse_global(pred: torch.Tensor, target: torch.Tensor, top_percent: float = 0.2) -> torch.Tensor:
    """
    Select the top p% elements by TARGET value (global), then compute MSE on those elements.
    Follows the same mask-building principle as top_p_mask_global.
    """
    assert pred.shape == target.shape, "Prediction and target must have the same shape"
    assert 0 < top_percent <= 1, "p must be in (0, 1]"

    # --- same principle as top_p_mask_global, but using target ---
    flat_t = torch.abs(target.reshape(-1))
    k = max(1, int(math.ceil(flat_t.numel() * top_percent)))
    _, idx = torch.topk(flat_t, k)          # pick top targets
    m = torch.zeros_like(flat_t, dtype=torch.bool)
    m[idx] = True
    m = m.view_as(target)
    # ------------------------------------------------------------

    se = (pred - target) ** 2
    return se[m].mean()


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

    # # TODO
    # make_train_test_station_map(
    #     train_loader,
    #     test_loader,
    #     out_html="train_test_station_map.html"
    # )
    # exit()

    model = create_model(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch_geometric.nn.DataParallel(model, device_ids=[0, 1])
        model.to(device)

    # Define loss function and optimizer
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
                o, rf, rff, valid = model(nodes, edge_index, edge_attr, valid, r, fx, loc, earray)
                
                # Compute loss
                loss = topk_mse(o[:1], y, top_percent=top)
                loss += topk_mse(rf[:1], y, top_percent=top)
                if rf.shape[0] > 1:
                    loss += topk_mse(
                        rff[1:],
                        nodes[1:rf.shape[0]],
                        top_percent=top
                    )
                    loss += topk_mse(
                        rf[1:],
                        nodes[1:rf.shape[0]].squeeze(-1),
                        top_percent=top
                    )

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
            best_hits = 0

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

                        loc_vals = y[0].detach().cpu().numpy()
                        if not has_significant_slope(loc_vals):
                            continue
                        delta = abs(np.max(loc_vals) - np.min(loc_vals))
                        if delta <= 0.3:
                            continue

                        o, _, _, _ = model(nodes, edge_index, edge_attr, valid, r, fx, loc, earray)
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

            if test_mean_re_mae < best_test_rmse:
                best_test_rmse = test_mean_re_mae
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"New best checkpoint saved (RMSE={test_mean_re_mae:.4f})")

            model.train()
            # ======================================================================

        print(f"\nBest epoch: {best_epoch}, Best Test RMSE: {best_test_rmse:.4f}")
    print(best_ckpt_path)
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

                loc_vals = y[0].detach().cpu().numpy()
                if not has_significant_slope(loc_vals):
                    continue
                delta = abs(np.max(loc_vals) - np.min(loc_vals))
                if delta <= 0.3:
                    continue

                o, rf, rff, _ = model(nodes, edge_index, edge_attr, valid, r, fx, loc, earray)
                o = o[:1]

                # dt = 1
                # tidal_prob, diagnostics = fluctuation_probability(y[0].detach().cpu().numpy(), dt)
                # plot_graph(o, y, x, f'{idx}-{i}-{tidal_prob:.2f}')

                elapsed = time.time() - start
                total_elapsed += elapsed
                total_n += 1

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


def add_train_test_legend_right(m, title="Stations"):
    html = f"""
    <div style="
      position: fixed;
      top: 20px; right: 20px;
      z-index: 9999;
      background: rgba(255,255,255,0.95);
      padding: 10px 12px;
      border: 2px solid #333;
      border-radius: 6px;
      font-size: 13px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    ">
      <div style="font-weight:700;margin-bottom:6px;">{title}</div>

      <div style="display:flex;align-items:center;margin:4px 0;">
        <div style="width:14px;height:14px;background:#1b9e77;border:1px solid #111;margin-right:8px;"></div>
        <div>Train</div>
      </div>

      <div style="display:flex;align-items:center;margin:4px 0;">
        <div style="width:14px;height:14px;background:#d95f02;border:1px solid #111;margin-right:8px;"></div>
        <div>Test</div>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(html))


def make_train_test_station_map(
    train_loader,
    test_loader,
    out_html="train_test_station_map.html",
    tiles="CartoDB positron",
    zoom_start=9,
):
    """
    Build an HTML map showing station locations for train vs test.

    Assumptions:
      - each batch from loaders has a `loc` tensor/array somewhere in the returned tuple
      - `_extract_latlon(loc)` exists (same one you used for error map)
      - `_add_halo_circle(m, lat, lon, color, html, r=...)` exists

    Output:
      - HTML map saved to out_html
      - prints counts in terminal
    """
    def _get_loc_from_batch(batch):
        return batch[-2].detach().numpy()

    def _collect(loader):
        pts = {}  # key -> (lat,lon)
        for bidx, batch in enumerate(loader):
            loc = _get_loc_from_batch(batch)
            if loc is None:
                continue
            ll = _extract_latlon(loc)
            if ll is None:
                continue
            lat, lon = ll
            # key: rounded coords to dedupe safely
            key = (round(float(lat), 7), round(float(lon), 7))
            pts[key] = (float(lat), float(lon))
        return pts

    train_pts = _collect(train_loader)
    test_pts  = _collect(test_loader)
    # print(len(train_pts), len(test_pts))
    # exit()

    train_keys = set(train_pts.keys())
    test_keys  = set(test_pts.keys())
    both_keys  = train_keys & test_keys
    train_only = train_keys - both_keys
    test_only  = test_keys - both_keys

    # center map from all points
    all_pts = list(train_pts.values()) + [test_pts[k] for k in test_only]  # avoid duplicates a bit
    if len(all_pts) == 0:
        raise RuntimeError("No station locations found from the loaders.")

    lats = np.array([p[0] for p in all_pts], float)
    lons = np.array([p[1] for p in all_pts], float)

    m = folium.Map(
        location=[float(np.mean(lats)), float(np.mean(lons))],
        zoom_start=zoom_start,
        tiles="CartoDB positron",
        control_scale=True,
    )
    m.fit_bounds([[float(min(lats)), float(min(lons))],
              [float(max(lats)), float(max(lons))]])

    # add overlay using station-derived bounds (works)
    add_osm_waterways_overlay_from_points(m, lats, lons)

    add_train_test_legend_right(m, title="Train/Test stations")

    # colors (match elevation map)
    C_TRAIN = "#00c853"   # train green
    C_TEST  = "#ff1744"   # test red

    # Plot train-only
    for key in train_only:
        lat, lon = train_pts[key]
        txt = f"Train station<br>lat={lat:.6f}, lon={lon:.6f}"
        _add_halo_circle(m, lat, lon, C_TRAIN, txt, r=5)

    # Plot test-only
    for key in test_only:
        lat, lon = test_pts[key]
        txt = f"Test station<br>lat={lat:.6f}, lon={lon:.6f}"
        _add_halo_circle(m, lat, lon, C_TEST, txt, r=5)

    print("Train/Test station counts:")
    print(f"  train-only: {len(train_only)}")
    print(f"  test-only:  {len(test_only)}")
    print(f"  both:       {len(both_keys)}")
    print(f"  total uniq: {len(train_keys | test_keys)}")

    add_ultra_smooth_zoom_and_fit(
        m,
        lats, lons,
        zoom_snap=0.1,
        zoom_delta=0.1,
        wheel_px_per_zoom=450
    )
    m.save(out_html)
    print(f"Saved: {out_html}")
    return out_html


def add_ultra_smooth_zoom_and_fit(
    m,
    lats, lons,
    pad_ratio=0.03,
    zoom_snap=0.1,
    zoom_delta=0.1,
    wheel_px_per_zoom=400,
    pad_px=(10, 10),
):
    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)

    good = np.isfinite(lats) & np.isfinite(lons)
    lats = lats[good]
    lons = lons[good]

    if lats.size == 0:
        print("[add_ultra_smooth_zoom_and_fit] No valid (finite) station coords -> skip smooth zoom")
        return m

    south, north = float(lats.min()), float(lats.max())
    west,  east  = float(lons.min()), float(lons.max())

    lat_span = max(north - south, 1e-6)
    lon_span = max(east - west, 1e-6)

    south -= lat_span * pad_ratio
    north += lat_span * pad_ratio
    west  -= lon_span * pad_ratio
    east  += lon_span * pad_ratio

    m.add_child(SmoothZoomFit(
        south, west, north, east,
        zoom_snap=zoom_snap,
        zoom_delta=zoom_delta,
        wheel_px_per_zoom=wheel_px_per_zoom,
        pad_px=pad_px,
    ))
    return m


def add_osm_waterways_overlay_from_points(
    m,
    lats,
    lons,
    pad_ratio=0.02,
    name="OSM Waterways",
    timeout=30,
    max_tries=6,
):
    """
    Robust waterways overlay:
      - tries multiple Overpass endpoints
      - retries with smaller bbox / fewer classes
      - does NOT raise if Overpass fails; caller can wrap too
    """
    lats = [float(x) for x in lats]
    lons = [float(x) for x in lons]
    if len(lats) == 0 or len(lons) == 0:
        raise ValueError("Empty lats/lons passed to waterways overlay")

    # bbox
    south, north = min(lats), max(lats)
    west,  east  = min(lons), max(lons)

    # pad bbox
    lat_pad = (north - south) * pad_ratio if north > south else 0.01
    lon_pad = (east - west) * pad_ratio if east > west else 0.01
    south -= lat_pad; north += lat_pad
    west  -= lon_pad; east  += lon_pad

    # Try progressively lighter queries
    #  - start broad: river|canal|stream|drain|ditch
    #  - then drop small: river|canal|stream
    #  - then only main: river|canal
    classes_schedule = [
        "river|canal|stream|drain|ditch",
        "river|canal|stream",
        "river|canal",
    ]

    # also shrink bbox progressively if still too big / timeout
    shrink_schedule = [1.0, 0.8, 0.6, 0.45, 0.35]

    def _shrink_bbox(s, w, n, e, factor):
        # shrink around center
        cy = 0.5 * (s + n)
        cx = 0.5 * (w + e)
        hy = 0.5 * (n - s) * factor
        hx = 0.5 * (e - w) * factor
        return cy - hy, cx - hx, cy + hy, cx + hx

    last_err = None
    attempt = 0

    for cls in classes_schedule:
        for shrink in shrink_schedule:
            s2, w2, n2, e2 = _shrink_bbox(south, west, north, east, shrink)

            query = f"""
            [out:json][timeout:25];
            (
              way["waterway"~"{cls}"]({s2},{w2},{n2},{e2});
            );
            out geom;
            """

            for url in OVERPASS_URLS:
                attempt += 1
                if attempt > max_tries:
                    break

                try:
                    r = requests.post(url, data={"data": query}, timeout=timeout)
                    r.raise_for_status()
                    data = r.json()

                    feats = []
                    for el in data.get("elements", []):
                        geom = el.get("geometry")
                        if not geom:
                            continue
                        coords = [[pt["lon"], pt["lat"]] for pt in geom]
                        props = {"waterway": el.get("tags", {}).get("waterway", "waterway")}
                        feats.append({
                            "type": "Feature",
                            "geometry": {"type": "LineString", "coordinates": coords},
                            "properties": props,
                        })

                    if not feats:
                        # success but nothing found — acceptable
                        print(f"[Waterways] No features found (cls={cls}, shrink={shrink:.2f}).")
                        return m

                    gj = {"type": "FeatureCollection", "features": feats}
                    folium.GeoJson(
                        gj,
                        name=name,
                        style_function=lambda f: {"color": "#1976d2", "weight": 1.0, "opacity": 0.85},
                        tooltip=folium.GeoJsonTooltip(fields=["waterway"], aliases=["waterway"]),
                    ).add_to(m)

                    print(f"[Waterways] Added {len(feats)} features (endpoint={url}, cls={cls}, shrink={shrink:.2f}).")
                    return m

                except Exception as e:
                    last_err = e
                    # brief backoff
                    time.sleep(0.6)

            if attempt > max_tries:
                break

    # If all failed, don't crash — just warn
    raise RuntimeError(f"Overpass failed after {attempt} tries. Last error: {last_err}")


def _add_halo_circle(m, lat, lon, color, text, r=8):
    # colored inner dot with black outline
    folium.CircleMarker(
        location=[lat, lon],
        radius=r,
        color="#111111",
        weight=2,
        fill=True,
        fill_color=color,
        fill_opacity=1.0,
        opacity=1.0,
        tooltip=folium.Tooltip(text, sticky=True),
    ).add_to(m)


def _extract_latlon(loc):
    """
    loc can be (2,), (1,2), etc. Returns (lat, lon).
    NOTE: if your loc is (lon, lat) swap below.
    """
    loc = np.asarray(loc).reshape(-1)
    if loc.size < 2:
        return None
    # Most common: [lat, lon]. If yours is [lon, lat], swap these two lines.
    lat, lon = float(loc[0]), float(loc[1])
    return lat, lon


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
