#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pickle
import argparse
import random
import time
import logging

import torch
import numpy as np
import matplotlib.pyplot as plt

import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_origin

from pyproj import Transformer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# IMPORTANT: use your same imports
from dataloader import *   # WaterDatasetY
# IMPORTANT: split_stations_by_clusters is already in your codebase (the same one your test() uses)
# If it is in the same file, do nothing. If not, import it from where it lives:
# from your_module import split_stations_by_clusters


logging.basicConfig(filename="log-create-dtm.txt",
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


# -------------------------
# DTM mosaic builder (one tif at a time)
# -------------------------
def build_lowres_dtm_mosaic(tif_paths, target_res_m=250.0, dst_crs=None, nodata_out=np.nan):
    assert len(tif_paths) > 0, "No DTM tif files found."

    with rasterio.open(tif_paths[0]) as src0:
        if dst_crs is None:
            dst_crs = src0.crs
        else:
            dst_crs = rasterio.crs.CRS.from_user_input(dst_crs)

    # If the CRS is geographic (degrees), convert meters -> degrees (approx at mid-latitude)
    if dst_crs is not None and dst_crs.is_geographic:
        # Belgium ~ 51N; 1 deg lat ~ 111.32 km, 1 deg lon ~ 111.32*cos(lat) km
        lat0 = 51.0
        deg_per_m_lat = 1.0 / 111_320.0
        deg_per_m_lon = 1.0 / (111_320.0 * np.cos(np.deg2rad(lat0)))
        target_res_x = target_res_m * deg_per_m_lon
        target_res_y = target_res_m * deg_per_m_lat
        print(f"DTM CRS is geographic. Using resolution ~ ({target_res_x:.6f} deg, {target_res_y:.6f} deg)")
    else:
        target_res_x = target_res_m
        target_res_y = target_res_m
        print(f"DTM CRS is projected. Using resolution = {target_res_m} CRS units")

    # Union bounds in dst_crs
    minx = miny = maxx = maxy = None
    for p in tif_paths:
        with rasterio.open(p) as src:
            if src.crs is None:
                raise RuntimeError(f"TIFF has no CRS: {p}")
            b = src.bounds
            if src.crs != dst_crs:
                b = transform_bounds(src.crs, dst_crs, *b, densify_pts=21)
            if minx is None:
                minx, miny, maxx, maxy = b
            else:
                minx = min(minx, b[0]); miny = min(miny, b[1])
                maxx = max(maxx, b[2]); maxy = max(maxy, b[3])

    width  = int(np.ceil((maxx - minx) / target_res_x))
    height = int(np.ceil((maxy - miny) / target_res_y))
    transform = from_origin(minx, maxy, target_res_x, target_res_y)

    mosaic = np.full((height, width), nodata_out, dtype=np.float32)

    for p in tif_paths:
        with rasterio.open(p) as src:
            src_data = src.read(1).astype(np.float32)

            tmp = np.full_like(mosaic, nodata_out, dtype=np.float32)

            # Important: if src.nodata is None, do NOT force a src_nodata.
            # Let reproject treat all pixels as valid.
            reproject(
                source=src_data,
                destination=tmp,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.average,
                src_nodata=src.nodata,   # can be None
                dst_nodata=nodata_out
            )

            m = np.isfinite(tmp) if np.isnan(nodata_out) else (tmp != nodata_out)
            mosaic[m] = tmp[m]

    return mosaic, transform, dst_crs


def plot_dtm_with_points(mosaic, transform, crs, train_lonlat, test_lonlat, out_path):
    transformer = Transformer.from_crs("EPSG:4326", crs.to_string(), always_xy=True)

    def lonlat_to_xy(lonlat):
        if lonlat.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float64)
        xs, ys = transformer.transform(lonlat[:, 0], lonlat[:, 1])
        return np.stack([xs, ys], axis=1)

    train_xy = lonlat_to_xy(train_lonlat)
    test_xy  = lonlat_to_xy(test_lonlat)

    # extent of the mosaic in CRS units
    left = transform.c
    top  = transform.f
    px   = transform.a
    py   = -transform.e
    H, W = mosaic.shape
    right  = left + W * px
    bottom = top  - H * py
    extent = [left, right, bottom, top]

    # choose a wide figure proportional to map aspect
    map_aspect = (right - left) / max(1e-9, (top - bottom))
    fig_w = 12.0
    fig_h = np.clip(fig_w / map_aspect, 4.5, 8.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=250, constrained_layout=True)

    # robust vmin/vmax
    valid = np.isfinite(mosaic)
    if valid.sum() == 0:
        raise RuntimeError("DTM mosaic has no valid pixels. Cannot plot.")
    vmin = np.nanpercentile(mosaic, 2)
    vmax = np.nanpercentile(mosaic, 98)

    # IMPORTANT: aspect='auto' fills the axes (no giant whitespace)
    im = ax.imshow(
        mosaic,
        extent=extent,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
        cmap="terrain",   # <- better for elevation readability
    )

    # nicer, compact colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Elevation")

    # Train: green circle + halo
    if train_xy.shape[0] > 0:
        ax.scatter(
            train_xy[:, 0], train_xy[:, 1],
            s=70, marker="o",
            c="black", alpha=0.25,
            linewidths=0,
            zorder=4
        )
        ax.scatter(
            train_xy[:, 0], train_xy[:, 1],
            s=32, marker="o",
            c="#00c853",              # green
            edgecolors="black",
            linewidths=0.6,
            alpha=0.95,
            zorder=5,
            label="Train"
        )

    # Test: red circle + halo
    if test_xy.shape[0] > 0:
        ax.scatter(
            test_xy[:, 0], test_xy[:, 1],
            s=80, marker="o",
            c="black", alpha=0.25,
            linewidths=0,
            zorder=4
        )
        ax.scatter(
            test_xy[:, 0], test_xy[:, 1],
            s=36, marker="o",
            c="#ff1744",              # red
            edgecolors="black",
            linewidths=0.6,
            alpha=0.95,
            zorder=6,
            label="Test"
        )

    ax.set_title("DTM (downsampled) with Train/Test Stations")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # legend inside but not covering too much
    ax.legend(loc="lower left", framealpha=0.85)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# MAIN: based on your test() function split + dataloader loop
# -------------------------
def make_dtm_figure(cfg, dtm_glob, out_path, target_res_m):
    # --- THIS BLOCK IS IDENTICAL TO YOUR test() SPLIT BLOCK ---
    with open('data/split.pkl', 'rb') as f:
        split = pickle.load(f)
        good_nb_extended = split['train']
        test_nb = split['test']
    # ---------------------------------------------------------

    # --- DATASETS + LOADERS: same style as your test() ---
    train_dataset = WaterDataset(
        path='data/selected_stats_rainfall_segment.pkl', train=True,
        selected_stations=good_nb_extended, input_type=cfg.dataset.inputs
    )
    test_dataset = WaterDatasetY(
        path='data/selected_stats_rainfall_segment.pkl',
        train=False,
        selected_stations=test_nb,
        input_type=cfg.dataset.inputs
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --- COLLECT LOCATIONS: same loop pattern as your test() (just no seg stuff) ---
    train_coords = []
    test_coords = []

    with torch.no_grad():
        for idx, (x, xs, y, kes, lrain, nrain, valid, loc) in enumerate(train_loader):
            loc_np = loc.detach().cpu().numpy().reshape(-1)

            if loc_np.size < 2:
                continue
            a, b = float(loc_np[0]), float(loc_np[1])

            # auto-detect (lat,lon) vs (lon,lat)
            if (-90 <= a <= 90) and (-180 <= b <= 180):
                lat, lon = a, b
            elif (-90 <= b <= 90) and (-180 <= a <= 180):
                lon, lat = a, b
            else:
                continue

            train_coords.append([lon, lat])

        for idx, (mxs, my, mlrain, mnbxs, mnby, mnrain, loc) in enumerate(test_loader):
            loc_np = loc.detach().cpu().numpy().reshape(-1)
            if loc_np.size < 2:
                continue
            a, b = float(loc_np[0]), float(loc_np[1])

            if (-90 <= a <= 90) and (-180 <= b <= 180):
                lat, lon = a, b
            elif (-90 <= b <= 90) and (-180 <= a <= 180):
                lon, lat = a, b
            else:
                continue

            test_coords.append([lon, lat])

    train_lonlat = np.asarray(train_coords, dtype=np.float64) if train_coords else np.zeros((0, 2), np.float64)
    test_lonlat  = np.asarray(test_coords, dtype=np.float64) if test_coords else np.zeros((0, 2), np.float64)

    print("Collected train GPS:", train_lonlat.shape)
    print("Collected test  GPS:", test_lonlat.shape)
    logging.info(f"Collected train GPS: {train_lonlat.shape}, test GPS: {test_lonlat.shape}")

    # --- Build DTM mosaic ---
    tif_paths = sorted(glob.glob(dtm_glob))
    print("DTM tiles:", len(tif_paths))
    assert len(tif_paths) > 0, f"No tif files found with glob: {dtm_glob}"

    mosaic, transform, crs = build_lowres_dtm_mosaic(
        tif_paths,
        target_res_m=target_res_m,
        dst_crs=None,
        nodata_out=np.nan
    )

    # --- Plot ---
    plot_dtm_with_points(mosaic, transform, crs, train_lonlat, test_lonlat, out_path)
    print("Saved:", out_path)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    parser = argparse.ArgumentParser(description="Create DTM mosaic + overlay train/test stations.")
    parser.add_argument("--cfg", type=str, default="config/idw.yaml")
    parser.add_argument("--dtm_glob", type=str, default="data/dtm/DHMVIIDTMRAS1m_*/GeoTIFF/*.tif")
    parser.add_argument("--out", type=str, default="dtm_train_test.png")
    parser.add_argument("--target_res", type=float, default=250.0, help="Downsample resolution in meters.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    make_dtm_figure(cfg, args.dtm_glob, args.out, args.target_res)


if __name__ == "__main__":
    main()
