import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from skimage.graph import route_through_array
from scipy.ndimage import binary_propagation, binary_dilation

from common.dsm import *


def crop_dsm(dsm, start, end, margin=50):
    """ Crop the DSM around the bounding box of start & end points with a margin. """
    rows, cols = dsm.shape
    min_row = max(0, min(start[0], end[0]) - margin)
    max_row = min(rows, max(start[0], end[0]) + margin)
    min_col = max(0, min(start[1], end[1]) - margin)
    max_col = min(cols, max(start[1], end[1]) + margin)

    cropped_dsm = dsm[min_row:max_row, min_col:max_col]
    return cropped_dsm, (min_row, min_col)


def vis(dsm, path='dsm_crop.png'):
    dsm_disp = (dsm - dsm.min()) / (dsm.max() - dsm.min())
    
    cmap = plt.get_cmap('terrain')  # Or 'jet', 'terrain', etc.
    
    # Apply colormap; returns RGBA (float in [0,1])
    colored_img = cmap(dsm_disp)

    # Drop alpha channel and convert to 8-bit integer
    colored_img = (colored_img[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(colored_img)
    img.save(path)

    return colored_img


def vis_with_path(dsm, indices=None, start=None, end=None, path='dsm_crop.png'):
    # Normalize for display
    dsm_disp = (dsm - dsm.min()) / (dsm.max() - dsm.min())
    cmap = plt.get_cmap('terrain')
    colored_img = cmap(dsm_disp)
    colored_img = (colored_img[:, :, :3] * 255).astype(np.uint8)
    
    img = Image.fromarray(colored_img)
    draw = ImageDraw.Draw(img)

    # Draw path if given
    if indices is not None:
        # Convert (row, col) to (x, y)
        path_points = [(col, row) for row, col in indices]
        draw.line(path_points, fill=(255, 0, 0), width=4)  # Blue path

    # Draw start/end points if given
    if start is not None:
        draw.ellipse(
            [(start[1]-3, start[0]-3), (start[1]+3, start[0]+3)],
            fill=(255, 0, 0)
        )
    if end is not None:
        draw.ellipse(
            [(end[1]-3, end[0]-3), (end[1]+3, end[0]+3)],
            fill=(255, 0, 0)
        )

    img.save(path)


def make_downhill_cost(dsm):
    cost = np.zeros_like(dsm)
    # For each pixel, the cost to move to a neighbor is:
    # - 1 if downhill or flat
    # - infinity if uphill
    # We'll use this in a custom neighbor function (see below)
    return cost


def find_path(dsm, start, end, ratio=1.2):
    cost_surface = np.maximum(0, dsm)  # We'll modify this below
    
    threshold = dsm[start] * ratio
    cost_surface[dsm < threshold] = (dsm[dsm < threshold] / 4) ** 2
    cost_surface[dsm >= threshold] = cost_surface[dsm >= threshold] ** 2
    
    building_mask = (dsm > dsm[start] * ratio)

    cost_surface[building_mask] = 1e3  # or np.inf if skimage supports it
    
    indices, weight = route_through_array(
        cost_surface,
        start,
        end,
        fully_connected=True,  # 8-connectivity (diagonals allowed)
    )

    indices = np.array(indices)
    return indices, building_mask


def corridor_mask(indices, shape, width=4):
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(indices.T)] = True
    return binary_dilation(mask, iterations=width)


def barrier_aware_corridor(dsm, indices, threshold, building_mask, corridor_width=4):
    allowed = (dsm <= threshold) & (~building_mask)
    # Initialize the seed as the path
    region = np.zeros_like(dsm, dtype=bool)
    region[tuple(indices.T)] = True

    # Iteratively dilate, always masking with allowed
    for _ in range(corridor_width):
        new_region = binary_dilation(region) & allowed
        # Stop if no new expansion
        if np.array_equal(new_region, region):
            break
        region = new_region

    return region


def flood_fill_surface(dsm, indices, threshold, building_mask, corridor_width=15):
    allowed = (dsm <= threshold) & (~building_mask)
    # corridor = corridor_mask(indices, dsm.shape, width=corridor_width)
    corridor = barrier_aware_corridor(dsm, indices, threshold, building_mask, corridor_width=corridor_width)
    mask = allowed & corridor  # Only allow propagation inside corridor and under threshold

    # Use all path pixels as seeds
    seed = np.zeros_like(dsm, dtype=bool)
    seed[tuple(indices.T)] = True

    flooded = binary_propagation(seed, mask=mask)
    return flooded


def save_overlay(colored_img, flooded_mask, indices, start, out_path='dsm_flooded.png'):
    # Convert to RGBA for transparency
    img = Image.fromarray(colored_img).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    # Draw the flooded mask with semi-transparent blue
    for y, x in zip(*np.where(flooded_mask)):
        draw.point((x, y), fill=(255, 144, 30, 100))  # DodgerBlue, alpha=100

    # Draw the path (red line)
    path_points = [(col, row) for row, col in indices]
    if len(path_points) > 1:
        draw.line(path_points, fill=(255, 0, 0, 255), width=4)

    # Draw the start point (yellow dot)
    if start is not None:
        sx, sy = start[1], start[0]
        draw.ellipse([(sx-4, sy-4), (sx+4, sy+4)], fill=(255, 255, 0, 255))

    # Combine overlays
    out = Image.alpha_composite(img, overlay)
    out.save(out_path)


def main():
    if not os.path.exists('dsm.pkl'):
        p = '/scratch/antruong/DFW/data/dsm/DHMVIIDSMRAS1m_k13.tif'
        rpj_p = '/scratch/antruong/DFW/data/dsm/DHMVIIDSMRAS1m_k13_rpj.tif'
        # reproject_dsm(p, rpj_p)

        dsm, transform, bounds = load_dsm(rpj_p)
        min_lon, min_lat = bounds.left, bounds.bottom
        max_lon, max_lat = bounds.right, bounds.top
        print([min_lon, min_lat, max_lon, max_lat])

        im = dsm[2000:5000, 2000:5000]
        with open('dsm.pkl', 'wb') as f:
            pickle.dump(im, f)
    else:
        with open('dsm.pkl', 'rb') as f:
            im = pickle.load(f)

    start = (2497, 2501)    # (row, col)
    end = (2234, 2639)      # (row, col)
    if im[start] < im[end]:
        start, end = end, start

    indices, building_mask = find_path(im, start, end)

    path_elev = im[tuple(indices.T)]
    threshold = path_elev.max() * 1.2  # Max elevation along path
    flooded_mask = flood_fill_surface(im, indices, threshold, building_mask)
    print('Flooded pixels:', flooded_mask.sum())
    colored_img = vis(im, 'dsm_crop.png')
    save_overlay(colored_img, flooded_mask, indices, start, out_path='dsm_flooded.png')
    vis_with_path(
        im, 
        indices=indices, 
        start=start, 
        end=end, 
        path='dsm_path_overlay.png'
    )


main()