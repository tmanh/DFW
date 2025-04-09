import cv2
import imageio
import numpy as np


def save_raw(dsm, out="plot_raw.jpg"):
    dsm_normalized = (dsm - np.min(dsm)) / (np.max(dsm) - np.min(dsm))
    dsm_img = (dsm_normalized * 255).astype(np.uint8)
    imageio.imwrite(out, dsm_img)


def save_path(dsm, path, out="plot_path.jpg"):
    """ Quickly render DSM with overlaid path and save as image. """
    # Normalize DSM to 0–255 grayscale
    dsm_norm = (dsm - np.nanmin(dsm)) / (np.nanmax(dsm) - np.nanmin(dsm))
    base_img = (dsm_norm * 255).astype(np.uint8)

    # Stack to RGB
    rgb = np.stack([base_img] * 3, axis=-1)
    rgb = (rgb * 0.8).astype(np.uint8)  # Slightly darken for contrast

    # Draw path
    for i in range(1, len(path)):
        y1, x1 = map(int, path[i - 1])
        y2, x2 = map(int, path[i])
        cv2.line(rgb, (x1, y1), (x2, y2), color=(255, 255, 0), thickness=1)  # Blue path

    # Save image
    imageio.imwrite(out, rgb)

    print(f"Path image saved to {out}")


def save_expanded_waterway_plot(dsm, waterway_region, filename="expanded_waterway.png"):
    """Fast DSM visualization with waterway region in blue, saved using imageio."""
    # Normalize DSM to 0–255 and convert to uint8
    dsm_normalized = (dsm - np.nanmin(dsm)) / (np.nanmax(dsm) - np.nanmin(dsm))
    dsm_img = (dsm_normalized * 255).astype(np.uint8)

    # Create grayscale RGB image
    rgb = np.stack([dsm_img] * 3, axis=-1)  # shape (H, W, 3)

    # Slightly darken overall image
    rgb = (rgb * 0.7).astype(np.uint8)

    # Paint waterway region in blue
    rgb[waterway_region] = [255, 255, 0]

    # Save image (fast and no axis/rendering overhead)
    imageio.imwrite(filename, rgb)

    print(f"Expanded waterway map saved as {filename}")