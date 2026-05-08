"""
Project LiDAR points onto a camera image using a KITTI-format calib.txt.

Usage:
    python tools/project_lidar_to_image.py \
        --seq_dir  data/validation/201 \
        --frame    000050 \
        --output   /tmp/projection.png

    # batch: save every N-th frame
    python tools/project_lidar_to_image.py \
        --seq_dir  data/validation/201 \
        --batch    --stride 50 \
        --output   /tmp/proj_201/
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
def load_calib(calib_path: str):
    """
    Parse a KITTI-format calib.txt.
    Returns:
        P2  : (3, 4) camera projection matrix (for image_2)
        Tr  : (3, 4) LiDAR-to-camera transform
    """
    data = {}
    with open(calib_path) as f:
        for line in f:
            if ":" in line:
                key, val = line.split(":", 1)
                data[key.strip()] = np.fromstring(val, dtype=np.float64, sep=" ")
    P2 = data["P2"].reshape(3, 4)
    Tr = data["Tr"].reshape(3, 4)
    return P2, Tr


def load_velodyne(bin_path: str) -> np.ndarray:
    """Load a .bin pointcloud → (N, 4) float32  [x, y, z, intensity]."""
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts


def project(pts_lidar: np.ndarray, P2: np.ndarray, Tr: np.ndarray,
            img_h: int, img_w: int, min_depth: float = 0.5):
    """
    Project LiDAR points into image space.

    Returns:
        uv      : (M, 2) int  pixel coordinates of visible points
        depths  : (M,)   float  depth of each point (metres)
        mask    : (N,)   bool  which input points are visible
    """
    N = pts_lidar.shape[0]
    # homogeneous LiDAR coords  (N, 4)
    ones = np.ones((N, 1), dtype=np.float64)
    pts_h = np.hstack([pts_lidar[:, :3].astype(np.float64), ones])   # (N,4)

    # LiDAR → camera  (N, 3)
    pts_cam = (Tr @ pts_h.T).T   # (N, 3)

    # keep only points in front of the camera
    depth = pts_cam[:, 2]
    front = depth > min_depth

    pts_cam_f = pts_cam[front]   # (M, 3)
    depth_f   = depth[front]

    # project  (M, 3) = P2 @ (M, 4).T  → (3, M)
    ones_f = np.ones((pts_cam_f.shape[0], 1))
    pts_cam_h = np.hstack([pts_cam_f, ones_f])                 # (M, 4)
    uvw = (P2 @ pts_cam_h.T).T                                 # (M, 3)

    u = (uvw[:, 0] / uvw[:, 2]).astype(int)
    v = (uvw[:, 1] / uvw[:, 2]).astype(int)

    # keep points inside the image
    in_img = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)

    uv     = np.stack([u[in_img], v[in_img]], axis=1)
    depths = depth_f[in_img]

    # full mask over original N points
    full_mask = np.zeros(N, dtype=bool)
    idx_front = np.where(front)[0]
    full_mask[idx_front[in_img]] = True

    return uv, depths, full_mask


def depth_colormap(depths: np.ndarray,
                   min_d: float = 2.0, max_d: float = 40.0) -> np.ndarray:
    """Map depth values to BGR colours using JET colormap."""
    norm = np.clip((depths - min_d) / (max_d - min_d), 0, 1)
    # JET: blue=near, red=far  (reverse for intuitive near=warm)
    norm_u8 = (norm * 255).astype(np.uint8)
    colors = cv2.applyColorMap(norm_u8[:, None], cv2.COLORMAP_JET).squeeze(1)
    return colors  # (M, 3) BGR


def draw_projection(img: np.ndarray, uv: np.ndarray, depths: np.ndarray,
                    dot_radius: int = 2) -> np.ndarray:
    """Overlay projected points on image, coloured by depth."""
    out = img.copy()
    colors = depth_colormap(depths)
    for (u, v), bgr in zip(uv, colors):
        cv2.circle(out, (int(u), int(v)), dot_radius, bgr.tolist(), -1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
def process_frame(seq_dir: Path, frame: str, output_path: Path,
                  dot_radius: int = 2):
    calib_path = seq_dir / "calib.txt"
    bin_path   = seq_dir / "velodyne" / f"{frame}.bin"
    img_path   = seq_dir / "image_2"  / f"{frame}.png"

    assert calib_path.exists(), f"calib.txt not found: {calib_path}"
    assert bin_path.exists(),   f"velodyne file not found: {bin_path}"
    assert img_path.exists(),   f"image not found: {img_path}"

    P2, Tr = load_calib(str(calib_path))
    pts    = load_velodyne(str(bin_path))
    img    = cv2.imread(str(img_path))
    h, w   = img.shape[:2]

    uv, depths, _ = project(pts, P2, Tr, h, w)
    result = draw_projection(img, uv, depths, dot_radius)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    print(f"  [{frame}] {len(uv):5d} points projected → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Project LiDAR onto image_2")
    parser.add_argument("--seq_dir", required=True,
                        help="Sequence root, e.g. data/validation/201")
    parser.add_argument("--frame", default=None,
                        help="Single frame stem, e.g. 000050")
    parser.add_argument("--batch", action="store_true",
                        help="Process all frames in the sequence")
    parser.add_argument("--stride", type=int, default=1,
                        help="Only save every N-th frame in batch mode")
    parser.add_argument("--output", default="/tmp/projection.png",
                        help="Output file (single) or directory (batch)")
    parser.add_argument("--dot_radius", type=int, default=2,
                        help="Projected point dot radius in pixels")
    parser.add_argument("--min_depth", type=float, default=0.5,
                        help="Minimum depth to project (metres)")
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)

    if args.batch:
        frames = sorted(p.stem for p in (seq_dir / "velodyne").glob("*.bin"))
        frames = frames[::args.stride]
        out_dir = Path(args.output)
        print(f"Batch mode: {len(frames)} frames → {out_dir}")
        for frame in frames:
            out_file = out_dir / f"{frame}.png"
            process_frame(seq_dir, frame, out_file, args.dot_radius)
    else:
        frame = args.frame or "000000"
        out_file = Path(args.output)
        process_frame(seq_dir, frame, out_file, args.dot_radius)


if __name__ == "__main__":
    main()
