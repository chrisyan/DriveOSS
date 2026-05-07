#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import open3d as o3d
except Exception:
    print("ERROR: Open3D is required: pip install open3d", file=sys.stderr)
    raise


# ============================================================
# Anomaly GT visualization classes
# -1: unlabeled / ignore  -> black
#  1: outlier / anomaly   -> red
#  2: ID classes          -> purple
# ============================================================
COLOR_MAP = {
    -1: [0.000, 0.000, 0.000],   # unlabeled / ignore, black
     1: [1.000, 0.000, 0.000],   # outlier / anomaly, red
     2: [0.500, 0.000, 0.500],   # ID classes, purple
}

CLASS_NAMES = {
    -1: "unlabeled / ignore",
     1: "outlier / anomaly",
     2: "ID class",
}


# ============================================================
# Raw SemanticKITTI / OSAW labels
# raw 0, 52, 99 -> unlabeled / ignore (-1)
# raw 1, 2      -> outlier / anomaly (1)
# all valid SemanticKITTI ID classes -> ID class (2)
# ============================================================
RAW_UNLABELED_IDS = {0, 52, 99}
RAW_OUTLIER_IDS = {1, 2}
RAW_ID_IDS = {
    10, 11, 13, 15, 16, 18, 20,
    30, 31, 32,
    40, 44, 48, 49,
    50, 51,
    60,
    70, 71, 72,
    80, 81,
    252, 253, 254, 255, 256, 257, 258, 259,
}

# If label_mode == learning:
# learning 0 is treated as unlabeled / ignore.
# learning 1--19 are ID classes.
# learning 20 or 21 are treated as outlier / anomaly if your previous script used them.
LEARNING_UNLABELED_IDS = {0, -1}
LEARNING_OUTLIER_IDS = {20, 21}
LEARNING_ID_IDS = set(range(1, 20))


def read_velodyne_bin(bin_path: Path) -> np.ndarray:
    scan = np.fromfile(str(bin_path), dtype=np.float32)

    if scan.size % 4 != 0:
        raise ValueError(f"Invalid .bin file: {bin_path}")

    return scan.reshape(-1, 4)


def read_labels(label_path: Path) -> np.ndarray:
    return np.fromfile(str(label_path), dtype=np.uint32)


def map_raw_to_anomaly_gt(raw_sem_ids: np.ndarray) -> np.ndarray:
    """
    Convert raw semantic labels to anomaly GT labels:
        -1 = unlabeled / ignore
         1 = outlier / anomaly
         2 = ID class
    """
    anomaly_gt = np.full(raw_sem_ids.shape, -1, dtype=np.int32)

    anomaly_gt[np.isin(raw_sem_ids, list(RAW_ID_IDS))] = 2
    anomaly_gt[np.isin(raw_sem_ids, list(RAW_OUTLIER_IDS))] = 1
    anomaly_gt[np.isin(raw_sem_ids, list(RAW_UNLABELED_IDS))] = -1

    return anomaly_gt


def map_learning_to_anomaly_gt(learning_ids: np.ndarray) -> np.ndarray:
    """
    Convert learning labels to anomaly GT labels:
        -1 = unlabeled / ignore
         1 = outlier / anomaly
         2 = ID class
    """
    anomaly_gt = np.full(learning_ids.shape, -1, dtype=np.int32)

    anomaly_gt[np.isin(learning_ids, list(LEARNING_ID_IDS))] = 2
    anomaly_gt[np.isin(learning_ids, list(LEARNING_OUTLIER_IDS))] = 1
    anomaly_gt[np.isin(learning_ids, list(LEARNING_UNLABELED_IDS))] = -1

    return anomaly_gt


def colors_from_anomaly_gt(anomaly_gt: np.ndarray) -> np.ndarray:
    colors = np.zeros((anomaly_gt.shape[0], 3), dtype=np.float32)

    for class_id, color in COLOR_MAP.items():
        mask = anomaly_gt == class_id
        colors[mask] = np.array(color, dtype=np.float32)

    return colors


def visualize(
    points_xyz: np.ndarray,
    colors01: np.ndarray,
    point_size: float = 3.0,
    save_ply: Optional[Path] = None,
    save_png: Optional[Path] = None,
):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors01)

    if save_ply is not None:
        o3d.io.write_point_cloud(str(save_ply), pcd)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Anomaly GT Visualization",
        width=1280,
        height=720,
        visible=True,
    )

    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.array([1.0, 1.0, 1.0])

    ctr = vis.get_view_control()
    ctr.set_front([0.0, -1.0, 0.0])
    ctr.set_lookat([0.0, 0.0, 0.0])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(0.35)

    vis.poll_events()
    vis.update_renderer()

    if save_png is not None:
        vis.capture_screen_image(str(save_png), do_render=True)

    print("Close the Open3D window to finish.")
    vis.run()
    vis.destroy_window()


def resolve_paths(args):
    if args.velodyne and args.label:
        return Path(args.velodyne), Path(args.label)

    if not args.dataset_root or not args.seq or not args.frame:
        print(
            "ERROR: Provide either (--velodyne and --label) or "
            "(--dataset_root, --seq, --frame).",
            file=sys.stderr,
        )
        sys.exit(1)

    seq_dir = Path(args.dataset_root) / "sequences" / args.seq
    bin_path = seq_dir / "velodyne" / f"{args.frame}.bin"

    if args.pred_root:
        label_path = Path(args.pred_root) / f"{args.frame}.label"
    else:
        label_path = seq_dir / "labels" / f"{args.frame}.label"

    return bin_path, label_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize anomaly GT: 1=outlier, 2=ID class, -1=unlabeled"
    )

    parser.add_argument(
        "--velodyne",
        type=Path,
        default=Path(
            "/media/zhiranworkstation/Expansion/tools/point_labeler/data/OOD_scenes/434/velodyne/000132.bin"
        ),
        help="Path to .bin point cloud",
    )

    parser.add_argument(
        "--label",
        type=Path,
        default=Path(
            "/media/zhiranworkstation/Expansion/tools/point_labeler/data/OOD_scenes/434/labels/000132.label"
        ),
        help="Path to .label file",
    )

    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--seq", type=str, default=None)
    parser.add_argument("--frame", type=str, default=None)
    parser.add_argument("--pred_root", type=Path, default=None)

    parser.add_argument(
        "--label_mode",
        type=str,
        default="raw",
        choices=["raw", "learning"],
        help=(
            "raw: .label contains original SemanticKITTI / OSAW IDs. "
            "learning: .label already contains learning labels."
        ),
    )

    parser.add_argument("--point_size", type=float, default=5.0)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--save_ply", type=Path, default=None)
    parser.add_argument("--save_png", type=Path, default=Path("anomaly_gt_000132.png"))

    args = parser.parse_args()

    bin_path, label_path = resolve_paths(args)

    if not bin_path.exists():
        print(f"ERROR: .bin not found: {bin_path}", file=sys.stderr)
        sys.exit(1)

    if not label_path.exists():
        print(f"ERROR: .label not found: {label_path}", file=sys.stderr)
        sys.exit(1)

    scan = read_velodyne_bin(bin_path)
    labels = read_labels(label_path)

    if scan.shape[0] != labels.shape[0]:
        print(
            f"ERROR: Point count mismatch: "
            f"{scan.shape[0]} points vs {labels.shape[0]} labels",
            file=sys.stderr,
        )
        sys.exit(1)

    sem_ids = (labels & np.uint32(0xFFFF)).astype(np.int32)

    if args.label_mode == "raw":
        anomaly_gt = map_raw_to_anomaly_gt(sem_ids)
    else:
        anomaly_gt = map_learning_to_anomaly_gt(sem_ids)

    print(f"\nFrame: {bin_path.name}")
    print(f"Number of points: {scan.shape[0]}")
    print(f"Label mode: {args.label_mode}")
    print("Anomaly GT mapping: 1=outlier, 2=ID class, -1=unlabeled")

    unique, counts = np.unique(anomaly_gt, return_counts=True)

    print("\nAnomaly GT distribution:")
    print(f"{'ID':<6} {'Class name':<20} {'Points':<10}")
    print("-" * 40)

    for cid, count in zip(unique, counts):
        name = CLASS_NAMES.get(int(cid), "unknown")
        print(f"{int(cid):<6} {name:<20} {int(count):<10}")

    colors01 = colors_from_anomaly_gt(anomaly_gt)
    pts_xyz = scan[:, :3].copy()

    if args.downsample > 1:
        idx = np.arange(pts_xyz.shape[0])[::args.downsample]
        pts_xyz = pts_xyz[idx]
        colors01 = colors01[idx]

    visualize(
        pts_xyz,
        colors01,
        point_size=args.point_size,
        save_ply=args.save_ply,
        save_png=args.save_png,
    )


if __name__ == "__main__":
    main()
