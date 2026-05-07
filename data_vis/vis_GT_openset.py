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


COLOR_MAP = {
    0:  [0.000, 0.000, 0.000],   # unlabeled

    1:  [0.392, 0.588, 0.961],   # car
    2:  [0.392, 0.902, 0.961],   # bicycle
    3:  [0.118, 0.235, 0.588],   # motorcycle
    4:  [0.314, 0.118, 0.706],   # truck
    5:  [0.000, 0.000, 1.000],   # other-vehicle
    6:  [1.000, 0.118, 0.118],   # person
    7:  [1.000, 0.157, 0.784],   # bicyclist
    8:  [0.588, 0.118, 0.353],   # motorcyclist
    9:  [1.000, 0.000, 1.000],   # road
    10: [1.000, 0.588, 1.000],   # parking
    11: [0.294, 0.000, 0.294],   # sidewalk
    12: [0.686, 0.000, 0.294],   # other-ground
    13: [1.000, 0.784, 0.000],   # building
    14: [1.000, 0.471, 0.196],   # fence
    15: [0.000, 0.686, 0.000],   # vegetation
    16: [0.529, 0.235, 0.000],   # trunk
    17: [0.588, 0.941, 0.314],   # terrain
    18: [1.000, 0.941, 0.588],   # pole
    19: [1.000, 0.000, 0.000],   # traffic-sign

    20: [1.000, 0.000, 0.000],   # outlier, bright red

}


CLASS_NAMES = {
    0:  "unlabeled",
    1:  "car",
    2:  "bicycle",
    3:  "motorcycle",
    4:  "truck",
    5:  "other-vehicle",
    6:  "person",
    7:  "bicyclist",
    8:  "motorcyclist",
    9:  "road",
    10: "parking",
    11: "sidewalk",
    12: "other-ground",
    13: "building",
    14: "fence",
    15: "vegetation",
    16: "trunk",
    17: "terrain",
    18: "pole",
    19: "traffic-sign",

    20: "outlier",

}

# note: semantickitti raw label "outlier" are NOT mapped to "unlabeled" in the learning map
LEARNING_MAP = {
    0: 0,      # unlabeled
    1: 20,     # outlier


    10: 1,
    11: 2,
    13: 5,
    15: 3,
    16: 5,
    18: 4,
    20: 5,
    30: 6,
    31: 7,
    32: 8,
    40: 9,
    44: 10,
    48: 11,
    49: 12,
    50: 13,
    51: 14,
    52: 0,
    60: 9,
    70: 15,
    71: 16,
    72: 17,
    80: 18,
    81: 19,
    99: 0,
    252: 1,
    253: 7,
    254: 6,
    255: 8,
    256: 5,
    257: 5,
    258: 4,
    259: 5,
}


def read_velodyne_bin(bin_path: Path) -> np.ndarray:
    scan = np.fromfile(str(bin_path), dtype=np.float32)

    if scan.size % 4 != 0:
        raise ValueError(f"Invalid .bin file: {bin_path}")

    return scan.reshape(-1, 4)


def read_labels(label_path: Path) -> np.ndarray:
    return np.fromfile(str(label_path), dtype=np.uint32)


def map_raw_to_learning(raw_sem_ids: np.ndarray) -> np.ndarray:
    mapped = np.zeros_like(raw_sem_ids, dtype=np.int32)

    for raw_id, train_id in LEARNING_MAP.items():
        mapped[raw_sem_ids == raw_id] = train_id

    return mapped


def colors_from_learning_ids(learning_ids: np.ndarray) -> np.ndarray:
    colors = np.zeros((learning_ids.shape[0], 3), dtype=np.float32)

    for class_id, color in COLOR_MAP.items():
        mask = learning_ids == class_id
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
        window_name="SemanticKITTI Label Visualization",
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
        description="Visualize SemanticKITTI / OSAW labels with Open3D"
    )

    parser.add_argument(
        "--velodyne",
        type=Path,
        default=Path(
            "xxx/data/OOD_scenes/434/velodyne/000132.bin"
        ),
        help="Path to .bin point cloud",
    )

    parser.add_argument(
        "--label",
        type=Path,
        default=Path(
            "xxx/data/OOD_scenes/434/labels/000132.label"
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
            "raw: .label contains original IDs, e.g. 1=outlier, 2=anomaly, "
            "10=car, 40=road. "
            "learning: .label already contains learning labels."
        ),
    )

    parser.add_argument("--point_size", type=float, default=5.0)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--save_ply", type=Path, default=None)
    parser.add_argument("--save_png", type=Path, default=Path("pred_000132.png"))

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
        learning_ids = map_raw_to_learning(sem_ids)
    else:
        learning_ids = sem_ids

    print(f"\nFrame: {bin_path.name}")
    print(f"Number of points: {scan.shape[0]}")
    print(f"Label mode: {args.label_mode}")

    unique, counts = np.unique(learning_ids, return_counts=True)

    print("\nClass distribution:")
    print(f"{'ID':<6} {'Class name':<20} {'Points':<10}")
    print("-" * 40)

    for cid, count in zip(unique, counts):
        name = CLASS_NAMES.get(int(cid), "unknown")
        print(f"{int(cid):<6} {name:<20} {int(count):<10}")

    colors01 = colors_from_learning_ids(learning_ids)
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