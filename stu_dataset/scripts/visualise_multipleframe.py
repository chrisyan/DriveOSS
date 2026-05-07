from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from utils.common import load_labels, load_point_cloud


def visualize_predictions(points, labels, scores, save_path=None, frame_id="Frame"):
    fig = plt.figure(figsize=(16, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title(f"{frame_id} - Ground Truth Labels")
    inlier_mask = labels == 0
    outlier_mask = labels == 2

    ax1.scatter(points[inlier_mask, 0], points[inlier_mask, 1], points[inlier_mask, 2],
                c='blue', s=1, label="Inlier (0)")
    ax1.scatter(points[outlier_mask, 0], points[outlier_mask, 1], points[outlier_mask, 2],
                c='red', s=1, label="Outlier (2)")
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title(f"{frame_id} - Predicted Anomaly Scores")
    p = ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
                    c=scores, cmap='hot', s=1)
    fig.colorbar(p, ax=ax2, label="Anomaly Score")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def main():
    data_root = Path("")         # replace this
    pred_root = Path("")     # replace this
    output_dir = Path("")             # save images here
    output_dir.mkdir(parents=True, exist_ok=True)

    for seq_path in tqdm(sorted(data_root.glob("1[0-9][0-9]")), desc="Sequences"):
        if not seq_path.is_dir():
            continue

        seq_id = seq_path.name
        velodyne_dir = seq_path / "velodyne"
        label_dir = seq_path / "labels"
        pred_dir = pred_root / seq_id

        for pcd_file in tqdm(sorted(velodyne_dir.glob("*.bin")), leave=False, desc=f"Frames in {seq_id}"):
            frame_id = pcd_file.stem

            label_file = label_dir / f"{frame_id}.label"
            pred_file = pred_dir / f"{frame_id}.txt"
            image_path = output_dir / f"{seq_id}_{frame_id}.png"

            if not label_file.exists() or not pred_file.exists():
                print(f"Skipping {frame_id} (missing label or prediction)")
                continue

            # Load data
            points, _ = load_point_cloud(pcd_file)
            labels, _ = load_labels(label_file)
            scores = np.loadtxt(pred_file).astype(np.float32)

            # Visualize and save
            visualize_predictions(points, labels, scores, save_path=image_path, frame_id=f"{seq_id}_{frame_id}")


if __name__ == "__main__":
    main()
