from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_point_cloud(scan_path):
    scan = np.fromfile(scan_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    points = scan[:, :3]
    return points, scan

def load_labels(label_path):
    labels = np.fromfile(label_path, dtype=np.uint32)
    return labels, None

def find_true_positives(gt_labels, pred_scores, threshold=0.3):
    pred_labels = (pred_scores > threshold).astype(np.uint8) * 2  # predicted outlier = 2

    inliers = (gt_labels != 0) & (gt_labels != 2)
    outliers = (gt_labels == 2)

    # True Positive mask
    tp = outliers & (pred_labels == 2)
    return tp, np.sum(tp)

def main():
    data_root = Path("")         
    pred_root = Path("")     

    threshold = 0.5

    frames_with_tp = []

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

            if not label_file.exists() or not pred_file.exists():
                continue

            # Load data
            points, _ = load_point_cloud(pcd_file)
            gt_labels, _ = load_labels(label_file)
            pred_scores = np.loadtxt(pred_file).astype(np.float32)

            tp_mask, tp_count = find_true_positives(gt_labels, pred_scores, threshold)

            if tp_count > 0:
                print(f"Sequence: {seq_id}, Frame: {frame_id}, True Positives: {tp_count}")
                frames_with_tp.append((seq_id, frame_id, tp_count))

    print(f"\nTotal frames with true positives: {len(frames_with_tp)}")
    # Optionally save to file
    with open("frames_with_true_positives.txt", "w") as f:
        for seq_id, frame_id, count in frames_with_tp:
            f.write(f"{seq_id},{frame_id},{count}\n")

if __name__ == "__main__":
    main()
