import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, average_precision_score, roc_curve
from tqdm import tqdm

#from matplotlib import cm
#import open3d as o3d

from utils.common import convert_to_builtin_types, load_labels, load_point_cloud

"""
# Example usage in a model
metrics = PointOODMetricsCalculator()

for batch in dataloader:
    inputs, labels = preprocess(batch)
    outputs = model(inputs)

    # Assume anomaly_scores is derived from model outputs
    anomaly_scores = compute_anomaly_scores(outputs)

    # Update metrics (convert tensors to numpy if needed)
    metrics.update(inputs.pcd.numpy(), anomaly_scores.numpy(), labels.numpy())

# Get final metrics
final_metrics = metrics.compute_metrics()
"""


class PointOODMetricsCalculator:
    min_eval_distance = 2.5
    max_eval_distance = 50
    min_num_points_to_eval = 5

    def __init__(
        self,
    ):
        self.all_scores = []
        self.all_labels = []
        #modified
        self.frame_counter = 0
        self.current_sequence = None

    def update(self, points, scores, target):
        """Update the stored scores and labels with new data.

        Args:
            points (np.ndarray): Point cloud coordinates
            scores (np.ndarray): Anomaly scores (higher means more anomalous).
            target (np.ndarray): Ground truth labels
        """
        distances = np.linalg.norm(points, axis=1)
        
        # Process labels and apply distance mask
        #Any point not equal to 0 in target becomes 0 (considered an inlier)
        #Any point equal to 0 becomes -1 (will be ignored)

        inlier_labels = np.where(target != 0, 0, -1)
        #print("inlier_labels:", inlier_labels)
        
        #Any point equals 2 in target becomes 1 (anomaly)
        #All other points keep their value from inlier_labels
        processed_labels = np.where(target == 2, 1, inlier_labels)
        #print("processed_labels:", processed_labels)

        processed_labels = np.where(
            (distances > self.max_eval_distance) | (distances < self.min_eval_distance),
            -1,
            processed_labels,
        )
'''
        #Verification data
        verification_data= [
            "=== Label Processing Verification ===",
            "Original Target | Processed Label | Distance",
            "-------------------------------------------"
        ]

        #Get indices of all target==2
        anomaly_indices=np.where(target==2)[0] 

        #Add sample points
        sample_indices=list(range(20))+list(range(-20,0))

        #Add anomaly indices max 20
        if len(anomaly_indices)>0:
            sample_indices.extend(anomaly_indices[:20].tolist())

        for i in sample_indices:
            verification_data.append(
                f"{target[i]:>13} | {processed_labels[i]:>15} | {distances[i]:>8.2f}"
                )

        verification_data.extend([
            "\nUnique Value Counts:",
            f"Original Targets: {dict(zip(*np.unique(target, return_counts=True)))}",
            f"Processed Labels: {dict(zip(*np.unique(processed_labels, return_counts=True)))}",
            f"Total anomalies (target==2):{len(anomaly_indices)}",
            "-------------------------------------------\n"
        ])

        # Save to file
        with open('label_verification.txt', 'a') as f:
            f.write('\n'.join(verification_data) + '\n')
'''
        ignore_mask = processed_labels != -1
        #print("ignore_mask:",ignore_mask)
        labels = processed_labels[ignore_mask]
        #print("labels:",labels)

        # Only evaluate if sufficient anomaly points
        if np.sum(labels) < self.min_num_points_to_eval:
            return
        if len(scores) != len(target):
            raise ValueError("Prediction and label count mismatch")

        prediction = scores[ignore_mask]
        self.all_scores.append(prediction)
        self.all_labels.append(labels)


'''
    def visulise_prediction_scores(self, points, scores, target):
        
        if not self.all_scores:
            return {}
        #combine accumulated data


        #all_points = np.concatenate(self.points, axis=0) 
        all_labels = np.concatenate(self.all_labels, axis=0)
        all_scores = np.concatenate(self.all_scores, axis=0)

        #Create colour map
        normalized_scores = (all_scores - all_scores.min()) / (all_scores.max() - all_scores.min())
        jet = cm.get_cmap("jet")
        score_colors = jet(normalized_scores)[:, :3]

        #Create Open3D point cloud
        pcd_scores = o3d.geometry.PointCloud()
        pcd_scores.points = o3d.utility.Vector3dVector(points)
        pcd_scores.colors = o3d.utility.Vector3dVector(score_colors)

        #Add coordinate frame for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

        #Visualise
        o3d.visualization.draw_geometries([pcd_scores,coord_frame], window_name="Prediction Confidence")

    def visulise_GT(self, points, scores, target):
        
        if not self.all_scores:
            return {}
        #combine accumulated data


        #all_points = np.concatenate(self.points, axis=0) 
        all_labels = np.concatenate(self.all_labels, axis=0)
        all_scores = np.concatenate(self.all_scores, axis=0)

        #Create colour map
        gt_colors = np.zeros((len(all_labels),3))
        gt_colors[all_labels == 0] = [0.5,0,0.5] # purple for inlier
        gt_colors[all_labels == 1] = [0,1,0] #green for anomaly
        gt_colors[all_labels >= 1] = [0,0,0] # black for void
        gt_colors[all_labels <= 0] = [0,0,0] # black for void

        #Create Open3D point cloud
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(points)
        pcd_gt.colors = o3d.utility.Vector3dVector(gt_colors)

        #Add coordinate frame for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

        #Visualise
        o3d.visualization.draw_geometries([pcd_gt,coord_frame], window_name="Ground Truth")
'''
    def compute_metrics(self):
        """Compute OOD detection metrics on accumulated data.

        Returns:
            dict: Metrics including AP, FPR95, AUROC, and optimal threshold.
        """
        if not self.all_scores:
            return {}

        targets = np.concatenate(self.all_labels, axis=0)
        predictions = np.concatenate(self.all_scores, axis=0)

        print("Labels:", np.unique(targets, return_counts=True))

        AP = average_precision_score(y_true=targets, y_score=predictions)
        roc_auc, fpr, threshold = self._calculate_auroc(predictions, targets)

        return {
            "AP": AP * 100,
            "FPR95": fpr * 100,
            "AUROC": roc_auc * 100,
            "threshold": threshold,
        }

    @staticmethod
    def _calculate_auroc(predictions, targets):
        fpr, tpr, thresholds = roc_curve(y_true=targets, y_score=predictions)
        roc_auc = auc(fpr, tpr)
        fpr_best = 0
        optimal_threshold = 0

        for tpr_val, fpr_val, thr in zip(tpr, fpr, thresholds):
            if tpr_val > 0.95:
                fpr_best = fpr_val
                optimal_threshold = thr
                break

        return roc_auc, fpr_best, optimal_threshold


def main(args):
    metrics_calculator = PointOODMetricsCalculator()

    for seq_path in tqdm(sorted(list(args.data_dir.glob("1[0-9][0-9]")))):
        if seq_path.is_dir():
            lidar_files = sorted((seq_path / "velodyne").glob("*.bin"))

            for pcd_file in tqdm(lidar_files, leave=False, position=1):
                points, _ = load_point_cloud(pcd_file)

                label_file = seq_path / "labels" / f"{pcd_file.stem}.label"
                gt_sem, _ = load_labels(label_file)

                pred_file = args.pred_dir / seq_path.name / f"{pcd_file.stem}.txt"
                #metrics_calculator.visulise_prediction_scores(points, np.loadtxt(pred_file).astype(np.float32), gt_sem)
                #metrics_calculator.visulise_GT(points, np.loadtxt(pred_file).astype(np.float32), gt_sem)
                metrics_calculator.update(
                    points, np.loadtxt(pred_file).astype(np.float32), gt_sem
                )

    metrics = metrics_calculator.compute_metrics()
    print(metrics)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=4, default=convert_to_builtin_types)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Detection Metrics")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default="detection_metrics.json")

    args = parser.parse_args()
    main(args)
