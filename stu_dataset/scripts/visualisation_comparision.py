import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def load_point_cloud(scan_path):
    scan = np.fromfile(scan_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    points = scan[:, :3]
    return points

def load_labels(label_path):
    # Assuming labels are stored as uint32 per point
    labels = np.fromfile(label_path, dtype=np.uint32)
    return labels

def load_predictions(pred_path):
    # Assuming predictions are float scores, one per point
    preds = np.loadtxt(pred_path).astype(np.float32)
    return preds



def visualize_comparison(points, gt_labels, pred_scores, threshold=0.5):
    """
    Visualize point cloud with color coding for prediction correctness.
    Args:
      points: Nx3 np.ndarray
      gt_labels: Nx1 np.ndarray (0=inlier, 2=outlier)
      pred_scores: Nx1 np.ndarray (continuous anomaly scores)
      threshold: float, above which prediction is considered outlier
    """

    # Binarize predictions using threshold
    pred_labels = (pred_scores > threshold).astype(np.uint8) * 2  # 2 means predicted outlier
    
    colors = np.zeros((points.shape[0], 3))

    # True Negative (correct inlier) - Green
    tn = (gt_labels == 0) & (pred_labels == 0)
    colors[tn] = [0, 1, 0]

    # True Positive (correct outlier) - Red
    tp = (gt_labels == 2) & (pred_labels == 2)
    colors[tp] = [1, 0, 0]

    # False Positive (wrongly predicted outlier) - Yellow
    fp = (gt_labels == 0) & (pred_labels == 2)
    colors[fp] = [1, 1, 0]

    # False Negative (missed outlier) - Blue
    fn = (gt_labels == 2) & (pred_labels == 0)
    colors[fn] = [0, 0, 1]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])

def visualize_comparison_orignal_logic(points, gt_labels, pred_scores, threshold=0.5):
    pred_labels = (pred_scores > threshold).astype(np.uint8) * 2  # predicted outlier = 2

    colors = np.zeros((points.shape[0], 3))

    # Inliers: label != 0 and != 2
    inliers = (gt_labels != 0) & (gt_labels != 2)
    outliers = (gt_labels == 2)

    # True Negative (correct inlier) - Green
    tn = inliers & (pred_labels == 0)
    colors[tn] = [0, 1, 0]

    # True Positive (correct outlier) - Red
    tp = outliers & (pred_labels == 2)
    colors[tp] = [1, 0, 0]

    # False Positive (wrongly predicted outlier) - Yellow (actual an inlier but predicted outlier)
    fp = inliers & (pred_labels == 2)
    colors[fp] = [1, 1, 0]

    # False Negative (missed outlier) - Blue
    fn = outliers & (pred_labels == 0)
    colors[fn] = [0, 0, 1]

    # Ignore unlabeled (gt_labels == 0) points by coloring them black or removing if desired
    unlabeled = (gt_labels == 0)
    colors[unlabeled] = [0, 0, 0]  # or skip these points from visualization

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


def visualize_only_red(points, gt_labels, pred_scores, threshold=0.5):
    """
    Visualize only True Positive points (red), i.e. points where:
      gt_label == 2 (outlier)
      AND
      pred_score > threshold (predicted outlier)
    """

    # Binarize predictions
    pred_labels = (pred_scores > threshold).astype(np.uint8) * 2  # 2 means predicted outlier

    # True Positives mask
    tp_mask = (gt_labels == 2) & (pred_labels == 2)

    # Filter points to only True Positives
    tp_points = points[tp_mask]

    # Red color for all these points
    colors = np.tile([1, 0, 0], (tp_points.shape[0], 1))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tp_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])

def print_and_visualize_predicted_outliers(points, pred_scores, threshold=0.5):
    """
    Print info about predicted outliers and visualize only those points in red.

    Args:
        points (np.ndarray): Nx3 point cloud coordinates.
        pred_scores (np.ndarray): Nx1 anomaly scores.
        threshold (float): Threshold above which points are considered outliers.
    """
    pred_labels = (pred_scores > threshold).astype(np.uint8) * 2
    outlier_indices = np.where(pred_labels == 2)[0]

    print(f"Total points: {len(pred_scores)}")
    print(f"Number of predicted outliers (score > {threshold}): {len(outlier_indices)}")
    print(f"Indices of predicted outliers (first 20): {outlier_indices[:20]}")
    print(f"Predicted scores of first 20 outliers: {pred_scores[outlier_indices[:20]]}")

    if len(outlier_indices) == 0:
        print("No predicted outliers to visualize.")
        return

    outlier_points = points[outlier_indices]

    # Color all outlier points red
    colors = np.tile([1, 0, 0], (outlier_points.shape[0], 1))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(outlier_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], window_name="Predicted Outliers (Red)")

def visualize_outliers_with_gt(points, gt_labels, pred_scores, threshold=0.5):
    """
    Visualize predicted outliers (red) along with ground truth inliers (green).
    """
    pred_outlier_mask = pred_scores > threshold
    gt_inlier_mask = gt_labels == 0

    # Points predicted as outliers (red)
    outlier_points = points[pred_outlier_mask]
    outlier_colors = np.tile(np.array([[1, 0, 0]]), (outlier_points.shape[0], 1))  # Red

    # Ground truth inliers (green)
    gt_inlier_points = points[gt_inlier_mask]
    gt_inlier_colors = np.tile(np.array([[0, 1, 0]]), (gt_inlier_points.shape[0], 1))  # Green

    # Create Open3D point clouds
    pcd_outliers = o3d.geometry.PointCloud()
    pcd_outliers.points = o3d.utility.Vector3dVector(outlier_points)
    pcd_outliers.colors = o3d.utility.Vector3dVector(outlier_colors)

    pcd_inliers = o3d.geometry.PointCloud()
    pcd_inliers.points = o3d.utility.Vector3dVector(gt_inlier_points)
    pcd_inliers.colors = o3d.utility.Vector3dVector(gt_inlier_colors)

    # Visualize together
    o3d.visualization.draw_geometries([pcd_inliers, pcd_outliers])


if __name__ == "__main__":
    # Replace these paths with your actual data paths
    bin_path = ""
    label_path = ""
    pred_path = "" # .txt

    points = load_point_cloud(bin_path)
    gt_labels = load_labels(label_path)
    pred_scores = load_predictions(pred_path)

    print("GT unique labels:", np.unique(gt_labels))
    print("Pred scores min/max:", pred_scores.min(), pred_scores.max())

    #visualize_comparison(points, gt_labels, pred_scores, threshold=0.5)
    #visualize_only_red(points, gt_labels, pred_scores, threshold=0.5)
    #print_and_visualize_predicted_outliers(points, pred_scores, threshold=0.5)
    #visualize_outliers_with_gt(points, gt_labels, pred_scores, threshold=0.5)
    visualize_comparison_orignal_logic(points, gt_labels, pred_scores, threshold=0.5)

