#This script is used to see prediction result for single frame in o3d. blue means inliers and red means outliers
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from utils.common import load_labels, load_point_cloud


def visualize_open3d(points, labels=None, scores=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Coloring by labels or scores
    if scores is not None:
        # Normalize scores to 0-1 and map to colormap
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
        colors = plt.cm.jet(norm_scores)[:, :3]
    elif labels is not None:
        # Color inliers and outliers differently
        colors = np.zeros_like(points)
        colors[labels == 0] = [0, 0, 1]    # Blue for inliers
        colors[labels == 2] = [1, 0, 0]    # Red for outliers
    else:
        colors = np.ones_like(points) * 0.5

    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

# Example usage
if __name__ == "__main__":
    # Change this to your actual data
    bin_path = ""
    label_path = ""
    pred_path = "" # .txt

    points, _ = load_point_cloud(bin_path)
    labels, _ = load_labels(label_path)
    scores = np.loadtxt(pred_path).astype(np.float32)

    visualize_open3d(points, labels=labels, scores=scores)
