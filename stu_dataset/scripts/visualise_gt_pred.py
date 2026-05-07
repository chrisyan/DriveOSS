import glob
import os

import matplotlib.pyplot as plt
import numpy as np

pred_dir = './pred/prediction'  # update if needed
gt_dir = './data/stu_dataset/val'  # or wherever your ground truth .txt files are

in_scores = []
ood_scores = []

pred_files = sorted(glob.glob(os.path.join(pred_dir, "*", "*.txt")))  # Recursively get all prediction txts

for pred_path in pred_files:
    # Example pred_path: pred/prediction/125/000000.txt
    # Extract scene and file name
    scene_id = os.path.basename(os.path.dirname(pred_path))  # "125"
    file_name = os.path.basename(pred_path)  # "000000.txt"

    # Construct matching GT path
    gt_path = os.path.join(gt_dir, scene_id, file_name)

    if not os.path.exists(gt_path):
        print(f"Missing ground truth for {scene_id}/{file_name}")
        continue

    pred_scores = np.loadtxt(pred_path)
    gt_labels = np.loadtxt(gt_path)  # Assume 0: in, 1: ood

    if len(pred_scores) != len(gt_labels):
        print(f"Length mismatch in {scene_id}/{file_name}: pred={len(pred_scores)}, gt={len(gt_labels)}")
        continue

    in_scores.extend(pred_scores[gt_labels == 0])
    ood_scores.extend(pred_scores[gt_labels == 1])

print("Number of in-distribution scores:", len(in_scores))
print("Number of OOD scores:", len(ood_scores))

# Plotting
plt.hist(in_scores, bins=50, alpha=0.6, label='In-distribution', color='blue', density=True)
plt.hist(ood_scores, bins=50, alpha=0.6, label='Out-of-distribution', color='red', density=True)
plt.xlabel('Prediction Score')
plt.ylabel('Density')
plt.legend()
plt.title('Score Distribution: In vs OOD')
plt.grid(True)
plt.show()
