#This is for train data
import os
from glob import glob

import numpy as np
import torch
import tqdm


def calculate_pointcloud_statistics(data_dir, num_classes=19, process_batches=100):
    """
    Calculate class-wise statistics for point cloud segmentation predictions
    matching the original image-based approach
    
    Args:
        data_dir: Directory containing prediction files
        num_classes: Number of semantic classes
        process_batches: Number of batches to process (matches original 50-batch approach)
    """
    pred_files = sorted(glob(os.path.join(data_dir, 'batch_*_logits_1.txt')))
    
    if not pred_files:
        raise ValueError(f"No prediction files found in {data_dir}")
    
    # Limit number of batches to process (like original)
    if process_batches > 0:
        pred_files = pred_files[:process_batches]
    
    # Initialize storage
    class_max_logits = [[] for _ in range(num_classes)]
    mean_dict = {}
    var_dict = {}
    
    print(f"Calculating statistics for first {len(pred_files)} batches...")
    
    for i, pred_file in enumerate(tqdm.tqdm(pred_files)):
        # Load prediction data
        base_name = pred_file.replace('_logits_', '')
        mask_file = pred_file.replace('_logits_', '_masks_')
        inv_map_file = pred_file.replace('_logits_1', '_inverse_maps')
        
        # # Load files - using .cpu() to ensure we're working on CPU like original
        # pred_logits = torch.load(pred_file).cpu()  # Shape: [100_queries, num_classes]
        # pred_masks = torch.load(mask_file).cpu()  # Shape: [num_masks, 100_queries]
        # inverse_map = torch.load(inv_map_file).cpu()  # Shape: [num_points]
        pred_logits = torch.from_numpy(np.loadtxt(pred_file)).float().cpu()
        pred_masks = torch.from_numpy(np.loadtxt(mask_file)).long().cpu()
        inverse_map = torch.from_numpy(np.loadtxt(inv_map_file)).long().cpu()
        #print("successful")
        
        # Get max logits and predictions (NO SOFTMAX - matching original)
        query_max_logits, query_preds = pred_logits.max(dim=1)  # Shapes: [100_queries]
        print(f"query_max_logits shape: {query_max_logits.shape}")  #torch.Size([100])
        print(f"Sample query max values: {query_max_logits[:5]}")  #tensor([ 8.5713,  7.6792,  5.5284, 11.7234,  5.4398])
        print(f"Sample predicted classes: {query_preds[:5]}")   #tensor([ 0,  5, 19,  0,  1])
        
        # Map query assignments to points using inverse_map
        mask_query_assignments = pred_masks.float().argmax(dim=1)  # [num_masks]
        print(f"mask_query_assignments shape: {mask_query_assignments.shape}")  #torch.Size([86116])
        print(f"Unique assigned queries: {torch.unique(mask_query_assignments)}") #tensor([ 2,  3,  6, 13, 14, 16, 18, 20, 22, 23, 25, 27, 28, 31, 33, 38, 42, 43, 46, 50, 51, 52, 58, 62, 63, 66, 67, 69, 70, 77, 78, 84, 96, 97])
        print(f"Assignment counts: {torch.bincount(mask_query_assignments)}")  # How many masks per query

        point_query_assignments = mask_query_assignments[inverse_map]
        print(f"point_query_assignments shape: {point_query_assignments.shape}")  
        print(f"Point assignments sample: {point_query_assignments[:10]}") #Point assignments sample: tensor([16, 16, 16, 16, 16, 16, 16, 16, 16, 16])  # Point assignments sample: tensor([75, 75, 75, 75, 75, 75, 75,  2, 75, 75])
        print(f"Assignment distribution:\n{torch.bincount(point_query_assignments)}") #567 masks got query 0


        # Expand query values to points
        point_max_logits = query_max_logits[point_query_assignments]  # [num_points]
        point_predictions = query_preds[point_query_assignments]  # [num_points]
        #print(f"point_max_logits shape: {point_max_logits.shape}")
        print(f"Sample point max logits: {point_max_logits[:100]}")  #tensor([5.6863, 5.6863, 5.6863, 5.6863, 5.6863, 5.6863, 5.6863, 5.6863, 5.6863, 5.6863])   #tensor([ 6.2352,  6.2352,  6.2352,  6.2352,  6.2352,  6.2352,  6.2352, 12.8370, 6.2352,  6.2352])
        print(f"Sample point classes: {point_predictions[:10]}")   #tensor([19, 19, 19, 19, 19, 19, 19, 19, 19, 19])  #tensor([ 2,  2,  2,  2,  2,  2,  2, 12,  2,  2])
        
        # Accumulate max logits per class
        for c in range(num_classes):
            class_mask = (point_predictions == c)
            if class_mask.any():
                class_max_logits[c].append(point_max_logits[class_mask])
        
        # Periodically process statistics (like original i%50 check)
        if i % 50 == 49 or i == len(pred_files) - 1:
        #if i == 149:
            # Compute current statistics
            temp_mean_dict, temp_var_dict = {}, {}
            for c in range(num_classes):
                if class_max_logits[c]:
                    all_logits = torch.cat(class_max_logits[c])
                    temp_mean_dict[c] = all_logits.mean().item()
                    temp_var_dict[c] = all_logits.var().item()
                else:
                    temp_mean_dict[c] = 0.0
                    temp_var_dict[c] = 0.0
            
            print(f"After batch {i}:")
            print(f"Class mean: {temp_mean_dict}")
            print(f"Class var: {temp_var_dict}")
    
    # Final statistics calculation
    for c in range(num_classes):
        if class_max_logits[c]:
            all_logits = torch.cat(class_max_logits[c])
            mean_dict[c] = all_logits.mean().item()
            var_dict[c] = all_logits.var().item()
        else:
            mean_dict[c] = 0.0
            var_dict[c] = 0.0
    
    print("\nFinal Statistics:")
    print(f"Class mean: {mean_dict}")
    print(f"Class var: {var_dict}")
    
    # Save statistics
    os.makedirs('stats', exist_ok=True)
    np.save('stats/pointcloud_mean.npy', mean_dict)
    np.save('stats/pointcloud_var.npy', var_dict)
    
    return mean_dict, var_dict


if __name__ == "__main__":
    data_directory = ""
    calculate_pointcloud_statistics(data_directory)