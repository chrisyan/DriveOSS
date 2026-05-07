
import sys
project_root = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D"
sys.path.append(project_root)

import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from pq_trainer import PanopticSegmentation
from utils.utils import flatten_dict
import torch
import MinkowskiEngine as ME
from collections import defaultdict
import numpy as np
from pathlib import Path

# Initialize logging
logger = logging.getLogger(__name__)

config_path = os.path.join(project_root, "conf")

@hydra.main(config_path=config_path, config_name="config_panoptic_3d.yaml")
def get_dataloader(cfg: DictConfig):
    # Change back to original working directory (same as main_panoptic.py)
    os.chdir(hydra.utils.get_original_cwd())

    #checkpoint_path = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/checkpoints/Ensemble_model_1.ckpt"
    
    # Load model from checkpoint if provided and initialize the model with same config
    ckpt_path = cfg.general.ckpt_path
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading model from checkpoint_skl: {ckpt_path}")
        model = PanopticSegmentation.load_from_checkpoint(ckpt_path, config=cfg)
    else:
        print("No checkpoint found — using random weights!")
        model = PanopticSegmentation(cfg)
    
    model.prepare_data()  # This creates train_dataset
    model.setup(stage="fit")  # This prepares data loaders
    model.eval()
    #print("Dropout active:", any(m.training for m in model.modules() if isinstance(m, torch.nn.Dropout)))
    # Get the train dataloader
    train_loader = model.train_dataloader()

    # Initialize containers for statistics
    pred_logits_list = []
    print(f"\nCollected_intermediate {len(pred_logits_list)} predictions for stats.")

    pred_masks_list = []
    inv_maps_list = []

    # Statistics tracking
    frame_count = 0
    sequence_counter = defaultdict(int)
    sequence_frame_counters = defaultdict(int)  # Initialize at script start
    total_points = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            print(f"Processing batch {batch_idx}")
            data, target = batch
            inverse_maps = data.inverse_maps 
            raw_coordinates = data.raw_coordinates
            sequences = data.sequences
            batch_size = len(sequences)

            frame_ids = data.frame_id  # new

            for seq_id, frame_id in zip(sequences, frame_ids):
                print(f"Processing Sequence {seq_id}, Frame {frame_id}")

            # # In your batch loop:
            # print("\nTRUE FRAME ACCESS:")
            # try:
            #     scan = getattr(data, 'sequences', [{}])[0]  # Safely get first scan
            #     if isinstance(scan, dict) and 'filepath' in scan:
            #         frame_path = Path(scan["filepath"])
            #         seq_id = frame_path.parent.parent.name  # Keep as string (00)
            #         frame_num = frame_path.stem            # Keep as string (000123)
            #         print(f"  • Seq {seq_id} | Frame {frame_num} | File: {frame_path}")
            #     else:
            #         print("  • No filepath found - using fallback numbering")
            #         print(f"  • Seq {data.sequences[0]:02d} | Frame {batch_idx:06d}")
            # except Exception as e:
            #     print(f"  • Frame access error: {str(e)}")

            # # In your batch loop:
            # print("\nTRUE FRAME ACCESS:")
            # try:
            #     # Get the current batch's dataset indices
            #     if hasattr(train_loader.sampler, 'indices'):
            #         dataset_indices = train_loader.sampler.indices
            #     else:
            #         # For sequential sampling
            #         dataset_indices = range(len(train_loader.dataset))
                
            #     # Get the dataset index for this batch
            #     dataset_idx = dataset_indices[batch_idx * train_loader.batch_size]
                
            #     # Access the original dataset
            #     sample = train_loader.dataset[dataset_idx]
                
            #     # Extract frame info
            #     if 'filepath' in sample:
            #         frame_path = Path(sample['filepath'])
            #         print(f"  • Seq {frame_path.parent.parent.name} | Frame {frame_path.stem}")
            #     elif isinstance(sample.get('sequence'), tuple):
            #         seq, frame = sample['sequence']
            #         print(f"  • Seq {seq} | Frame {frame}")
            #     else:
            #         print("  • No frame data available")
            # except Exception as e:
            #     print(f"  • Error: {str(e)}")


            # Update counters - handle both integer and tuple/list sequences
            frame_count += batch_size
            for seq in sequences:
                if isinstance(seq, (list, tuple)):
                    seq_id = seq[0]
                else:
                    seq_id = seq  # Assume it's already a sequence ID
                sequence_counter[seq_id] += 1
            


            # # Inside batch loop:
            # for seq_id in set(sequences):
            #     current_frame = sequence_frame_counters[seq_id]
            #     frames_in_batch = sum(1 for s in sequences if s == seq_id)
                
            #     print(f"Sequence {seq_id:02d}: Frame {current_frame:06d}")
            #     print(f"  File: {seq_id:02d}/velodyne/{current_frame:06d}.bin")
                
            #     # Update counter by actual frames processed
            #     sequence_frame_counters[seq_id] += frames_in_batch

            # Count points in this batch
            total_points += sum(len(inv_map) for inv_map in data.inverse_maps)
            
            print(f"\nBatch {batch_idx} processing:")
            print(f"Frames in this batch: {batch_size}")
            print(f"Current sequences: {set(sequence_counter.keys())}")
            print(f"Total frames processed: {frame_count}")
            print(f"Total points processed: {total_points}")


            data = ME.SparseTensor(
                coordinates=data.coordinates, features=data.features, device=model.device
            )
            output = model(data, raw_coordinates=raw_coordinates, is_eval=True)
            #print("output:",output)
            print(f"[DEBUG] Predicted classes (from logits): {output['pred_logits'].argmax(dim=-1).unique()}")

            pred_logits = output["pred_logits"] #Accessing via keys so output is dictionary
            # #self.save_pred_logits(pred_logits, batch_idx)
            # #print("pred_logits:",pred_logits) #shape is (B,N-queries, N-classes)
            # pred_logits = torch.functional.F.softmax(pred_logits, dim=-1)[..., :-1]
            # #self.save_pred_logits_sm(pred_logits, batch_idx)
            # #print("pred_logits_sm:",pred_logits)
            pred_masks = output["pred_masks"]
            # #self.save_pred_mask(pred_masks, batch_idx)
            # #print("pred_masks:",pred_masks)
            #print(f"Shape of pred_masks: ",pred_masks.shape)        

            # Append to lists
            pred_logits_list.append(pred_logits[0].cpu())
            pred_masks_list.append(pred_masks[0].cpu())
            inv_maps_list.append(inverse_maps[0].cpu())
            # for i in range(len(pred_logits)):
            #     pred_logits_list.append(pred_logits[i].cpu())
            #     pred_masks_list.append(pred_masks[i].cpu())
            #     inv_maps_list.append(inverse_maps[i].cpu())
            
            #print(f"Data shape: {data.coordinates.shape}")
            #print(data.coordinates[:5].tolist())
            #print(f"Target is sequence with {len(target)} elements")
            #print("First few elements:", target[:2])

            if batch_idx >= 9:  # Just check first 3 batches
                break

    print(f"\nCollected {len(pred_logits_list)} predictions for stats.")

    # Final statistics
    print("\nFinal Statistics:")
    print(f"Total frames processed: {frame_count}")
    print(f"Total sequences encountered: {len(sequence_counter)}")
    print(f"Points per frame average: {total_points/frame_count:.1f}")
    print("\nSequence distribution:")
    for seq_id, count in sequence_counter.items():
        print(f"Sequence {seq_id}: {count} frames")

    mean_dict, var_dict = calculate_pointcloud_statistics_from_data(
    pred_logits_list, pred_masks_list, inv_maps_list, num_classes=19
    )

def calculate_pointcloud_statistics_from_data(pred_logits_list, pred_masks_list, inv_maps_list, num_classes=19):
    """
    Compute class-wise max-logit statistics directly from in-memory tensors.
    
    Args:
        pred_logits_list: List of Tensors [num_queries, num_classes]
        pred_masks_list:  List of Tensors [num_masks, num_queries]
        inv_maps_list:    List of Tensors [num_points]
        num_classes: Number of semantic classes
    """
    assert len(pred_logits_list) == len(pred_masks_list) == len(inv_maps_list), "Mismatch in batch lengths"

    class_max_logits = [[] for _ in range(num_classes)]
    mean_dict = {}
    var_dict = {}

    print(f"Processing {len(pred_logits_list)} batches...")

    for i, (logits, masks, inv_map) in enumerate(zip(pred_logits_list, pred_masks_list, inv_maps_list)):
        logits = logits.float().cpu()
        masks = masks.long().cpu()
        inv_map = inv_map.long().cpu()

        query_max_logits, query_preds = logits.max(dim=1)  # [num_queries]

        mask_query_assignments = masks.float().argmax(dim=1)  # [num_masks]
        point_query_assignments = mask_query_assignments[inv_map]  # [num_points]

        point_max_logits = query_max_logits[point_query_assignments]  # [num_points]
        point_predictions = query_preds[point_query_assignments]      # [num_points]

        for c in range(num_classes):
            class_mask = (point_predictions == c)
            #print(f"Class {c}: mask sum = {(point_predictions == c).sum().item()}")
            if class_mask.any():
                class_max_logits[c].append(point_max_logits[class_mask])

        # Optional: progress print
        if i % 10 == 0:
            print(f"Processed batch {i}/{len(pred_logits_list)}")

    # Final aggregation
    for c in range(num_classes):
        if class_max_logits[c]:
            all_logits = torch.cat(class_max_logits[c])
            mean_dict[c] = all_logits.mean().item()
            var_dict[c] = all_logits.var().item()
        else:
            mean_dict[c] = 0.0
            var_dict[c] = 0.0

    print("\nFinal Statistics:")
    print(f"Class means: {mean_dict}")
    print(f"Class vars:  {var_dict}")

    # Optionally save
    os.makedirs('stats', exist_ok=True)
    np.save('stats/pointcloud_mean.npy', mean_dict)
    np.save('stats/pointcloud_var.npy', var_dict)

    return mean_dict, var_dict


if __name__ == "__main__":
    # You can override configs just like in CLI:
    # get_dataloader(general={"mode": "train"}, data={"datasets": "semantic_kitti_206"})
    get_dataloader()