
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

def save_pred_logits(pred_logits, batch_idx):
    """Save raw logits to a text file with batch index"""
    os.makedirs('trainlogits_dump', exist_ok=True)
    filename = f'trainlogits_dump/batch_{batch_idx}_logits_1.txt'

    # Convert to numpy and save
    np.savetxt(filename, pred_logits.cpu().numpy().reshape(-1, pred_logits.shape[-1]))
    print(f"Saved raw logits to {filename}")
    
def save_pred_logits_sm(pred_logits, batch_idx):
    """Save raw logits to a text file with batch index"""
    os.makedirs('trainlogits_dump', exist_ok=True)
    filename = f'trainlogits_dump/batch_{batch_idx}_logits_sm_1.txt'

    # Convert to numpy and save
    np.savetxt(filename, pred_logits.cpu().numpy().reshape(-1, pred_logits.shape[-1]))
    print(f"Saved raw logit_sm to {filename}")

def save_pred_mask(pred_masks, batch_idx):
    """Save raw logits to a text file with batch index"""
    os.makedirs('trainlogits_dump', exist_ok=True)
    filename = f'trainlogits_dump/batch_{batch_idx}_masks_1.txt'

    if isinstance(pred_masks, list):
        # Case 1: pred_masks is a list of tensors
        mask_array = torch.cat([m.cpu() for m in pred_masks], dim=0).numpy()
    elif isinstance(pred_masks, torch.Tensor):
        # Case 2: pred_masks is already a tensor
        mask_array = pred_masks.cpu().numpy()
    else:
        raise ValueError(f"Unsupported mask type: {type(pred_masks)}")
    
    # Save with 4 decimal places
    np.savetxt(filename, mask_array, fmt='%.4f')
    print(f"Saved masks to {filename} (shape: {mask_array.shape})")

def save_remapping_data(batch_data, batch_idx):
    """Save inverse maps in .txt format for short-term visualization"""
    os.makedirs('trainlogits_dump', exist_ok=True)
        
    # Save inverse maps as .txt (same format as your other files)
    inverse_maps = batch_data.inverse_maps[0].cpu().numpy()
    np.savetxt(f'trainlogits_dump/batch_{batch_idx}_inverse_maps.txt', 
               inverse_maps, fmt='%d')  # Save as integers
        
    # Save raw coordinates (optional but recommended)
    raw_coords = batch_data.raw_coordinates.cpu().numpy()
    np.savetxt(f'trainlogits_dump/batch_{batch_idx}_raw_coords.txt', 
               raw_coords, fmt='%.6f')  # 6 decimal places for coordinates
        
    print(f"Saved remapping data for batch {batch_idx} (TXT format)")


# Initialize logging
logger = logging.getLogger(__name__)

config_path = os.path.join(project_root, "conf")

@hydra.main(config_path=config_path, config_name="config_panoptic_3d.yaml")
def get_dataloader(cfg: DictConfig):
    # Change back to original working directory (same as main_panoptic.py)
    os.chdir(hydra.utils.get_original_cwd())
    
    # Load model from checkpoint if provided and initialize the model with same config
    ckpt_path = cfg.general.ckpt_path
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading model from checkpoint_skl: {ckpt_path}")
        model = PanopticSegmentation.load_from_checkpoint(ckpt_path, config=cfg)
    else:
        print("No checkpoint found — using random weights!")
        model = PanopticSegmentation(cfg)

    # # Initialize model with the same config
    # model = PanopticSegmentation(cfg)
    
    model.prepare_data()  # This creates train_dataset
    model.setup(stage="fit")  # This prepares data loaders
    model.eval()
    #print("Dropout active:", any(m.training for m in model.modules() if isinstance(m, torch.nn.Dropout)))
    # Get the train dataloader
    train_loader = model.train_dataloader()

    # Statistics tracking
    frame_count = 0
    sequence_counter = defaultdict(int)
    total_points = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            print(f"Processing batch {batch_idx}")
            data, target = batch
            save_remapping_data(data, batch_idx)
            inverse_maps = data.inverse_maps
            raw_coordinates = data.raw_coordinates
            sequences = data.sequences
            print(f"Frames being processed in batch {batch_idx}: {sequences}")
            batch_size = len(sequences)

            # Debug print to check sequence format
            print(f"Sample sequence value: {sequences[0]} (type: {type(sequences[0])})")

            # Update counters - handle both integer and tuple/list sequences
            frame_count += batch_size
            for seq in sequences:
                if isinstance(seq, (list, tuple)):
                    seq_id = seq[0]
                else:
                    seq_id = seq  # Assume it's already a sequence ID
                sequence_counter[seq_id] += 1

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
            save_pred_logits(pred_logits, batch_idx)
            # #print("pred_logits:",pred_logits) #shape is (B,N-queries, N-classes)
            # pred_logits = torch.functional.F.softmax(pred_logits, dim=-1)[..., :-1]
            # #self.save_pred_logits_sm(pred_logits, batch_idx)
            # #print("pred_logits_sm:",pred_logits)
            pred_masks = output["pred_masks"]
            save_pred_mask(pred_masks, batch_idx)
            # #print("pred_masks:",pred_masks)
            # #print(f"Shape of pred_masks: ",pred_masks.shape)        

            
            #print(f"Data shape: {data.coordinates.shape}")
            #print(data.coordinates[:5].tolist())
            #print(f"Target is sequence with {len(target)} elements")
            #print("First few elements:", target[:2])

            if batch_idx >= 99:  # Just check first 3 batches
                break

    # Final statistics
    print("\nFinal Statistics:")
    print(f"Total frames processed: {frame_count}")
    print(f"Total sequences encountered: {len(sequence_counter)}")
    print(f"Points per frame average: {total_points/frame_count:.1f}")
    print("\nSequence distribution:")
    for seq_id, count in sequence_counter.items():
        print(f"Sequence {seq_id}: {count} frames")
            

if __name__ == "__main__":
    # You can override configs just like in CLI:
    # get_dataloader(general={"mode": "train"}, data={"datasets": "semantic_kitti_206"})
    get_dataloader()