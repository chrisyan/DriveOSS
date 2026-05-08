
For all the model, the inference script will run the inference and save the follwing ´´´.txt´´´ files for each frame in each sequence:
- Semantic segmentation predictions as files
- Anomaly score files for each anomaly functions mentioned in the paper.

#### P3Former

**Important Notes** : 
- By default, the predictions will be stored in the project root. If you
want to save it somewhere else, make sure to change the ```prediction_save_path``` variable in function ```predict```  inside file 
```./P3Former/p3former/segmentors/p3former.py```.
- make sure to change the paths in the following scripts according to your setup.
  
```bash
# single GPU
cd /path/to/DriveOSS/P3Former
python3 test.py "./configs/p3former/p3former_8xb2_3x_semantickitti_cudal_driveoss_submit.py" \
  "./work_dirs/p3former_8xb2_3x_semantickitti_cudal_driveoss/best.pth" \
  --work-dir "./work_dirs/p3former_8xb2_3x_semantickitti_cudal_driveoss/"

# multi GPU
torchrun     --nnodes=1     --nproc_per_node=2     --master_port=29500 \
    test.py "./configs/p3former/p3former_8xb2_3x_semantickitti_cudal_driveoss_submit.py" \
    "./work_dirs/p3former_8xb2_3x_semantickitti_cudal_driveoss/best.pth" \
    --launcher pytorch \
    --work-dir "./work_dirs/p3former_8xb2_3x_semantickitti_cudal_driveoss/"

```
#### LCPS

Before running inference, make sure to update the ```model_load_path``` in ```configs/pa_po_kitti_val_fusion.yaml```.
By default, the outputs wioll be stored in the ```./LCPS``` folder. To avoid this, change the ```prediction_save_path``` inside ```infer.py```.
```bash
cd /path/to/DriveOSS/LCPS
# single GPU
python infer.py \
-c configs/pa_po_kitti_val_fusion.yaml \
-l ./work_dir/consol.log \
--resume

# multi GPU
python -m torch.distributed.launch --nproc_per_node=2  infer.py \
-c configs/pa_po_kitti_val_fusion.yaml \
-l ./work_dir/consol.log \
--resume

```
#### Mask4Former

```bash
cd /path/to/DriveOSS/Mask4Former3D
python main_panoptic_all_methods.py \
  model=mask4former3d \
  data/datasets=semantic_kitti_206 \
  general.ckpt_path=path/to/checkpoint \
  general.mode=test

```

#### Mask4Former-Fusion
```bash
cd /path/to/DriveOSS/Mask4Former3D
python main_panoptic_camera_all_methods.py \
  model=mask4former3d_camera \
  data/datasets=semantic_kitti_206 \
  general.ckpt_path=path/to/checkpoint \
  general.mode=test

```
