
### pre-processing ###
python -m datasets.preprocessing.semantic_kitti_preprocessing preprocess --data_dir "data" --save_dir "./data"
python -m datasets.preprocessing.semantic_kitti_preprocessing make_instance_database --data_dir "data" --save_dir "./data"

### train lidar-mask4former ###
python main_panoptic.py model=mask4former3d data/datasets=semantic_kitti_206 general.mode=train general.gpus=2


### train fusion-mask4former ###
python main_panoptic_camera.py general.mode=train general.gpus=2

### inference lidar-mask4former ###
python main_panoptic_all_methods.py \
  model=mask4former3d \
  data/datasets=semantic_kitti_206 \
  general.ckpt_path=xxx/stu_dataset/Mask4Former3D/checkpoints_lidar/40p9pq.ckpt \
  general.mode=test

### inference fusion-mask4former  ###
python main_panoptic_camera_all_methods.py \
  model=mask4former3d_camera \
  data/datasets=semantic_kitti_206 \
  general.ckpt_path=xxx/stu_dataset/Mask4Former3D/checkpoints_fusion/41p2pq.ckpt \
  general.mode=test



  

