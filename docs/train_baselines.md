## Training

First download all the necessary datasets from the official sources. These are : 1) SemanticKITTI, 2) Panoptic-CUDAL, and 3) DriveOSS.

To train the benchmark model(s) in the paper, run this command:

#### P3Former

Training p3former model requires info.pkl files. You can generate these files by running the command:

```bash
cd /path/to/DriveOSS/P3Former
python tools/create_data.py semantickitti --root-path "/path/to/semantickitti/dataset" \
      --out-dir "path/to/semantickitti/dataset" --extra-tag semantickitti
```
For more information, have a look at the original p3former [codebase](https://github.com/InternRobotics/P3Former)

The information in the .pkl files is dictated by the file ```./P3former/tools/dataset_converters/semantickkitti_converter.py```.
Ensure that correct sequences are places in according to the split in ```fold_split``` dictionary.

You will also need to generate these .pkl files for the DriveOSS-OOD val and test set for evaluation purposes. In this case, simply put the sequence numbers in the ´´´test´´´ key of the ```fold_split``` dictionary. Once the file is generated, rename it.  

Furthermore, change the dataset_root in  ```./P3Former/configs/_base_/datasets/semantickitti_panoptic_lpmix.py``` file.
```bash
# single GPU
cd /path/to/DriveOSS/P3Former
python3 train.py "./configs/p3former/p3former_8xb2_3x_semantickitti_cudal_driveoss.py"
# to resume, provide --checkpoint "/path/to/checkpoint" and --resume flags.

# the dist_train.sh files take config path as first positional arg, and number of gpus as the second arg
bash dist_train.sh "./P3former//configs/p3former/p3former_8xb2_3x_semantickitti_cudal_driveoss.py" 2
```
#### LCPS
Prepare the dataset according to the instructions provided in the original LCPS [codebase](https://github.com/zhangzw12319/lcps).
Note that we do not use the ```instance augmentation``` feature. Hence, the instance .pkl files are not required.

Before training, edit the ```pa_po_kitti_cudal_driveosss_trainval.yaml``` config file. Then run:

```bash
cd /path/to/DriveOSS/LCPS

#  single GPU
python train.py \
-c configs/pa_po_kitti_cudal_driveosss_trainval.yaml \
-l ./work_dir/train.log

# multi GPU
torchrun --nproc_per_node=2 --master_port=29503 train.py -c configs/pa_po_kitti_cudal_driveosss_trainval.yaml -l ./work_dir/train.log

```
#### Mask4Former

For both variants of Mask4Former, we follow the STU dataset [codebase](https://github.com/kumuji/stu_dataset).

```bash
cd /path/to/DriveOSS/stu_dataset/Mask4Former
# preprocessing
python -m datasets.preprocessing.semantic_kitti_preprocessing preprocess --data_dir "data" --save_dir "./data"
python -m datasets.preprocessing.semantic_kitti_preprocessing make_instance_database --data_dir "data" --save_dir "./data"

# train
python main_panoptic.py model=mask4former3d data/datasets=semantic_kitti_206 general.mode=train general.gpus=2
```

#### Mask4Former-Fusion
```bash
cd /path/to/DriveOSS/stu_dataset/Mask4Former
# preprocessing
python -m datasets.preprocessing.semantic_kitti_preprocessing preprocess --data_dir "data" --save_dir "./data"
python -m datasets.preprocessing.semantic_kitti_preprocessing make_instance_database --data_dir "data" --save_dir "./data"

# train
python main_panoptic.py model=mask4former3d data/datasets=semantic_kitti_206 general.mode=train general.gpus=2
```
