# DriveOSS: Dataset and Benchmark for 3D Open-set Segmentation under Adverse Weather in Autonomous Driving

This repository is the official implementation of DriveOSS: Dataset and Benchmark for 3D Open-set Segmentation under Adverse Weather in Autonomous Driving. 



<p align="center">
<img  src="assets/figure_1.png" width="90%" />
</p>

</div>


<!-- <p align="center">
<img src="assets/vis2.png" width="90%" />
</p>
<p align="center">
<em>
Point-cloud-to-image projection under different weather conditions. 
From left to right, the columns show clear, night, cloudy, rainy, 
and snowy conditions. From top to bottom, the rows correspond to 
DriveOSS-ID, DriveOSS-OOD validation, and DriveOSS-OOD test scenes. 
The projected points reflect weather-dependent LiDAR return 
distributions, with notably fewer road-surface reflections observed 
under rainy conditions.
</em>
</p>
 -->


**DriveOSS** is a multimodal dataset for 3D open-set semantic segmentation under adverse weather conditions in autonomous driving. It contains synchronized surround-view RGB images, 128-beam LiDAR point clouds, and GNSS/IMU measurements, collected using a custom vehicle platform equipped with six cameras and an Ouster OS1-128 LiDAR sensor.

#### Dataset Organization

  The dataset is organized into two subsets:
</p>
<ul>
  <li>
    <strong>In-distribution (ID) subset:</strong> Contains normal driving 
    scenarios with 27 semantic classes. Used for closed-set model training 
    and validation.
  </li>
  <li>
    <strong>Out-of-distribution (OOD) subset:</strong> Contains the same 
    27 semantic classes plus an additional <em>outlier</em> class 
    representing unknown road obstacles deliberately placed in controlled 
    environments. Further divided into validation and test splits for 
    open-set evaluation.
  </li>
</ul>

#### Data Collection
<p>
  Data was collected across public roads and industrial areas under diverse 
  weather and lighting conditions, including clear, night, cloudy, rain, 
  and snow. Point-wise semantic annotations were produced using an automatic 
  annotation and human refinement workflow, with independent annotator groups 
  for labeling and quality inspection. All sensitive information, including 
  license plates and human faces, has been anonymized prior to release.
</p>

#### Dataset Statistics 
<p>In total, DriveOSS contains:</p>
<ul>
  <li><strong>8,309</strong> LiDAR frames</li>
  <li><strong>49,854</strong> camera images from six surround-view cameras</li>
  <li><strong>8,309</strong> synchronized GNSS/IMU measurements</li>
</ul>

#### Important Notes
<p>
  Annotations for the <strong>OOD test split are not provided</strong> in 
  order to ensure fair and unbiased evaluation. Furthermore, due to the 
  2.5 GB per-file size limit imposed by the Dataverse platform, files 
  exceeding this limit have been split into multiple zip archives.
</p>

---


## Dataset and Pre-trained Models Download

The full DriveOSS dataset, along with the pretrained benchmark models are uploaded to the [dataverse](https://doi.org/10.7910/DVN/SIEZCG) 

## Requirements

To install requirements:
As this paper proposes a new dataset and benchmark rather than a new method, we request the users to follow the instructions mentioned on the github  pages of the original works. We recommend creating separate virtual environments for each method. 

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

## Inference and Anomaly Scoring

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
```
cd /path/to/DriveOSS/stu_dataset/Mask4Former
python main_panoptic_all_methods.py \
  model=mask4former3d \
  data/datasets=semantic_kitti_206 \
  general.ckpt_path=path/to/checkpoint \
  general.mode=test

```

#### Mask4Former-Fusion
```
cd /path/to/DriveOSS/stu_dataset/Mask4Former
python main_panoptic_camera_all_methods.py \
  model=mask4former3d_camera \
  data/datasets=semantic_kitti_206 \
  general.ckpt_path=path/to/checkpoint \
  general.mode=test

```

## Open-Set Evaluation

After the inference you should have the following folder structure. 

```tree
outputs
|── msp/
|   ├── msp_000000.txt
|   |   .
|   ├── msp_000XXX.txt
|── energy/
|   ├── energy_000000.txt
|   |   .
|   ├── energy_000XXX.txt
|── entropy/
|   ├── entropy_000000.txt
|   |   .
|   ├── entropy_000XXX.txt
|── maxlogit/
|   ├── maxlogit_000000.txt
|   |   .
|   ├── maxlogit_000XXX.txt
|── predictions/
|   ├── 000000.txt
|   |   .
|   ├── 000XXX.txt
```
Now, copy all .txt files from the ```predictions``` folder to each of the anomaly scoring method folders. Each anomaly folder should contain anomaly scores for the corresponding method as well as the semantic predictions.  
```tree
outputs
|── msp/
|   ├── msp_000000.txt
|   ├── 000000.txt
```

The steps for evaluation: 
1. first run the evaluation on the DriveOSS-OOD val set to obtain the threshold for open-iou calculation. For this purpose, use the evaluation script for each anomaly method as follows:
```bash
cd /path/to/DriveOSS
# this must be run for each anomaly method, keep in mind to change the output path and anomaly prefix flags.
python evaluate_multi_sequence_openset.py \
  --data-root "data/semantickitti/dataset/sequences" \
  --pred-root "outputs/" \
  --yaml "./semantic-kitti.yaml" \
  --sequences 411 421 431 441 451 412 422 432 442 452 \
  --output results_10seq_val_energy.json \
  --closed-prefix "" \ 
  --anomaly-prefix energy_ 
```
This script will generate a json file with per-sequence as well as overall anomaly segmentation results (AUROC, FPR@95, AP), threshold for segmentation, closed set segmentation evaluation metrics (per class iou and global miou), and open-set results.

2. For the results on DriveOSS-OOD test set, first run the above script but change just The sequence numbers. This will generate similar json file with the content mentioned above. However, only the closed-set and anomaly segmentation metrics should be used from this script. The open-set metrics generated in this json file uses the threshold from the test set itself. Hence, it must be discarded. To obtain the actual open-set evaluation result for a given anomaly score, copy the overall ```threshold_at_TPR95``` from the json file generated using the VAL set, and run the script again:
   
```bash
cd /path/to/DriveOSS
python evaluate_multi_sequence_openset.py \
  --data-root "data/semantickitti/dataset/sequences" \
  --pred-root "outputs/" \
  --yaml "./semantic-kitti.yaml" \
  --sequences 413 414 423 424 433 434 443 444 453 454 \
  --output results_10seq_test_energy.json \
  --fixed-threshold <VAL_threshold_at_TPR95> \
  --closed-prefix ""\ 
  --anomaly-prefix energy_ \ 
```
**Important Note** : The above script expects semantic segmentation output in the range 0-19. The models P3Former and Mask4Former (both variants) follows this output format. However, the LCPS model provides output in the range 1-20. In this case, an additional flag ```--pred-label-offset 1``` must be provided to the script above.  
 

## Results

Open-set evaluation results on the DriveOSS-OOD test set:


## Citation

