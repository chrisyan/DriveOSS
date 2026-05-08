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
<table>
  <thead>
    <tr>
      <th align="left">Weather Condition</th>
      <th align="center">OOD Validation Scenes</th>
      <th align="center">OOD Test Scenes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Day</strong></td>
      <td align="center">411, 412</td>
      <td align="center">413, 414</td>
    </tr>
    <tr>
      <td><strong>Night</strong></td>
      <td align="center">421, 422</td>
      <td align="center">423, 424</td>
    </tr>
    <tr>
      <td><strong>Rain</strong></td>
      <td align="center">441, 442</td>
      <td align="center">443, 444</td>
    </tr>
    <tr>
      <td><strong>Cloudy</strong></td>
      <td align="center">431, 432</td>
      <td align="center">433, 434</td>
    </tr>
    <tr>
      <td><strong>Snow</strong></td>
      <td align="center">451, 452</td>
      <td align="center">453, 454</td>
    </tr>
  </tbody>
</table>

#### Important Notes
<p>
  Annotations for the <strong>OOD test split are not provided</strong> in 
  order to ensure fair and unbiased evaluation. Furthermore, due to the 
  2.5 GB per-file size limit imposed by the Dataverse platform, files 
  exceeding this limit have been split into multiple zip archives.
</p>

## Dataset and Pre-trained Models Download

The full DriveOSS dataset, along with the pretrained benchmark models are uploaded to the [dataverse](https://doi.org/10.7910/DVN/SIEZCG) 

## Requirements

To install requirements:
As this paper proposes a new dataset and benchmark rather than a new method, we request the users to follow the instructions mentioned on the github  pages of the original works. We recommend creating separate virtual environments for each method. 

## Training Benchmark Models

To train the benchmark models presented in the paper, you can find the detailed instructions in this [file](./docs/train_baselines.md)

## Inference and Anomaly Scoring

To run inference and obtain anomaly scores on the benchmark models presented in the paper, you can find the detailed instructions in this [file](./docs/inference_on_baselines.md)

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
**Important Note** : The below script (```evaluate_multi_sequence_openset.py```) expects semantic segmentation output in the range 0-19. The models P3Former and Mask4Former (both variants) follows this output format. However, the LCPS model provides output in the range 1-20. In this case, an additional flag ```--pred-label-offset 1``` must be provided to the script below. 

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

2. For the results on DriveOSS-OOD test set, first run the above script but change just the sequence numbers. This will generate similar json file with the content mentioned above. However, only the closed-set and anomaly segmentation metrics should be used from this script. The open-set metrics generated in this json file uses the threshold from the test set itself. Hence, it must be discarded. To obtain the actual open-set evaluation result for a given anomaly method, copy the overall ```threshold_at_TPR95``` from the json file generated using the VAL set, and run the script again:
   
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

To obtain the weather specific evaluation results, please follow the same steps as above, but replace the sequences with specific weather scenes.

## Citation


