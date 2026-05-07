# DriveOSS: Dataset and Benchmark for 3D Open-set Segmentation under Adverse Weather in Autonomous Driving

This repository is the official implementation of DriveOSS: Dataset and Benchmark for 3D Open-set Segmentation under Adverse Weather in Autonomous Driving. 



<p align="center">
<img  src="assets/figure_1.png" width="90%" />
</p>

</div>


<p align="center">
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

## Requirements

To install requirements:

```setup
As this paper proposes a new dataset and benchmark rather than a new method, we request the users to follow the instructions mentioned on the github  pages of the original works. We recommend creating separate virtual environments for each method. 
```

>📋 Dataset and model weights are hosted on the harvard dataverse

## Training

To train the benchmark model(s) in the paper, run this command:

#### P3Former
```

```
#### LCPS
```

```
#### Mask4Former
```

```

#### Mask4Former-Fusion
```
```

## Closed-Set Evaluation

#### P3Former
```

```
#### LCPS
```

```
#### Mask4Former
```

```

#### Mask4Former-Fusion
```
```

## Open-Set Evaluation


#### P3Former
```

```
#### LCPS
```

```
#### Mask4Former
```

```

#### Mask4Former-Fusion
```
```

## Pre-trained Models

You can download pretrained models here:

-  TODO(link). 

## Results

Open-set evaluation results on the DriveOSS-OOD test set:


## Citation

