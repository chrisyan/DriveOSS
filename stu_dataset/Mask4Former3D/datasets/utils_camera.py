import MinkowskiEngine as ME
import numpy as np
import torch

from datasets.utils import VoxelizeCollate, voxelize, generate_target, NoGpu


class NoGpuCamera(NoGpu):
    """
    Extends NoGpu to carry the batched camera images and projection matrices.
    Both tensors stay on CPU (Lightning moves them to GPU during transfer_batch_to_device).

    Extra attributes:
        images      : torch.Tensor [B, 3, H, W]  — normalised RGB images
        calib_projs : torch.Tensor [B, 3, 4]     — LiDAR→image projection matrices
    """

    def __init__(self, *args, images=None, calib_projs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.images      = images       # [B, 3, H, W]
        self.calib_projs = calib_projs  # [B, 3, 4]


class VoxelizeCollateCamera(VoxelizeCollate):
    """
    Extends VoxelizeCollate to also batch camera images and calibration matrices.
    Samples that lack an "image" key (e.g. test sequences without images) receive
    a zero-filled image so the batch is always homogeneous.
    """

    def __call__(self, batch):
        images, calib_projs = [], []
        (
            coordinates, features, labels,
            original_labels, inverse_maps, num_points, sequences,
        ) = [], [], [], [], [], [], []

        for sample in batch:
            original_labels.append(sample["labels"])
            num_points.append(sample["num_points"])
            sequences.append(sample["sequence"])

            sample_c, sample_f, sample_l, inverse_map = voxelize(
                sample["coordinates"], sample["features"],
                sample["labels"], self.voxel_size,
            )
            inverse_maps.append(inverse_map)
            coordinates.append(sample_c)
            features.append(sample_f)
            labels.append(sample_l)

            img = sample.get("image")
            if img is None:
                # fallback: infer H×W from first good sample or use default
                img = np.zeros((3, 192, 640), dtype=np.float32)
            images.append(torch.from_numpy(img))                        # (3, H, W)
            cp = sample.get("calib_proj")
            if cp is None:
                cp = np.zeros((3, 4), dtype=np.float32)
            calib_projs.append(torch.from_numpy(cp))                    # (3, 4)

        target = generate_target(features, labels, self.ignore_label)
        coordinates, features = ME.utils.sparse_collate(coordinates, features)
        raw_coordinates = features[:, :4]
        features        = features[:, 4:]

        return (
            NoGpuCamera(
                coordinates, features, raw_coordinates,
                original_labels, inverse_maps, num_points, sequences,
                images=torch.stack(images),           # [B, 3, H, W]
                calib_projs=torch.stack(calib_projs), # [B, 3, 4]
            ),
            target,
        )
