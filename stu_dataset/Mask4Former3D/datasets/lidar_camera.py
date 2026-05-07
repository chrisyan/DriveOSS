import numpy as np
import cv2
from pathlib import Path

from datasets.lidar import LidarDataset


class LidarCameraDataset(LidarDataset):
    """
    Extends LidarDataset to load the corresponding camera image (image_2/) and
    the full LiDAR→image projection matrix derived from calib.txt.

    Added keys in __getitem__ output:
        "image"      : np.ndarray (3, H, W) float32 — normalised RGB image
        "calib_proj" : np.ndarray (3, 4)   float32 — P2 @ [Tr | 0001]

    If an image file is missing the image is filled with zeros (graceful fallback).

    Args:
        img_size (tuple): (H, W) to resize images to. Default (192, 640).
        All other args/kwargs forwarded to LidarDataset.
    """

    IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, *args, img_size=(192, 640), **kwargs):
        super().__init__(*args, **kwargs)
        self.img_size   = img_size          # (H, W)
        self._calib_cache: dict = {}        # str(calib_path) → (3,4) array

    # ─────────────────────────────────────────────────────────────────────────
    def _load_calib_proj(self, calib_path: str) -> np.ndarray:
        """
        Parse calib.txt and return the combined 3×4 projection matrix
        proj = P2 @ Tr_4x4  that maps a LiDAR point [x,y,z,1]^T to image
        homogeneous coords.

        Result is cached per calib file.
        """
        if calib_path in self._calib_cache:
            return self._calib_cache[calib_path]

        calib: dict = {}
        with open(calib_path) as f:
            for line in f:
                if ":" in line:
                    key, val = line.split(":", 1)
                    calib[key.strip()] = np.fromstring(val, dtype=np.float64, sep=" ")

        P2 = calib["P2"].reshape(3, 4)
        Tr = calib["Tr"].reshape(3, 4)
        Tr_4x4 = np.vstack([Tr, [0.0, 0.0, 0.0, 1.0]])   # (4, 4)
        proj = (P2 @ Tr_4x4).astype(np.float32)            # (3, 4)

        self._calib_cache[calib_path] = proj
        return proj

    # ─────────────────────────────────────────────────────────────────────────
    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)

        # Use the last frame in the sweep for image / calibration
        last_scan  = self.data[idx][-1]
        velo_path  = Path(last_scan["filepath"])
        seq_root   = velo_path.parent.parent          # e.g. data/train/00/

        # ── Calibration ───────────────────────────────────────────────────
        calib_path = str(seq_root / "calib.txt")
        try:
            calib_proj = self._load_calib_proj(calib_path)   # (3, 4), original pixel space
        except Exception as e:
            print(f"Warning: calib load failed for {calib_path}: {e}")
            calib_proj = np.zeros((3, 4), dtype=np.float32)
            calib_proj[0, 0] = calib_proj[1, 1] = calib_proj[2, 2] = 1.0

        # ── Image ─────────────────────────────────────────────────────────
        img_path = seq_root / "image_2" / velo_path.with_suffix(".png").name
        img = np.zeros((3, *self.img_size), dtype=np.float32)
        if img_path.exists():
            try:
                raw = cv2.imread(str(img_path))
                if raw is not None:
                    H_orig, W_orig = raw.shape[:2]
                    # Scale calib_proj so it maps to resized image pixel space
                    scale_u = self.img_size[1] / W_orig
                    scale_v = self.img_size[0] / H_orig
                    S = np.array(
                        [[scale_u, 0, 0], [0, scale_v, 0], [0, 0, 1]],
                        dtype=np.float32,
                    )
                    calib_proj = S @ calib_proj   # (3, 4), resized pixel space
                    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                    raw = cv2.resize(raw, (self.img_size[1], self.img_size[0]))
                    raw = raw.astype(np.float32) / 255.0
                    raw = (raw - self.IMG_MEAN) / self.IMG_STD
                    img = raw.transpose(2, 0, 1)             # HWC → CHW  (3, H, W)
            except Exception as e:
                print(f"Warning: image load failed for {img_path}: {e}")

        sample["image"]      = img
        sample["calib_proj"] = calib_proj
        # Override sequence to match the (seq_name, frame_stem) tuple format
        # used by lidar_no_preprocessing.LidarDataset, which the trainer expects.
        # seq[0] = sequence directory name, seq[1] = frame stem (for saving predictions)
        sample["sequence"] = (seq_root.name, velo_path.stem)
        return sample
