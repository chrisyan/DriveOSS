"""
Entry point for Mask4Former3D + Camera cross-attention training / validation / test.
Mirror of main_panoptic.py but uses:
    - PanopticSegmentationCamera  (trainer/pq_trainer_camera.py)
    - data=kitti3d_camera         (conf/data/kitti3d_camera.yaml)
    - model=mask4former3d_camera  (conf/model/mask4former3d_camera.yaml)
    - data/datasets=semantic_kitti_206_camera
    - data/collation_functions=voxelize_collate_camera

Example:
    python main_panoptic_camera.py general.mode=train general.gpus=2
"""
import logging
import os

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything

from trainer.pq_trainer_camera import PanopticSegmentationCamera
from utils.utils import flatten_dict, RegularCheckpointing


def get_parameters(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    load_dotenv(".env")

    seed_everything(cfg.general.seed)

    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    if not os.path.exists(cfg.general.save_dir):
        os.makedirs(cfg.general.save_dir)
    else:
        print("EXPERIMENT ALREADY EXISTS")
        cfg.general.ckpt_path = f"{cfg.general.save_dir}/last-epoch.ckpt"

    loggers = []
    for log in cfg.logging:
        print(log)
        loggers.append(hydra.utils.instantiate(log))

    model = PanopticSegmentationCamera(cfg)

    logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, loggers


def _make_trainer(cfg, loggers, callbacks=None):
    return Trainer(
        logger=loggers,
        accelerator="gpu",
        devices=cfg.general.gpus,
        strategy=(
            "ddp_find_unused_parameters_false" if cfg.general.gpus > 1 else "auto"
        ),
        callbacks=callbacks or [],
        default_root_dir=str(cfg.general.save_dir),
        **cfg.trainer,
    )


# ── default config: camera-specific overrides ─────────────────────────────────
@hydra.main(config_path="conf", config_name="config_panoptic_3d_camera.yaml")
def train(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    callbacks = [hydra.utils.instantiate(cb) for cb in cfg.callbacks]
    callbacks.append(RegularCheckpointing())
    runner = _make_trainer(cfg, loggers, callbacks)
    runner.fit(model, ckpt_path=cfg.general.ckpt_path)


@hydra.main(config_path="conf", config_name="config_panoptic_3d_camera.yaml")
def validate(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = _make_trainer(cfg, loggers)
    runner.validate(model=model, ckpt_path=cfg.general.ckpt_path)


@hydra.main(config_path="conf", config_name="config_panoptic_3d_camera.yaml")
def test(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = _make_trainer(cfg, loggers)
    runner.test(model=model, ckpt_path=cfg.general.ckpt_path)


@hydra.main(config_path="conf", config_name="config_panoptic_3d_camera.yaml")
def main(cfg: DictConfig):
    if cfg["general"]["mode"] == "train":
        train(cfg)
    elif cfg["general"]["mode"] == "validate":
        validate(cfg)
    else:
        test(cfg)


if __name__ == "__main__":
    main()
