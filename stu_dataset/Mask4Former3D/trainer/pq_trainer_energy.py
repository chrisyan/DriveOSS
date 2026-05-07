import statistics
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import hydra
import MinkowskiEngine as ME
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.cluster import DBSCAN

from utils.utils import associate_instances
import torch
import torch.nn.functional as F
import time


class PanopticSegmentation(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.save_hyperparameters()
        # model
        self.model = hydra.utils.instantiate(config.model)
        self.optional_freeze = nullcontext

        matcher = hydra.utils.instantiate(config.matcher)
        weight_dict = {
            "loss_ce": matcher.cost_class,
            "loss_mask": matcher.cost_mask,
            "loss_dice": matcher.cost_dice,
            "loss_box": matcher.cost_box,
        }

        aux_weight_dict = {}
        for i in range(self.model.num_levels * self.model.num_decoders):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

        self.criterion = hydra.utils.instantiate(
            config.loss, matcher=matcher, weight_dict=weight_dict
        )
        # metrics
        self.class_evaluator = hydra.utils.instantiate(config.metric)
        self.last_seq = None
                                                            # 记录每一帧的时间（单位：ms）
        self.test_inference_times = []
        self.test_energy_times = []

    def forward(self, x, raw_coordinates=None, is_eval=False):
        with self.optional_freeze():
            x = self.model(x, raw_coordinates=raw_coordinates, is_eval=is_eval)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch

        raw_coordinates = data.raw_coordinates
        data = ME.SparseTensor(
            coordinates=data.coordinates, features=data.features, device=self.device
        )

        output = self.forward(data, raw_coordinates=raw_coordinates)
        losses = self.criterion(output, target)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        logs = {f"train_{k}": v.detach().cpu().item() for k, v in losses.items()}

        logs["train_mean_loss_ce"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_ce" in k]]
        )

        logs["train_mean_loss_mask"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_mask" in k]]
        )

        logs["train_mean_loss_dice"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_dice" in k]]
        )

        logs["train_mean_loss_box"] = statistics.mean(
            [item for item in [v for k, v in logs.items() if "loss_box" in k]]
        )

        self.log_dict(logs)
        return sum(losses.values())

    def test_step(self, batch, batch_idx):
        data, target = batch
        inverse_maps = data.inverse_maps
        # original_labels = data.original_labels
        raw_coordinates = data.raw_coordinates
        # num_points = data.num_points
        sequences = data.sequences
        start_inference = time.time()
        data = ME.SparseTensor(
            coordinates=data.coordinates, features=data.features, device=self.device
        )
        output = self.forward(data, raw_coordinates=raw_coordinates, is_eval=True)
        inference_time = (time.time() - start_inference) * 1000



        def get_energy(logit, mask, inv_map, T: float = 1.0,
                    normalize: str = "minmax",  # "none" | "minmax" | "quantile" | "sigmoid"
                    q_low: float = 0.01, q_high: float = 0.99, eps: float = 1e-6):
            """
            Energy-based OOD score per point.
            Args:
                logit:  [Q, C]  query-level class logits
                mask:   [N, Q]  point-query mask logits or probs
                inv_map:[N']    indices to restore original order
                T: temperature for energy (default 1.0)
                normalize: 
                    - "none":    返回原始 energy（更高=更异常，通常为更接近0的负数或正数）
                    - "minmax":  线性缩放到[0,1]，更高=更异常
                    - "quantile":分位数鲁棒归一化到[0,1]
                    - "sigmoid": z-score 后过 sigmoid 到[0,1]
                q_low/q_high: 分位数归一化的上下界
                eps: 数值稳定项
            Returns:
                energy_score: [N']  越大越异常的分数（若 normalize="none" 则为原始能量值）
            """
            print("get_energy called")
            print("logit shape:", logit.shape, "min/max:", logit.min().item(), logit.max().item())
            print("mask  shape:", mask.shape,  "min/max:", mask.min().item(),  mask.max().item())

            # 1) 点级类别logits：用 mask 对 query 进行加权
            m = mask.float()
            if m.min() < 0 or m.max() > 1:
                # 认为是mask logits，则先 sigmoid 成概率
                m = m.sigmoid()
            confid = m.matmul(logit)  # [N, C]
            print("confid shape:", confid.shape, "min/max:", confid.min().item(), confid.max().item())

            # 2) Energy（越大越异常）
            #    E = -T * logsumexp(z/T)
            energy_all = -T * torch.logsumexp(confid / max(T, eps), dim=1)  # [N]
            energy = energy_all[inv_map]  # [N']
            print("energy (raw) min/max:", energy.min().item(), energy.max().item())

            # 3) 归一化为 [0,1]（可选）
            if normalize == "none":
                return energy  # 保留原始能量（更高=更异常）

            if normalize == "minmax":
                mn, mx = energy.min(), energy.max()
                if (mx - mn).abs() > eps:
                    score = (energy - mn) / (mx - mn)
                else:
                    score = torch.zeros_like(energy)
                print("energy (minmax) min/max:", score.min().item(), score.max().item())
                return score

            if normalize == "quantile":
                ql, qh = torch.quantile(energy, torch.tensor([q_low, q_high], device=energy.device))
                denom = (qh - ql).clamp_min(eps)
                score = torch.clamp((energy - ql) / denom, 0.0, 1.0)
                print("energy (quantile) min/max:", score.min().item(), score.max().item())
                return score

            if normalize == "sigmoid":
                mu = energy.mean()
                std = energy.std().clamp_min(eps)
                score = torch.sigmoid((energy - mu) / std)  # 中心化后映射到[0,1]
                print("energy (sigmoid) min/max:", score.min().item(), score.max().item())
                return score

            # fallback
            return energy




        pred_logits = output["pred_logits"] # [Nq, C]
        pred_logits = torch.functional.F.softmax(pred_logits, dim=-1)[..., :-1]
        pred_masks = output["pred_masks"] # [Nq, N_voxel]

        save_path = Path(self.config.general.save_dir) / "prediction_energy"
        for b_idx in range(len(pred_logits)):
            # rba_score = get_rba(
            #     pred_logits[b_idx], pred_masks[b_idx], inverse_maps[b_idx]
            # )
            # here are some of the commented-out lines that were used to predict rba and max_logit
            # at the same time
            # msp = get_msp(
            #     pred_logits[b_idx], pred_masks[b_idx], inverse_maps[b_idx]
            # )
            start_energy_uncertainity = time.time()
            energy = get_energy(
                pred_logits[b_idx], pred_masks[b_idx], inverse_maps[b_idx]
            )

            total_energy_uncertainity = (time.time() - start_energy_uncertainity) * 1000
            print("total_energy_uncertainity-----------------: ", total_energy_uncertainity)

                        # ------- 新增：把每一帧的时间记录下来 -------
            self.test_inference_times.append(float(inference_time))
            self.test_energy_times.append(float(total_energy_uncertainity))
                



            base_path = save_path / f"{sequences[b_idx][0]}"

            #rba_save_path = base_path / f"{sequences[b_idx][1]}.txt"
            # rba_save_path = base_path / f"rba_{sequences[b_idx][1]}.txt"
            energy_save_path = base_path / f"energy_{sequences[b_idx][1]}.txt"
            # if not rba_save_path.parent.exists():
            #     rba_save_path.parent.mkdir(exist_ok=True, parents=True)

            if not energy_save_path.parent.exists():
                energy_save_path.parent.mkdir(exist_ok=True, parents=True)
            #np.savetxt(rba_save_path, rba_score.detach().cpu().numpy())
            np.savetxt(energy_save_path, energy.detach().cpu().numpy())

            # Save inlier predictions
            confid = pred_masks[b_idx].float().sigmoid().matmul(pred_logits[b_idx])  # [N_voxel, C]

            #4) 体素级语义预测：对类别做argmax # [N_voxel]
            # 5) 回映射到“原始点顺序”：用 inv_map 把体素标签扩展到每个原始点 # [N_raw]
            sem_preds = torch.argmax(confid, dim=1)[inverse_maps[b_idx]].cpu().numpy()
            sem_preds_path = base_path / f"sem_preds_{sequences[b_idx][1]}.txt"
            np.savetxt(sem_preds_path, sem_preds)



            # 2) 组织输出路径：.../prediction/<sequence_name>/
            save_path = Path(self.config.general.save_dir) / "prediction_energy"
            base_path = save_path / f"{sequences[b_idx][0]}"    # sequences[b_idx] = (seq_name, frame_id_str)
            base_path.mkdir(parents=True, exist_ok=True)

            # 3) 规范帧号为 6 位：000000
            frame_id = int(sequences[b_idx][1])  # 如果已经是 "000123" 这种字符串，也可以直接用
            frame_str = f"{frame_id:06d}"

            # 4) 保存为 .npy，名字要和 --prediction_type 一致（例如 ml）
            np.save(base_path / f"energy_{frame_str}.npy",
                    energy.detach().cpu().numpy().astype(np.float32))

        return {}

    def validation_step(self, batch, batch_idx):
        data, target = batch
        inverse_maps = data.inverse_maps
        original_labels = data.original_labels
        raw_coordinates = data.raw_coordinates
        num_points = data.num_points
        sequences = data.sequences

        data = ME.SparseTensor(
            coordinates=data.coordinates, features=data.features, device=self.device
        )
        output = self.forward(data, raw_coordinates=raw_coordinates, is_eval=True)
        losses = self.criterion(output, target)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        pred_logits = output["pred_logits"]
        pred_logits = torch.functional.F.softmax(pred_logits, dim=-1)[..., :-1]
        pred_masks = output["pred_masks"]
        offset_coords_idx = 0

        for logit, mask, map, label, n_point, seq in zip(
            pred_logits,
            pred_masks,
            inverse_maps,
            original_labels,
            num_points,
            sequences,
        ):
            seq = seq[0]
            if seq != self.last_seq:
                self.last_seq = seq
                self.previous_instances = None
                self.max_instance_id = self.config.model.num_queries
                self.scene = 0

            class_confidence, classes = torch.max(logit.detach().cpu(), dim=1)
            foreground_confidence = mask.detach().cpu().float().sigmoid()
            confidence = class_confidence[None, ...] * foreground_confidence
            confidence = confidence[map].numpy()

            ins_preds = np.argmax(confidence, axis=1)
            sem_preds = classes[ins_preds].numpy() + 1
            ins_preds += 1
            ins_preds[
                np.isin(
                    sem_preds, range(1, self.config.data.min_stuff_cls_id), invert=True
                )
            ] = 0
            sem_labels = self.validation_dataset._remap_model_output(label[:, 0])
            ins_labels = label[:, 1] >> 16

            db_max_instance_id = self.config.model.num_queries
            if self.config.general.dbscan_eps is not None:
                curr_coords_idx = mask.shape[0]
                curr_coords = raw_coordinates[
                    offset_coords_idx : curr_coords_idx + offset_coords_idx, :3
                ]
                curr_coords = curr_coords[map].detach().cpu().numpy()
                offset_coords_idx += curr_coords_idx

                ins_ids = np.unique(ins_preds)
                for ins_id in ins_ids:
                    if ins_id != 0:
                        instance_mask = ins_preds == ins_id
                        clusters = (
                            DBSCAN(
                                eps=self.config.general.dbscan_eps,
                                min_samples=1,
                                n_jobs=-1,
                            )
                            .fit(curr_coords[instance_mask])
                            .labels_
                        )
                        new_mask = np.zeros(ins_preds.shape, dtype=np.int64)
                        new_mask[instance_mask] = clusters + 1
                        for cluster_id in np.unique(new_mask):
                            if cluster_id != 0:
                                db_max_instance_id += 1
                                ins_preds[new_mask == cluster_id] = db_max_instance_id

            self.max_instance_id = max(db_max_instance_id, self.max_instance_id)
            for i in range(len(n_point) - 1):
                indices = range(n_point[i], n_point[i + 1])
                if i == 0 and self.previous_instances is not None:
                    current_instances = ins_preds[indices]
                    associations = associate_instances(
                        self.previous_instances, current_instances
                    )
                    for id in np.unique(ins_preds):
                        if associations.get(id) is None:
                            self.max_instance_id += 1
                            associations[id] = self.max_instance_id
                    ins_preds = np.vectorize(associations.__getitem__)(ins_preds)
                else:
                    self.class_evaluator.addBatch(
                        sem_preds, ins_preds, sem_labels, ins_labels, indices, seq
                    )
            if i > 0:
                self.previous_instances = ins_preds[indices]

        return {f"val_{k}": v.detach().cpu().item() for k, v in losses.items()}

    def training_epoch_end(self, outputs):
        train_loss = sum([out["loss"].cpu().item() for out in outputs]) / len(outputs)
        results = {"train_loss_mean": train_loss}
        self.log_dict(results)

    def validation_epoch_end(self, outputs):
        self.last_seq = None
        class_names = self.config.data.class_names
        pq, sq, rq, all_pq, all_sq, all_rq = self.class_evaluator.getPQ()
        self.class_evaluator.reset()
        results = {}
        results["val_mean_pq"] = pq
        results["val_mean_sq"] = sq
        results["val_mean_rq"] = rq
        # print(class_names)
        # print(all_pq)
        # print(all_sq)
        # print(all_rq)
        for i, (pq, sq, rq) in enumerate(zip(all_pq, all_sq, all_rq)):
            results[f"val_{class_names[i-1]}_pq"] = pq.item()
            results[f"val_{class_names[i-1]}_sq"] = sq.item()
            results[f"val_{class_names[i-1]}_rq"] = rq.item()
        self.log_dict(results)
        print(results)

        dd = defaultdict(list)
        for output in outputs:
            for key, val in output.items():
                dd[key].append(val)

        dd = {k: statistics.mean(v) for k, v in dd.items()}

        dd["val_mean_loss_ce"] = statistics.mean(
            [item for item in [v for k, v in dd.items() if "loss_ce" in k]]
        )
        dd["val_mean_loss_mask"] = statistics.mean(
            [item for item in [v for k, v in dd.items() if "loss_mask" in k]]
        )
        dd["val_mean_loss_dice"] = statistics.mean(
            [item for item in [v for k, v in dd.items() if "loss_dice" in k]]
        )
        dd["val_mean_loss_box"] = statistics.mean(
            [item for item in [v for k, v in dd.items() if "loss_box" in k]]
        )
        self.log_dict(dd)

    # def test_epoch_end(self, outputs):
    #     return {}
    
    def test_epoch_end(self, outputs):
        # 防止除零
        if len(self.test_inference_times) > 0:
            avg_inf = sum(self.test_inference_times) / len(self.test_inference_times)
            avg_energy = sum(self.test_energy_times) / len(self.test_energy_times)

            print(f"\n=== Test Epoch Summary ===")
            print(f"Average pure inference time: {avg_inf:.2f} ms")
            print(f"Average energy post-processing time: {avg_energy:.2f} ms")

            # 如果你也想用 Lightning 的 logger 记录到日志里：
            self.log("test/avg_inference_time_ms", avg_inf)
            self.log("test/avg_energy_time_ms", avg_energy)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.config.optimizer, params=self.parameters()
        )
        if "steps_per_epoch" in self.config.scheduler.scheduler.keys():
            self.config.scheduler.scheduler.steps_per_epoch = len(
                self.train_dataloader()
            )
        lr_scheduler = hydra.utils.instantiate(
            self.config.scheduler.scheduler, optimizer=optimizer
        )
        scheduler_config = {"scheduler": lr_scheduler}
        scheduler_config.update(self.config.scheduler.pytorch_lightning_params)
        return [optimizer], [scheduler_config]

    def prepare_data(self):
        self.train_dataset = hydra.utils.instantiate(self.config.data.train_dataset)
        self.validation_dataset = hydra.utils.instantiate(
            self.config.data.validation_dataset
        )
        self.test_dataset = hydra.utils.instantiate(self.config.data.test_dataset)

    def train_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.train_collation)
        return hydra.utils.instantiate(
            self.config.data.train_dataloader,
            self.train_dataset,
            collate_fn=c_fn,
        )

    def val_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
        return hydra.utils.instantiate(
            self.config.data.validation_dataloader,
            self.validation_dataset,
            collate_fn=c_fn,
        )

    def test_dataloader(self):
        c_fn = hydra.utils.instantiate(self.config.data.test_collation)
        return hydra.utils.instantiate(
            self.config.data.test_dataloader,
            self.test_dataset,
            collate_fn=c_fn,
        )
