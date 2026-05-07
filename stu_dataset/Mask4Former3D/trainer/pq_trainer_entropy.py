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
        self.test_entropy_times = []

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


        def get_entropy(logit, mask, inv_map, T: float = 1.0, base: str = "e"):
            """
            Entropy-based OOD score per point (higher = more OOD).
            Steps:
            1) 点级类别logits: confid = sigmoid(mask) @ logit
            2) 温度缩放: confid/T
            3) softmax -> p
            4) 熵 H = -sum p*log(p)
            5) 归一化: H / log(C) ∈ [0,1]  (自然对数为基)
            Args:
                logit: [Q, C]  query-level class logits
                mask:  [N, Q]  point-query mask logits/probs
                inv_map: [N']  indices to restore original order
                T: temperature for softmax (T>1 更平坦 -> 熵更大)
                base: "e" | "2" 仅影响打印，归一化始终按 ln(C)
            Returns:
                ent_norm: [N'] ∈ [0,1], higher means more uncertain (more OOD)
            """
            print("get_entropy called")
            print("logit shape:", logit.shape, "min/max:", logit.min().item(), logit.max().item())
            print("mask  shape:", mask.shape,  "min/max:", mask.min().item(),  mask.max().item())

            # 1) 点级类别 logits
            m = mask.float()
            if m.min() < 0 or m.max() > 1:
                m = m.sigmoid()
            confid = m.matmul(logit)                      # [N, C]
            print("confid shape:", confid.shape, "min/max:", confid.min().item(), confid.max().item())

            # 2) 温度缩放
            confid = confid / max(T, 1e-6)

            # 3) softmax 概率
            probs = F.softmax(confid, dim=1)              # [N, C]

            # 4) 熵（自然对数）
            eps = 1e-12
            ent_all = -(probs * (probs.add(eps).log())).sum(dim=1)   # [N]
            ent = ent_all[inv_map]                                   # [N']
            print("entropy (nat log) min/max:", ent.min().item(), ent.max().item())

            # 5) 归一化到 [0,1]，用 ln(C) 作为上界
            C = logit.shape[1]
            ent_norm = ent / (torch.log(torch.tensor(C, dtype=ent.dtype, device=ent.device)) + 1e-12)
            ent_norm = ent_norm.clamp(0.0, 1.0)

            # 可选：显示以2为底的熵范围（仅打印）
            if base == "2":
                ent_base2 = ent / (torch.log(torch.tensor(2.0, dtype=ent.dtype, device=ent.device)) + 1e-12)
                print("entropy base-2 min/max:", ent_base2.min().item(), ent_base2.max().item())

            print("entropy (normalized) min/max:", ent_norm.min().item(), ent_norm.max().item())
            return ent_norm




        pred_logits = output["pred_logits"] # [Nq, C]
        pred_logits = torch.functional.F.softmax(pred_logits, dim=-1)[..., :-1]
        pred_masks = output["pred_masks"] # [Nq, N_voxel]

     
        save_path = Path(self.config.general.save_dir) / "prediction_entropy"
        for b_idx in range(len(pred_logits)):
            # rba_score = get_rba(
            #     pred_logits[b_idx], pred_masks[b_idx], inverse_maps[b_idx]
            # )
            # here are some of the commented-out lines that were used to predict rba and max_logit
            # at the same time


            start_entropy_uncertainity = time.time()
            entropy = get_entropy(
                pred_logits[b_idx], pred_masks[b_idx], inverse_maps[b_idx]
            )

            total_entropy_uncertainity = (time.time() - start_entropy_uncertainity) * 1000
            print("total_entropy_uncertainity-----------------: ", total_entropy_uncertainity)

                        # ------- 新增：把每一帧的时间记录下来 -------
            self.test_inference_times.append(float(inference_time))
            self.test_entropy_times.append(float(total_entropy_uncertainity))
                



            base_path = save_path / f"{sequences[b_idx][0]}"

            #rba_save_path = base_path / f"{sequences[b_idx][1]}.txt"
            # rba_save_path = base_path / f"rba_{sequences[b_idx][1]}.txt"
            entropy_save_path = base_path / f"entropy_{sequences[b_idx][1]}.txt"
            # if not rba_save_path.parent.exists():
            #     rba_save_path.parent.mkdir(exist_ok=True, parents=True)

            if not entropy_save_path.parent.exists():
                entropy_save_path.parent.mkdir(exist_ok=True, parents=True)
            #np.savetxt(rba_save_path, rba_score.detach().cpu().numpy())
            np.savetxt(entropy_save_path, entropy.detach().cpu().numpy())

            # Save inlier predictions
            confid = pred_masks[b_idx].float().sigmoid().matmul(pred_logits[b_idx])  # [N_voxel, C]

            #4) 体素级语义预测：对类别做argmax # [N_voxel]
            # 5) 回映射到“原始点顺序”：用 inv_map 把体素标签扩展到每个原始点 # [N_raw]
            sem_preds = torch.argmax(confid, dim=1)[inverse_maps[b_idx]].cpu().numpy()
            sem_preds_path = base_path / f"sem_preds_{sequences[b_idx][1]}.txt"
            np.savetxt(sem_preds_path, sem_preds)



            # 2) 组织输出路径：.../prediction/<sequence_name>/
            
            base_path = save_path / f"{sequences[b_idx][0]}"    # sequences[b_idx] = (seq_name, frame_id_str)
            base_path.mkdir(parents=True, exist_ok=True)

            # 3) 规范帧号为 6 位：000000
            frame_id = int(sequences[b_idx][1])  # 如果已经是 "000123" 这种字符串，也可以直接用
            frame_str = f"{frame_id:06d}"

            # 4) 保存为 .npy，名字要和 --prediction_type 一致（例如 ml）
            np.save(base_path / f"entropy_{frame_str}.npy",
                    entropy.detach().cpu().numpy().astype(np.float32))

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
            avg_entropy = sum(self.test_entropy_times) / len(self.test_entropy_times)

            print(f"\n=== Test Epoch Summary ===")
            print(f"Average pure inference time: {avg_inf:.2f} ms")
            print(f"Average entropy post-processing time: {avg_entropy:.2f} ms")

            # 如果你也想用 Lightning 的 logger 记录到日志里：
            self.log("test/avg_inference_time_ms", avg_inf)
            self.log("test/avg_entropy_time_ms", avg_entropy)
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
