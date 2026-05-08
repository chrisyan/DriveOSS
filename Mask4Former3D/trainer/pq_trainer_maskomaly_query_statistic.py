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

import os

import time
import csv
import os

'''
maskomaly
'''


def _sync_device(dev):
    try:
        if torch.cuda.is_available() and hasattr(dev, "type") and dev.type == "cuda":
            torch.cuda.synchronize(dev)
    except Exception:
        pass

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
    
    def calculate_statistics(self):
        self.model.eval()
        self.stats_computed = True
        #print("\n[STATS] Computing dataset statistics...")

        # Initialize statistics storage
        stats = {
            'sum_logit': defaultdict(float),       # For mean calculation
            'sum_sq_logit': defaultdict(float),    # For variance calculation
            'sum_mask': defaultdict(float),        # For mask mean
            'count': defaultdict(int)              # Query counts per class
        }

        #Try to print the above stats
        os.makedirs('stats', exist_ok=True)

        with torch.no_grad():
            for i, batch in enumerate(self.train_dataloader()):    ##In this at a time only 24 iterations happens and this calculates mean of just 24 batch
                #print(f"\nBatch {i} type: {type(batch)}")
                #print(f"Batch contents: {batch}")
                if i >= 4:  # Limit to 24 batches
                    break

                data, _ = batch
                raw_coordinates = data.raw_coordinates

                data = ME.SparseTensor(
                    coordinates=data.coordinates,
                    features=data.features,
                    device=self.device
                )
                output = self.forward(data, raw_coordinates=raw_coordinates, is_eval=True)   ##I guess output loading is happeining twice so it is taking log time

                pred_logits = output["pred_logits"]  # [B, N_queries, N_classes]
                pred_masks = output["pred_masks"]    # List of [N_queries, H, W] tensors

                probs = torch.softmax(pred_logits, dim=-1)[..., :-1]  # Exclude background
                _, pred_classes = probs.max(-1)  # [B, N_queries]

                for b in range(pred_logits.shape[0]):  #Iterates over batches
                    batch_masks = pred_masks[b] if len(pred_masks) > b else None

                    if batch_masks is None:
                        print(f"Warning: No masks for batch index {b}")
                        continue

                    for q in range(pred_logits.shape[1]):   #Iterates over queries within that batch
                        c = pred_classes[b,q].item()        # Predicted class index from that batch and specific query which is class has max logit
                        logit_val = pred_logits[b,q,c].item()   #raw score for selected class

                        # Update running statistics
                        stats['sum_logit'][c] += logit_val
                        stats['sum_sq_logit'][c] += logit_val ** 2  
                        stats['count'][c] += 1

                        # Handle mask statistics
                        if q < len(batch_masks):
                            mask_prob = batch_masks[q].sigmoid().mean().item()
                            stats['sum_mask'][c] += mask_prob

                # Memory cleanup
                del output, pred_logits, pred_masks
                torch.cuda.empty_cache()

        # Compute final statistics
        mean_dict = {}
        var_dict = {}
        mask_mean_dict = {}

        for c in stats['count']:
            count = stats['count'][c]
            if count == 0:
                continue

            mean_dict[c] = stats['sum_logit'][c] / count   #it has mean of 24 batches and this we are saving
            var_dict[c] = (stats['sum_sq_logit'][c] / count) - mean_dict[c]**2
            mask_mean_dict[c] = stats['sum_mask'][c] / count

        # Save to .npy files
        np.save('stats/query_mean.npy', mean_dict)
        np.save('stats/query_var.npy', var_dict)
        np.save('stats/query_mask_mean.npy', mask_mean_dict)

        print("\n==========================STATISTICS SUMMARY==============================")
        for c in sorted(mean_dict.keys()):
            print(f"Class {c}:")
            print(f"  Queries: {stats['count'][c]}")
            print(f"  Mean logit: {mean_dict[c]:.3f}")
            print(f"  Logit variance: {var_dict[c]:.3f}")
            print(f"  Mean mask prob: {mask_mean_dict[c]:.3f}\n")

        return mean_dict, var_dict    

    def save_pred_logits(self, pred_logits, batch_idx):
        """Save raw logits to a text file with batch index"""
        os.makedirs('logits_dump', exist_ok=True)
        filename = f'logits_dump/batch_{batch_idx}_logits_1.txt'

        # Convert to numpy and save
        np.savetxt(filename, pred_logits.cpu().numpy().reshape(-1, pred_logits.shape[-1]))
        print(f"Saved raw logits to {filename}")
    
    def save_pred_logits_sm(self, pred_logits, batch_idx):
        """Save raw logits to a text file with batch index"""
        os.makedirs('logits_dump', exist_ok=True)
        filename = f'logits_dump/batch_{batch_idx}_logits_sm_1.txt'

        # Convert to numpy and save
        np.savetxt(filename, pred_logits.cpu().numpy().reshape(-1, pred_logits.shape[-1]))
        print(f"Saved raw logit_sm to {filename}")

    def save_pred_mask(self, pred_masks, batch_idx):
        """Save raw logits to a text file with batch index"""
        os.makedirs('logits_dump', exist_ok=True)
        filename = f'logits_dump/batch_{batch_idx}_masks_1.txt'

        if isinstance(pred_masks, list):
            # Case 1: pred_masks is a list of tensors
            mask_array = torch.cat([m.cpu() for m in pred_masks], dim=0).numpy()
        elif isinstance(pred_masks, torch.Tensor):
            # Case 2: pred_masks is already a tensor
            mask_array = pred_masks.cpu().numpy()
        else:
            raise ValueError(f"Unsupported mask type: {type(pred_masks)}")
    
        # Save with 4 decimal places
        np.savetxt(filename, mask_array, fmt='%.4f')
        print(f"Saved masks to {filename} (shape: {mask_array.shape})")

    def save_remapping_data(self, batch_data, batch_idx):
        """Save inverse maps in .txt format for short-term visualization"""
        os.makedirs('logits_dump', exist_ok=True)
        
        # Save inverse maps as .txt (same format as your other files)
        inverse_maps = batch_data.inverse_maps[0].cpu().numpy()
        np.savetxt(f'logits_dump/batch_{batch_idx}_inverse_maps.txt', 
                   inverse_maps, fmt='%d')  # Save as integers
        
        # Save raw coordinates (optional but recommended)
        raw_coords = batch_data.raw_coordinates.cpu().numpy()
        np.savetxt(f'logits_dump/batch_{batch_idx}_raw_coords.txt', 
                   raw_coords, fmt='%.6f')  # 6 decimal places for coordinates
        
        print(f"Saved remapping data for batch {batch_idx} (TXT format)")



    def test_step(self, batch, batch_idx):
        data, _ = batch
        inverse_maps   = data.inverse_maps         # List[Tensor], 每个 Tensor: [N_raw_points]
        raw_coordinates = data.raw_coordinates     # List[Tensor], 每个 Tensor: [N_raw_points, >=3]
        sequences      = data.sequences           # [(sequence_name, frame_id), ...]

        # ---------------- 计时：总开始 ----------------
        t_total_start = time.perf_counter()

        # 构建稀疏张量
        data = ME.SparseTensor(
            coordinates=data.coordinates,
            features=data.features,
            device=self.device
        )

        # =============== 模型推理 ===============
        _sync_device(self.device)
        t_fwd_start = time.perf_counter()

        output = self.forward(data, raw_coordinates=raw_coordinates, is_eval=True)

        _sync_device(self.device)
        t_fwd_end = time.perf_counter()
        model_forward_time = t_fwd_end - t_fwd_start

        pred_logits = output["pred_logits"]       # 期望: [B, Q, C]
        pred_masks  = output["pred_masks"]        # 可能是 list(每元素 [N_vox, Q]) 或 Tensor([B, Q, N_vox])

        # ---- 打印尺寸（便于确认结构）----
        if isinstance(pred_logits, torch.Tensor):
            print("pred_logits shape:", tuple(pred_logits.shape))
        else:
            print("pred_logits type:", type(pred_logits))
            if isinstance(pred_logits, (list, tuple)) and len(pred_logits) > 0:
                print("pred_logits[0] shape:", tuple(pred_logits[0].shape))

        if isinstance(pred_masks, torch.Tensor):
            print("pred_masks shape:", tuple(pred_masks.shape))
        else:
            print("pred_masks type:", type(pred_masks))
            if isinstance(pred_masks, (list, tuple)) and len(pred_masks) > 0:
                print("pred_masks[0] shape:", tuple(pred_masks[0].shape))

        save_root = Path(self.config.general.save_dir) / "prediction_maskomaly"
        save_root.mkdir(parents=True, exist_ok=True)

        # 准备计时 CSV（一次性写表头）
        timing_csv = save_root / "timing.csv"
        write_header = (not timing_csv.exists())
        if write_header:
            with timing_csv.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sequence", "frame_id", "model_forward_s", "maskomaly_s", "io_s", "total_s", "fps"])

        # ---------------- maskomaly + 保存（按样本循环并各自计时） ----------------
        B = pred_logits.shape[0] if isinstance(pred_logits, torch.Tensor) else len(sequences)
        for b_idx in range(B):
            # 各子阶段计时累加器
            t_mask_start = time.perf_counter()  # maskomaly 计时起点

            # ------------------ 取出本样本的数据 ------------------
            logits = pred_logits[b_idx] if isinstance(pred_logits, torch.Tensor) else pred_logits[b_idx]
            # 取 masks（统一拿到 voxel-level 的 [N_vox, Q] 形状）
            if isinstance(pred_masks, list):
                if len(pred_masks) == 1:
                    masks_voxQ = pred_masks[0]             # [N_vox, Q]
                elif len(pred_masks) > b_idx:
                    masks_voxQ = pred_masks[b_idx]         # [N_vox, Q]
                else:
                    raise IndexError(f"pred_masks list length {len(pred_masks)} <= b_idx {b_idx}")
                assert masks_voxQ.ndim == 2, f"pred_masks[{b_idx}] ndim={masks_voxQ.ndim} !=2"
            elif isinstance(pred_masks, torch.Tensor):
                m = pred_masks[b_idx]
                if m.ndim != 2:
                    if m.shape[0] <= m.shape[1]:
                        masks_voxQ = m.T
                    else:
                        masks_voxQ = m
                else:
                    if m.shape[0] < m.shape[1]:
                        masks_voxQ = m.T
                    else:
                        masks_voxQ = m
            else:
                raise TypeError(f"Unsupported pred_masks type: {type(pred_masks)}")

            inv_map = inverse_maps[b_idx]   # [N_raw_points]
            seq_name, frame_id = sequences[b_idx]

            # ------------------ 路径与帧号 ------------------
            base_path = save_root / f"{seq_name}"
            base_path.mkdir(parents=True, exist_ok=True)
            dump_dir = base_path / "logits_dump"
            dump_dir.mkdir(parents=True, exist_ok=True)

            try:
                frame_int = int(frame_id)
                frame_str = f"{frame_int:06d}"
            except Exception:
                frame_str = str(frame_id).zfill(6)

            # ---------- 保存前：先把 maskomaly 相关计算计时完 ----------
            # （1）拿到 numpy
            logits_np = logits.detach().cpu().numpy().astype(np.float32, copy=False)     # [Q, C]
            masks_np  = masks_voxQ.detach().cpu().numpy().astype(np.float32, copy=False) # [N_vox, Q]

            # （2）分类 softmax 概率
            cls_probs = torch.from_numpy(logits_np).softmax(dim=-1).numpy()  # [Q, C]
            # （3）mask 概率 [Q, N_vox]
            mask_probs_QV = 1.0 / (1.0 + np.exp(-masks_np.T))

            Q, C = cls_probs.shape
            N_vox = mask_probs_QV.shape[1]
            print("num_queries", Q)
            print("num_voxel_points", N_vox)

            # （4）S_r：Rejection-based
            soft_mask_rej = np.ones(N_vox, dtype=np.float32)
            max_class = np.argmax(cls_probs, axis=1)          # [Q]
            max_conf  = cls_probs[np.arange(Q), max_class]    # [Q]
            for i in range(Q):
                if max_class[i] != (C - 1) and max_conf[i] > 0.7:
                    soft_mask_rej = np.minimum(soft_mask_rej, 1.0 - mask_probs_QV[i] * max_conf[i])

            # （5）S_a：Acceptance-based
            soft_mask_acc = np.zeros(N_vox, dtype=np.float32)
            #ranking = [49, 31, 83, 32]
                        # ranking = [49, 31, 83, 32]  # top anomaly-sensitive queries zhiran
            ranking = [9, 15, 16, 2]  # top anomaly-sensitive queries zhiran
            for ind in ranking:
                if 0 <= ind < Q:
                    conf = float(cls_probs[ind].max())
                    soft_mask_acc = np.maximum(soft_mask_acc, mask_probs_QV[ind] * conf)

            # （6）最终 voxel-level soft mask
            final_soft_mask_voxel = 0.6 * soft_mask_rej + 0.4 * soft_mask_acc  # [N_vox]

            # （7）voxel → raw points
            inv_map_np = inv_map.detach().cpu().numpy().astype(np.int64).reshape(-1)     # [N_raw]
            if inv_map_np.max() >= final_soft_mask_voxel.shape[0]:
                raise IndexError(
                    f"inv_map max index {inv_map_np.max()} exceeds soft mask length {final_soft_mask_voxel.shape[0]}"
                )
            final_soft_mask = final_soft_mask_voxel[inv_map_np].astype(np.float32, copy=False)  # [N_raw]

            # ------- maskomaly 计时结束 -------
            t_mask_end = time.perf_counter()
            maskomaly_time = t_mask_end - t_mask_start

            # =============== I/O 保存 ===============
            t_io_start = time.perf_counter()

            # logits / masks0
            np.save(dump_dir / f"{frame_str}_pred_logits.npy", logits_np)
            np.save(dump_dir / f"{frame_str}_pred_masks0.npy", masks_np)
            print(f"[Save] {seq_name}/{frame_str}: logits {logits_np.shape}, masks0 {masks_np.shape}")

            # inv_map
            inv_txt_path = base_path / f"{frame_str}_inv_map.txt"
            np.savetxt(inv_txt_path, inv_map_np, fmt="%d")
            print(f"[Save] {seq_name}/{frame_str}: inv_map {inv_map_np.shape} -> {inv_txt_path}")

            # 体素坐标（与 masks_voxQ 对齐）
            coords_all = data.C.detach().cpu()  # [N_all, 1 + D]
            mask_this = (coords_all[:, 0] == b_idx)
            coords_b = coords_all[mask_this, 1:4].to(torch.int32)  # [N_vox, 3]
            coords_np = coords_b.numpy().astype(np.int32, copy=False)
            vox_coords_path = base_path / f"{frame_str}_voxel_coords.npy"
            np.save(vox_coords_path, coords_np)
            print(f"[Save] {seq_name}/{frame_str}: voxel_coords -> {vox_coords_path} shape={coords_np.shape}")

            # final soft mask
            txt_path = base_path / f"{frame_str}.txt"
            npy_path = base_path / f"maskomaly_{frame_str}.npy"
            np.savetxt(txt_path, final_soft_mask, fmt="%.6f")
            np.save(npy_path, final_soft_mask)
            print(f"[Save] {seq_name}/{frame_str}: final_soft_mask -> {txt_path}, {npy_path}")

            t_io_end = time.perf_counter()
            io_time = t_io_end - t_io_start

            # 这一帧总耗时
            total_time = model_forward_time + maskomaly_time + io_time
            fps = (1.0 / total_time) if total_time > 0 else float("inf")

            print(
                f"[Timing] {seq_name}/{frame_str} | forward: {model_forward_time:.4f}s, "
                f"maskomaly: {maskomaly_time:.4f}s, io: {io_time:.4f}s, total: {total_time:.4f}s "
                f"({fps:.2f} FPS)"
            )

            # 写入 CSV
            with timing_csv.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([seq_name, frame_str, f"{model_forward_time:.6f}",
                                f"{maskomaly_time:.6f}", f"{io_time:.6f}", f"{total_time:.6f}", f"{fps:.3f}"])

        # ---------------- 计时：总结束（可选） ----------------
        t_total_end = time.perf_counter()
        # 如果你想包含 batch 级别所有样本在内的总时间，可在此打印：
        print(f"[Timing] batch#{batch_idx} total wall time: {t_total_end - t_total_start:.4f}s")

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

    def test_epoch_end(self, outputs):
        return {}

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
