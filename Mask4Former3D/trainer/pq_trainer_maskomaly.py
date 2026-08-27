import os
import statistics
import time
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

'''
maskomaly
'''

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
        self.test_maskomaly_times = []


    def on_test_start(self) -> None:
        # forward（按 batch 和按帧）与 anomaly scoring（按帧）的累计器
        self._fw_time_sum_batches = 0.0
        self._fw_time_batches = 0

        self._fw_time_sum_frames = 0.0
        self._fw_time_frames = 0

        self._ano_time_sum_frames = 0.0
        self._ano_time_frames = 0

        print("✅ Inference timers initialized (forward/anomaly scoring)")



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
        inverse_maps = data.inverse_maps
        raw_coordinates = data.raw_coordinates
        sequences = data.sequences

                # =============================
        # BEGIN: Forward 推理计时（GPU）
        # =============================
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        start_inference = time.time()

        data = ME.SparseTensor(
            coordinates=data.coordinates,
            features=data.features,
            device=self.device
        )



        output = self.forward(data, raw_coordinates=raw_coordinates, is_eval=True)

        inference_time = (time.time() - start_inference) * 1000


        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        forward_time_batch = t1 - t0  # 当前 batch 的前向总耗时（包含 B 帧）

        pred_logits = output["pred_logits"]
        pred_masks = output["pred_masks"]

        B = len(pred_logits)  # batch size（帧数）
        # 按 batch 统计
        self._fw_time_sum_batches += forward_time_batch
        self._fw_time_batches += 1
        # 按帧均分统计与打印
        forward_time_per_frame = forward_time_batch / max(B, 1)
        self._fw_time_sum_frames += forward_time_per_frame * B
        self._fw_time_frames += B

        print(f"[Timing][Batch {batch_idx}] forward: {forward_time_batch:.4f}s "
            f"(~{forward_time_per_frame:.4f}s/frame for {B} frames)")
        # =============================
        # END: Forward 推理计时（GPU）
        # =============================

        save_path = Path(self.config.general.save_dir) / "prediction_maskomaly"

        for b_idx in range(B):
            logits = pred_logits[b_idx]     # [Nq, C]
            masks = pred_masks[b_idx]       # [Nq, N_voxel]
            inv_map = inverse_maps[b_idx]   # [N_raw]
            seq_name, frame_id = sequences[b_idx]

            # ----------------------------------------------------
            # BEGIN: anomaly scoring 计时（CPU/Numpy 端）
            # （包含 sigmoid/softmax、S_r/S_a、融合、inv_map 回填）
            # ----------------------------------------------------
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # 确保显存到 Host 的拷贝完成再计时
            t2 = time.perf_counter()

  
            logits = pred_logits[b_idx]     # [N_queries, C]
            masks = pred_masks[b_idx]       # [N_queries, N_voxel_points]
            inv_map = inverse_maps[b_idx]   # [N_raw_points]
            seq_name, frame_id = sequences[b_idx]

            # 转为 numpy
            mask_probs = masks.sigmoid().detach().cpu().numpy()  # [N_queries, N_voxel_points]
            cls_probs = logits.softmax(dim=-1).detach().cpu().numpy()  # [N_queries, num_classes]
            mask_probs = mask_probs.T
            print("mask_probs", mask_probs.shape)
            print("cls_probs", cls_probs.shape)

            num_queries = cls_probs.shape[0]          # 应该是 100
            num_voxel_points = mask_probs.shape[1]    # 应该是 63753
            print("num_voxel_points", num_voxel_points)

            num_classes = cls_probs.shape[1]
            print("num_queries", num_queries)

            # -------------------------------
            # Rejection-based Soft Mask (S_r)
            # -------------------------------

            start_maskomaly_uncertainity = time.time()

            soft_mask_rej = np.ones(num_voxel_points, dtype=np.float32)
            print("soft_mask_rej", soft_mask_rej.shape)
            for i in range(num_queries):
                max_class = np.argmax(cls_probs[i])
                max_conf = cls_probs[i][max_class]
                if max_class != num_classes - 1 and max_conf > 0.4: #0.7
                    soft_mask_rej = np.minimum(soft_mask_rej, 1 - mask_probs[i] * max_conf)

            # -------------------------------
            # Acceptance-based Soft Mask (S_a)
            # -------------------------------
            soft_mask_acc = np.zeros(num_voxel_points, dtype=np.float32)
            print("soft_mask_acc", soft_mask_acc.shape)
            # ranking = [49, 31, 83, 32]  # top anomaly-sensitive queries 
            ranking = [9, 15, 16, 2]  # top anomaly-sensitive queries 
            for ind in ranking:
                if ind < num_queries:
                    conf = np.max(cls_probs[ind])
                    soft_mask_acc = np.maximum(soft_mask_acc, mask_probs[ind] * conf)

            # -------------------------------
            # Final Maskomaly Soft Mask (S) ablation study on lambda
            # 0    1
            # 0.2   0.8
            # 0.4   0.6
            # 0.5   0.5
            # 0.6   0.4  best done
            # 0.8   0.2
            # 0.9 0.1 :
            # 0.7: AP': 8.380308480000508, 'FPR95': 34.416770166560134, 'AUROC': 91.8 (0.7)
            # 0.8 : {'AP': 10.233141788901346, 'FPR95': 34.280625041829545, 'AUROC': 92.85369451795876
            # 0.9: AP': 8.730725568077096, 'FPR95': 34.521605509233666, 'AUROC': 93.0510196 0.9

            # 1     0
            # -------------------------------
            final_soft_mask_voxel = 0.9 * soft_mask_rej + 0.1 * soft_mask_acc  # np.ndarray, shape: [N_voxel_points]
            #print("maskomaly before vocel inverse inference time------------------", (time.time() - start_maskomaly_uncertainity) * 1000)

            # 回填到 raw points
            inv_map_np = inv_map.cpu().numpy().astype(np.int64)
            if inv_map_np.max() >= final_soft_mask_voxel.shape[0]:
                raise IndexError(
                    f"inv_map max index {inv_map_np.max()} exceeds soft mask length {final_soft_mask_voxel.shape[0]}"
                )
            final_soft_mask = final_soft_mask_voxel[inv_map_np]
            total_maskomaly_uncertainty = (time.time() - start_maskomaly_uncertainity) * 1000


            print("maskomaly after vocel inference time------------------", total_maskomaly_uncertainty)
                                    # ------- 新增：把每一帧的时间记录下来 -------
            self.test_inference_times.append(float(inference_time))
            self.test_maskomaly_times.append(float(total_maskomaly_uncertainty))
                

            t3 = time.perf_counter()
            ano_time = t3 - t2
            self._ano_time_sum_frames += ano_time
            self._ano_time_frames += 1

            print(f"[Timing][Batch {batch_idx} | Frame {seq_name}/{frame_id}] "
                f"anomaly_scoring: {ano_time:.4f}s")
            # ----------------------------------------------------
            # END: anomaly scoring 计时
            # ----------------------------------------------------

            # ===== 下面保持你已有的存盘与语义预测保存逻辑 =====
            base_path = save_path / f"{seq_name}"
            base_path.mkdir(parents=True, exist_ok=True)

            try:
                frame_int = int(frame_id)
                frame_str = f"{frame_int:06d}"
            except Exception:
                frame_str = str(frame_id).zfill(6)

            # 语义预测保存（保持原样）
            confid = pred_masks[b_idx].float().sigmoid().matmul(pred_logits[b_idx])  # [N_voxel, C]
            voxel_sem_preds = torch.argmax(confid, dim=1)
            sem_preds = voxel_sem_preds[inverse_maps[b_idx]].cpu().numpy()
            sem_preds_path_txt = base_path / f"sem_preds_{frame_id}.txt"
            sem_preds_path_npy = base_path / f"sem_preds_{frame_id}.npy"
            np.savetxt(sem_preds_path_txt, sem_preds, fmt="%d")
            np.save(sem_preds_path_npy, sem_preds)

            # soft mask 保存（保持原样）
            txt_path = base_path / f"maskomaly_{frame_id}.txt"
            npy_path = base_path / f"maskomaly_{frame_id}.npy"
            np.savetxt(txt_path, final_soft_mask, fmt="%.6f")
            np.save(npy_path, final_soft_mask)

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
    #     # forward（batch 粒度）
    #     avg_fw_per_batch = (self._fw_time_sum_batches / self._fw_time_batches) if self._fw_time_batches > 0 else float('nan')
    #     # forward（帧粒度，按 batch 推理时间均分到每帧）
    #     avg_fw_per_frame = (self._fw_time_sum_frames / self._fw_time_frames) if self._fw_time_frames > 0 else float('nan')
    #     # anomaly scoring（帧粒度）
    #     avg_ano_per_frame = (self._ano_time_sum_frames / self._ano_time_frames) if self._ano_time_frames > 0 else float('nan')

    #     print("\n========== Inference Timing (Test) ==========")
    #     print(f"- Forward (per batch): {avg_fw_per_batch:.4f} s/batch")
    #     print(f"- Forward (per frame): {avg_fw_per_frame:.4f} s/frame (batch time均分)")
    #     print(f"- Anomaly scoring (per frame): {avg_ano_per_frame:.4f} s/frame")
    #     print("=============================================\n")

    #     # 也可用 self.log 写入 Lightning 日志
    #     try:
    #         self.log("test/forward_s_per_batch", avg_fw_per_batch, prog_bar=False)
    #         self.log("test/forward_s_per_frame", avg_fw_per_frame, prog_bar=True)
    #         self.log("test/anomaly_scoring_s_per_frame", avg_ano_per_frame, prog_bar=True)
    #     except Exception:
    #         pass
    def test_epoch_end(self, outputs):
        # 防止除零
        if len(self.test_inference_times) > 0:
            avg_inf = sum(self.test_inference_times) / len(self.test_inference_times)
            avg_maskomaly = sum(self.test_maskomaly_times) / len(self.test_maskomaly_times)

            print(f"\n=== Test Epoch Summary ===")
            print(f"Average pure inference time: {avg_inf:.2f} ms")
            print(f"Average maskomaly post-processing time: {avg_maskomaly:.2f} ms")

            # 如果你也想用 Lightning 的 logger 记录到日志里：
            self.log("test/avg_inference_time_ms", avg_inf)
            self.log("test/avg_maskomaly_time_ms", avg_maskomaly)


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

