"""
Camera-aware trainer: extends PanopticSegmentation (pq_trainer.py) to pass
batched camera images to the Mask4Former3D_Camera model in every step.

Only the three step methods and `forward` are overridden — everything else
(loss, metrics, optimizers, dataloaders, epoch hooks) is inherited unchanged.
"""
import MinkowskiEngine as ME

from trainer.pq_trainer import PanopticSegmentation


class PanopticSegmentationCamera(PanopticSegmentation):

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x, raw_coordinates=None, is_eval=False, images=None):
        with self.optional_freeze():
            x = self.model(
                x,
                raw_coordinates=raw_coordinates,
                is_eval=is_eval,
                images=images,
            )
        return x

    # ─────────────────────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        import statistics

        data, target = batch

        raw_coordinates = data.raw_coordinates
        _imgs = getattr(data, "images", None)
        images = _imgs.to(self.device) if _imgs is not None else None

        data = ME.SparseTensor(
            coordinates=data.coordinates, features=data.features, device=self.device
        )

        output = self.forward(
            data, raw_coordinates=raw_coordinates, images=images
        )
        losses = self.criterion(output, target)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                losses.pop(k)

        logs = {f"train_{k}": v.detach().cpu().item() for k, v in losses.items()}
        logs["train_mean_loss_ce"]   = statistics.mean(
            [v for k, v in logs.items() if "loss_ce"   in k]
        )
        logs["train_mean_loss_mask"] = statistics.mean(
            [v for k, v in logs.items() if "loss_mask" in k]
        )
        logs["train_mean_loss_dice"] = statistics.mean(
            [v for k, v in logs.items() if "loss_dice" in k]
        )
        logs["train_mean_loss_box"]  = statistics.mean(
            [v for k, v in logs.items() if "loss_box"  in k]
        )
        self.log_dict(logs)
        return sum(losses.values())

    # ─────────────────────────────────────────────────────────────────────────
    def validation_step(self, batch, batch_idx):
        data, target = batch

        inverse_maps    = data.inverse_maps
        original_labels = data.original_labels
        raw_coordinates = data.raw_coordinates
        num_points      = data.num_points
        sequences       = data.sequences
        _imgs = getattr(data, "images", None)
        images = _imgs.to(self.device) if _imgs is not None else None

        data = ME.SparseTensor(
            coordinates=data.coordinates, features=data.features, device=self.device
        )
        output = self.forward(
            data, raw_coordinates=raw_coordinates, is_eval=True, images=images
        )

        # ── delegate the rest to the parent's validation_step logic ──────────
        # Re-use by constructing a fake "processed" batch tuple the parent
        # expects; because the parent's validation_step is long and accesses
        # local variables we replicate it here via super()-delegation trick:
        # Instead, we duplicate the parent post-processing inline.
        import numpy as np
        import torch
        from utils.utils import associate_instances

        losses = self.criterion(output, target)
        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                losses.pop(k)

        pred_logits = output["pred_logits"]
        pred_logits = torch.functional.F.softmax(pred_logits, dim=-1)[..., :-1]
        pred_masks  = output["pred_masks"]
        offset_coords_idx = 0

        for logit, mask, inv_map, label, n_point, seq in zip(
            pred_logits, pred_masks,
            inverse_maps, original_labels, num_points, sequences,
        ):
            seq = seq[0]
            if seq != self.last_seq:
                self.last_seq          = seq
                self.previous_instances = None
                self.max_instance_id   = self.config.model.num_queries
                self.scene             = 0

            class_confidence    = torch.max(logit.detach().cpu(), dim=1)[0]
            foreground_confidence = mask.detach().cpu().float().sigmoid()
            confidence          = class_confidence[None, ...] * foreground_confidence
            confidence          = confidence[inv_map].numpy()

            ins_preds  = np.argmax(confidence, axis=1)
            sem_preds  = logit.detach().cpu().argmax(1)[ins_preds].numpy() + 1
            ins_preds += 1
            ins_preds[
                np.isin(
                    sem_preds,
                    range(1, self.config.data.min_stuff_cls_id),
                    invert=True,
                )
            ] = 0

            sem_labels = self.validation_dataset._remap_model_output(label[:, 0])
            ins_labels = label[:, 1] >> 16

            db_max_instance_id = self.config.model.num_queries
            if self.config.general.dbscan_eps is not None:
                from sklearn.cluster import DBSCAN
                curr_coords_idx = mask.shape[0]
                curr_coords = raw_coordinates[
                    offset_coords_idx: curr_coords_idx + offset_coords_idx, :3
                ]
                curr_coords = curr_coords[inv_map].detach().cpu().numpy()
                offset_coords_idx += curr_coords_idx
                for ins_id in np.unique(ins_preds):
                    if ins_id != 0:
                        instance_mask = ins_preds == ins_id
                        clusters = (
                            DBSCAN(
                                eps=self.config.general.dbscan_eps,
                                min_samples=1, n_jobs=-1,
                            ).fit(curr_coords[instance_mask]).labels_
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
                    for id_ in np.unique(ins_preds):
                        if associations.get(id_) is None:
                            self.max_instance_id += 1
                            associations[id_] = self.max_instance_id
                    ins_preds = np.vectorize(associations.__getitem__)(ins_preds)
                else:
                    self.class_evaluator.addBatch(
                        sem_preds, ins_preds, sem_labels, ins_labels, indices, seq
                    )
            if i > 0:
                self.previous_instances = ins_preds[indices]

        return {f"val_{k}": v.detach().cpu().item() for k, v in losses.items()}

    # ─────────────────────────────────────────────────────────────────────────
    def test_step(self, batch, batch_idx):
        import numpy as np
        import torch
        from pathlib import Path

        data, target = batch

        inverse_maps = data.inverse_maps
        raw_coordinates = data.raw_coordinates
        sequences = data.sequences
        _imgs = getattr(data, "images", None)
        images = _imgs.to(self.device) if _imgs is not None else None

        data = ME.SparseTensor(
            coordinates=data.coordinates, features=data.features, device=self.device
        )
        output = self.forward(
            data, raw_coordinates=raw_coordinates, is_eval=True, images=images
        )

        def get_rba(logit, mask, inv_map):
            confid = mask.float().sigmoid().matmul(logit)
            rba = -confid.tanh().sum(dim=1)[inv_map]
            rba[rba < -1] = -1
            rba = rba + 1
            return rba

        pred_logits = output["pred_logits"]
        pred_logits = torch.functional.F.softmax(pred_logits, dim=-1)[..., :-1]
        pred_masks  = output["pred_masks"]

        save_path = Path(self.config.general.save_dir) / "prediction"
        for b_idx, (logit, mask, inv_map) in enumerate(
            zip(pred_logits, pred_masks, inverse_maps)
        ):
            rba_score = get_rba(logit, mask, inv_map)
            base_path = save_path / f"{sequences[b_idx][0]}"
            rba_save_path = base_path / f"{sequences[b_idx][1]}.txt"
            if not rba_save_path.parent.exists():
                rba_save_path.parent.mkdir(exist_ok=True, parents=True)
            np.savetxt(rba_save_path, rba_score.detach().cpu().numpy())

        return {}
