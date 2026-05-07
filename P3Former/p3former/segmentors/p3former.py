
import os
from pathlib import Path

exit(1)

import numpy as np
import torch
import torch.nn.functional as F
from mmdet3d.models.segmentors.cylinder3d import Cylinder3D
from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class _P3Former(Cylinder3D):
    """P3Former."""

    def __init__(self,
                 voxel_encoder: ConfigType,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 loss_regularization: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(voxel_encoder=voxel_encoder,
                        backbone=backbone,
                        decode_head=decode_head,
                        neck=neck,
                        auxiliary_head=auxiliary_head,
                        loss_regularization=loss_regularization,
                        train_cfg=train_cfg,
                        test_cfg=test_cfg,
                        data_preprocessor=data_preprocessor,
                        init_cfg=init_cfg)

    def loss(self, batch_inputs_dict,batch_data_samples):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        # extract features using backbone
        x = self.extract_feat(batch_inputs_dict)
        batch_inputs_dict['features'] = x.features
        losses = dict()
        loss_decode = self._decode_head_forward_train(batch_inputs_dict, batch_data_samples)
        losses.update(loss_decode)

        return losses
    
    def normalize(self, x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    def save_uncertainty_scores(self,
        prediction_path,
        seq_id,
        frame_name,
        sem_pred_labels,
        msp_3d,
        energy,
        maxlogit,
        entropy
    ):
        """
        Save uncertainty scores for all methods in .txt format
        """
        dirs = {
            "maxlogit": os.path.join(prediction_path, "maxlogit", seq_id),
            "msp": os.path.join(prediction_path, "msp", seq_id),
            "entropy": os.path.join(prediction_path, "entropy", seq_id),
            "energy": os.path.join(prediction_path, "energy", seq_id)}
        
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        os.makedirs(os.path.join(prediction_path, "predictions", seq_id), exist_ok=True)


        #frame_int = int(frame_id)
        frame_str =  frame_name.split(".")[0] #f"{frame_int:06d}"

        def save(name, data_3d):
            np_data = data_3d.detach().cpu().numpy() # .astype(np.float32)
            #np.save(os.path.join(dirs[name], f"{name}_{frame_str}.npy"), np_data)
            np.savetxt(
                os.path.join(dirs[name], f"{name}_{frame_str}.txt"),
                np_data,#.flatten(),
                fmt="%.6f",
            )
            
        # pred_np_origin.tofile(os.path.join(dirs[name], f"{frame_str}.label"))
        np.savetxt(os.path.join(prediction_path, "predictions", seq_id, f"{frame_str}.txt"), sem_pred_labels.astype(np.int32), fmt="%d")

        save("maxlogit", maxlogit)
        save("msp", msp_3d)
        save("entropy", entropy)
        save("energy", energy)

        print(f"Saved uncertainties for {seq_id}/{frame_str}")

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):

        x = self.extract_feat(batch_inputs_dict)
        batch_inputs_dict['features'] = x.features
        pts_semantic_preds, pts_instance_preds, confids = self.decode_head.predict(batch_inputs_dict, batch_data_samples)

        # ---- Anomaly Scoring ----
        for batch_idx in range(len(batch_data_samples)):
            confid = confids[batch_idx]  # [V, 19]
            # anomaly scores at voxel level
            point_probs = F.softmax(confid, dim=1)   # [V, 19]

            msp       = 1.0 - point_probs.max(dim=1).values                             # [V]
            max_logit = -confid.max(dim=1).values                                       # [V]
            energy    = -torch.logsumexp(confid, dim=1)                                 # [V]
            entropy   = -torch.sum(point_probs * torch.log(point_probs + 1e-8), dim=1) # [V]

            # map voxel scores to original points
            point2voxel_map = batch_data_samples[
                batch_idx].gt_pts_seg.point2voxel_map.long()

            msp_pts       = msp[point2voxel_map]        # [N]
            max_logit_pts = max_logit[point2voxel_map]  # [N]
            energy_pts    = energy[point2voxel_map]     # [N]
            entropy_pts   = entropy[point2voxel_map]    # [N]

            max_logit_pts = self.normalize(max_logit_pts)
            energy_pts = self.normalize(energy_pts)
            entropy_pts = self.normalize(entropy_pts)

            pred_labels = pts_semantic_preds[batch_idx]

            meta = batch_data_samples[0].metainfo
            lidar_path = meta['lidar_path']
            seq_id  = lidar_path.split('/')[-3]
            frame_name =  lidar_path.split('/')[-1].split('.')[-2]+'.label'

            # change
            prediction_save_path = str(print(Path(__file__).parents[2]))

            self.save_uncertainty_scores(
                    prediction_path =prediction_save_path,
                    seq_id=seq_id,
                    frame_name=frame_name,
                    sem_pred_labels=pred_labels,
                    msp_3d=msp_pts,
                    energy=energy_pts,
                    entropy=entropy_pts,
                    maxlogit=max_logit_pts
                )
    
        # ANOMALY SCORING END



        return self.postprocess_result(pts_semantic_preds, pts_instance_preds, batch_data_samples)

    def postprocess_result(self, pts_semantic_preds, pts_instance_preds, batch_data_samples):
        for i in range(len(pts_semantic_preds)):
            batch_data_samples[i].set_data(
                {'pred_pts_seg': PointData(**{'pts_semantic_mask': pts_semantic_preds[i],
                                                'pts_instance_mask': pts_instance_preds[i]})})
        return batch_data_samples
