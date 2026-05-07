import math

import hydra
import MinkowskiEngine.MinkowskiOps as me
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from pytorch3d.ops import sample_farthest_points
from torch.cuda.amp import autocast

from models.mask4former import Mask4Former3D


class Mask4Former3D_CalibFusion(Mask4Former3D):
    """
    Mask4Former3D with geometry-guided camera fusion.

    Instead of attending globally to all image patch tokens, each query's 3D
    position is projected onto the image using the LiDAR→image calibration
    matrix (calib_proj), and image features are bilinear-sampled at the
    projected location.  The sampled per-query feature is fused into the query
    via a per-decoder-level MLP after the LiDAR cross-attention step.

    The camera branch is skipped entirely when `images` or `calib_projs` is None.

    Args:
        img_size (tuple): (H, W) images are resized to.  Feature stride = 16,
                          so the feature map is (H//16, W//16).
        *args / **kwargs: forwarded to Mask4Former3D.__init__
    """

    def __init__(self, *args, img_size=(192, 640), **kwargs):
        super().__init__(*args, **kwargs)
        mask_dim = kwargs["mask_dim"]

        self.img_size = img_size          # (H, W) of the resized image

        # ── Image backbone: ResNet18 up to layer3 (stride=16, C_out=256) ─────
        resnet = tv_models.resnet18(pretrained=True)
        self.img_backbone = nn.Sequential(
            resnet.conv1,    # stride 2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # stride 2
            resnet.layer1,   # stride 1
            resnet.layer2,   # stride 2
            resnet.layer3,   # stride 2  → total stride 16
        )
        self.img_proj = nn.Conv2d(256, mask_dim, kernel_size=1)

        # ── Per-query image feature fusion (one MLP per decoder level) ───────
        # Fuses the sampled image feature into the query via a residual add.
        self.calib_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mask_dim, mask_dim),
                nn.ReLU(inplace=True),
                nn.Linear(mask_dim, mask_dim),
            )
            for _ in range(self.num_levels)
        ])

    # ─────────────────────────────────────────────────────────────────────────
    def _sample_image_features(
        self,
        sampled_coords: torch.Tensor,   # [B, Q, 3]  LiDAR world coords
        calib_projs: torch.Tensor,      # [B, 3, 4]  proj to resized-image pixels
        img_feats: torch.Tensor,        # [B, C, Hf, Wf]
    ):
        """
        For each query, project its 3D position to the image and bilinear-sample
        the feature at that location.

        Returns:
            sampled : [Q, B, C]   per-query image features (zero where invisible)
            vis_mask: [B, Q] bool  True = the query projects onto the image
        """
        B, Q, _ = sampled_coords.shape
        H_img, W_img = self.img_size

        # Homogeneous 3D coords  [B, Q, 4]
        ones  = torch.ones(B, Q, 1, device=sampled_coords.device, dtype=sampled_coords.dtype)
        pts_h = torch.cat([sampled_coords, ones], dim=-1)

        # Project: [B, Q, 4] @ [B, 4, 3] → [B, Q, 3]
        uvw = torch.bmm(pts_h, calib_projs.transpose(1, 2))

        depth = uvw[..., 2]                                     # [B, Q]
        in_front = depth > 0.1

        # Pixel coords in resized image
        uv = uvw[..., :2] / depth.clamp(min=1e-6).unsqueeze(-1)  # [B, Q, 2]

        # Normalise to [-1, 1] for F.grid_sample  (x=col, y=row)
        u_norm = (uv[..., 0] / W_img) * 2.0 - 1.0              # [B, Q]
        v_norm = (uv[..., 1] / H_img) * 2.0 - 1.0              # [B, Q]

        in_bounds = (u_norm >= -1) & (u_norm <= 1) & (v_norm >= -1) & (v_norm <= 1)
        vis_mask  = in_front & in_bounds                         # [B, Q]

        # F.grid_sample needs [B, H_out, W_out, 2] with last dim = (x, y) = (u, v)
        grid    = torch.stack([u_norm, v_norm], dim=-1)          # [B, Q, 2]
        grid    = grid[:, :, None, :]                            # [B, Q, 1, 2]
        sampled = F.grid_sample(
            img_feats.float(), grid.float(),
            mode="bilinear", align_corners=False, padding_mode="zeros",
        )                                                        # [B, C, Q, 1]
        sampled = sampled.squeeze(-1).permute(2, 0, 1)          # [Q, B, C]

        # Zero out features for queries that don't project onto the image
        vis_t   = vis_mask.permute(1, 0).unsqueeze(-1).float()  # [Q, B, 1]
        sampled = sampled * vis_t

        return sampled, vis_mask

    # ─────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        x,
        raw_coordinates=None,
        is_eval=False,
        images=None,
        calib_projs=None,
    ):
        """
        Args:
            x               : MinkowskiEngine SparseTensor (LiDAR voxels)
            raw_coordinates : [N_voxels, 4] raw xyz+time coords
            is_eval         : bool
            images          : [B, 3, H, W] float tensor, or None
            calib_projs     : [B, 3, 4]  projection matrices (resized px space), or None
        """
        device = x.device

        # ── Camera: extract image feature map ────────────────────────────────
        img_feats_map = None
        if images is not None and calib_projs is not None:
            img_f         = self.img_backbone(images.to(device))  # [B, 256, Hf, Wf]
            img_feats_map = self.img_proj(img_f)                   # [B, mask_dim, Hf, Wf]

        # ── LiDAR backbone ───────────────────────────────────────────────────
        all_features   = self.backbone(x)
        point_features = self.point_features_head(all_features[-1])

        with torch.no_grad():
            coordinates = me.SparseTensor(
                features=raw_coordinates, coordinates=x.C, device=device
            )
        pos_encodings_pcd = self.get_pos_encs(coordinates)

        # ── FPS query initialisation ──────────────────────────────────────────
        sampled_coords, mins, maxs = [], [], []
        for coords, feats in zip(x.decomposed_coordinates,
                                 coordinates.decomposed_features):
            _, fps_idx = sample_farthest_points(
                coords[None, ...].float(), K=self.num_queries
            )
            sampled_coords.append(feats[fps_idx.squeeze(0).long(), :3])
            mins.append(feats[:, :3].min(dim=0)[0])
            maxs.append(feats[:, :3].max(dim=0)[0])

        sampled_coords = torch.stack(sampled_coords)   # [B, Q, 3]
        mins           = torch.stack(mins)
        maxs           = torch.stack(maxs)

        query_pos = self.pos_enc(sampled_coords.float(), input_range=[mins, maxs])
        query_pos = self.query_projection(query_pos)

        queries   = torch.zeros_like(query_pos).permute((0, 2, 1))
        query_pos = query_pos.permute((2, 0, 1))

        # ── Project query coords → image, sample once before decoder loop ─────
        # sampled_img_feats: [Q, B, mask_dim]
        sampled_img_feats = None
        if img_feats_map is not None:
            sampled_img_feats, _ = self._sample_image_features(
                sampled_coords, calib_projs.to(device), img_feats_map
            )

        predictions_class, predictions_bbox, predictions_mask = [], [], []

        # ── Decoder loop ──────────────────────────────────────────────────────
        for _ in range(self.num_decoders):
            for hlevel in range(self.num_levels):
                output_class, outputs_bbox, outputs_mask, attn_mask = self.mask_module(
                    queries, point_features, self.num_levels - hlevel
                )

                decomposed_feat = all_features[hlevel].decomposed_features
                decomposed_attn = attn_mask.decomposed_features

                pcd_sizes        = [pcd.shape[0] for pcd in decomposed_feat]
                curr_sample_size = max(pcd_sizes)
                if not is_eval:
                    curr_sample_size = min(curr_sample_size, self.sample_sizes[hlevel])

                rand_idx, mask_idx = self.get_random_samples(
                    pcd_sizes, curr_sample_size, device
                )

                batched_feat    = torch.stack(
                    [feat[idx] for feat, idx in zip(decomposed_feat, rand_idx)]
                )
                batched_attn    = torch.stack(
                    [attn[idx] for attn, idx in zip(decomposed_attn, rand_idx)]
                )
                batched_pos_enc = torch.stack(
                    [pe[idx] for pe, idx in zip(pos_encodings_pcd[hlevel], rand_idx)]
                )

                batched_attn.permute((0, 2, 1))[
                    batched_attn.sum(1) == curr_sample_size
                ] = False
                m = torch.stack(mask_idx)
                batched_attn = torch.logical_or(batched_attn, m[..., None])

                src_pcd = self.lin_squeeze[hlevel](batched_feat.permute((1, 0, 2)))

                # 1. LiDAR cross-attention (unchanged)
                output = self.cross_attention[hlevel](
                    queries.permute((1, 0, 2)),
                    src_pcd,
                    memory_mask=batched_attn.repeat_interleave(
                        self.num_heads, dim=0
                    ).permute((0, 2, 1)),
                    memory_key_padding_mask=None,
                    pos=batched_pos_enc.permute((1, 0, 2)),
                    query_pos=query_pos,
                )

                # 2. Geometry-guided camera fusion (NEW)
                # Fuse per-query sampled image feature via residual MLP.
                if sampled_img_feats is not None:
                    output = output + self.calib_fusion[hlevel](sampled_img_feats)

                # 3. Self-attention
                output = self.self_attention[hlevel](
                    output,
                    tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_pos,
                )

                # 4. FFN
                queries = self.ffn_attention[hlevel](output).permute((1, 0, 2))

                predictions_class.append(output_class)
                predictions_bbox.append(outputs_bbox)
                predictions_mask.append(outputs_mask)

        output_class, outputs_bbox, outputs_mask = self.mask_module(
            queries, point_features
        )
        predictions_class.append(output_class)
        predictions_bbox.append(outputs_bbox)
        predictions_mask.append(outputs_mask)

        return {
            "pred_logits": predictions_class[-1],
            "pred_bboxs":  predictions_bbox[-1],
            "pred_masks":  predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class, predictions_bbox, predictions_mask
            ),
        }
