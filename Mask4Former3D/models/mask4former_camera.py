import math

import hydra
import MinkowskiEngine.MinkowskiOps as me
import torch
import torch.nn as nn
import torchvision.models as tv_models
from pytorch3d.ops import sample_farthest_points
from torch.cuda.amp import autocast

from models.mask4former import Mask4Former3D
from models.modules.attention import CrossAttentionLayer


class Mask4Former3D_Camera(Mask4Former3D):
    """
    Mask4Former3D with an additional camera cross-attention branch in the decoder.

    Decoder loop (per decoder-iteration × level):
        1. LiDAR cross-attention   (existing, unchanged)
        2. Camera cross-attention  (NEW — queries attend to image patch tokens)
        3. Self-attention          (existing, unchanged)
        4. FFN                     (existing, unchanged)

    The camera branch is optional at runtime: if `images` is None the model
    behaves identically to the base Mask4Former3D.

    Args:
        img_size (tuple): (H, W) images are resized to before encoding.
                          Feature map will be (H//8, W//8).
        *args / **kwargs: forwarded to Mask4Former3D.__init__
    """

    def __init__(self, *args, img_size=(192, 640), **kwargs):
        super().__init__(*args, **kwargs)
        mask_dim = kwargs["mask_dim"]

        self.img_size = img_size
        h_feat = img_size[0] // 16   # 12 for img_size=(192,640)
        w_feat = img_size[1] // 16   # 40

        # ── Image backbone: ResNet18 up to layer3 (stride=16, C_out=256) ─────
        # Stride breakdown: conv1(s2) → maxpool(s2) → layer1(s1) → layer2(s2) → layer3(s2)
        # Total stride = 2*2*1*2*2 = 16
        resnet = tv_models.resnet18(pretrained=True)
        self.img_backbone = nn.Sequential(
            resnet.conv1,    # stride 2  (total s=2)
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # stride 2  (total s=4)
            resnet.layer1,   # stride 1  (total s=4)
            resnet.layer2,   # stride 2  (total s=8)
            resnet.layer3,   # stride 2  (total s=16, C=256)
        )
        self.img_proj = nn.Conv2d(256, mask_dim, kernel_size=1)

        # ── 2D sinusoidal positional encoding (fixed, pre-computed) ──────────
        # Shape: [h_feat * w_feat, 1, mask_dim]
        self.register_buffer(
            "img_pos_enc",
            self._make_2d_sine_pos_enc(h_feat, w_feat, mask_dim),
        )

        # ── Camera cross-attention (one layer per decoder level) ─────────────
        self.img_cross_attention = nn.ModuleList([
            CrossAttentionLayer(d_model=mask_dim, nhead=self.num_heads)
            for _ in range(self.num_levels)
        ])

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _make_2d_sine_pos_enc(h: int, w: int, d_model: int) -> torch.Tensor:
        """
        2D sine-cosine positional encoding for a (h, w) feature map.
        Returns tensor of shape [h*w, 1, d_model].
        """
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D sine PE"
        d = d_model // 4

        y_pos = torch.arange(h, dtype=torch.float).unsqueeze(1) / h  # [h, 1]
        x_pos = torch.arange(w, dtype=torch.float).unsqueeze(1) / w  # [w, 1]
        div   = torch.exp(torch.arange(0, d, dtype=torch.float) *
                          -(math.log(10000.0) / d))                   # [d]

        y_enc = torch.cat([torch.sin(y_pos * div),
                           torch.cos(y_pos * div)], dim=1)            # [h, 2d]
        x_enc = torch.cat([torch.sin(x_pos * div),
                           torch.cos(x_pos * div)], dim=1)            # [w, 2d]

        y_enc = y_enc.unsqueeze(1).expand(-1, w, -1)   # [h, w, 2d]
        x_enc = x_enc.unsqueeze(0).expand(h, -1, -1)   # [h, w, 2d]

        pos = torch.cat([y_enc, x_enc], dim=-1)         # [h, w, d_model]
        return pos.view(h * w, 1, d_model)              # [h*w, 1, d_model]

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x, raw_coordinates=None, is_eval=False, images=None):
        """
        Args:
            x               : MinkowskiEngine SparseTensor (LiDAR voxels)
            raw_coordinates : [N_voxels, 4] raw xyz+time coords
            is_eval         : bool
            images          : [B, 3, H, W] float tensor, or None
        """
        device = x.device

        # ── Camera: extract image tokens ─────────────────────────────────────
        img_tokens = None
        img_pos    = None
        if images is not None:
            img_feats  = self.img_backbone(images.to(device))   # [B, 256, Hf, Wf]
            img_feats  = self.img_proj(img_feats)               # [B, mask_dim, Hf, Wf]
            B = img_feats.shape[0]
            img_tokens = img_feats.flatten(2).permute(2, 0, 1) # [Hf*Wf, B, mask_dim]
            img_pos    = self.img_pos_enc.expand(-1, B, -1)    # [Hf*Wf, B, mask_dim]

        # ── LiDAR backbone ───────────────────────────────────────────────────
        all_features  = self.backbone(x)
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

        sampled_coords = torch.stack(sampled_coords)
        mins           = torch.stack(mins)
        maxs           = torch.stack(maxs)

        query_pos = self.pos_enc(sampled_coords.float(), input_range=[mins, maxs])
        query_pos = self.query_projection(query_pos)

        queries   = torch.zeros_like(query_pos).permute((0, 2, 1))
        query_pos = query_pos.permute((2, 0, 1))

        predictions_class, predictions_bbox, predictions_mask = [], [], []

        # ── Decoder loop ──────────────────────────────────────────────────────
        for _ in range(self.num_decoders):
            for hlevel in range(self.num_levels):
                output_class, outputs_bbox, outputs_mask, attn_mask = self.mask_module(
                    queries, point_features, self.num_levels - hlevel
                )

                decomposed_feat = all_features[hlevel].decomposed_features
                decomposed_attn = attn_mask.decomposed_features

                pcd_sizes       = [pcd.shape[0] for pcd in decomposed_feat]
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

                # 2. Camera cross-attention (NEW)
                if img_tokens is not None:
                    output = self.img_cross_attention[hlevel](
                        output,
                        img_tokens,
                        memory_mask=None,
                        memory_key_padding_mask=None,
                        pos=img_pos,
                        query_pos=query_pos,
                    )

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
