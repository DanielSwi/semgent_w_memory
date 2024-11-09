# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import numpy as np
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed
import torch.nn.functional as F

from torch.nn.init import trunc_normal_

from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer

from sam2.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames
from sam2.utils.misc import concat_points

from sam2.utils.new_utils import Sam2Output

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class SAM2ClassifierBase(SAM2Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward_sam_heads(
            self,
            backbone_features,
            point_inputs=None,
            mask_inputs=None,
            high_res_features=None,
            multimask_output=False,
    ):
        """
        Forward SAM prompt encoders and mask heads.

        Inputs:
        - backbone_features: image features of [B, C, H, W] shape
        - point_inputs: a dictionary with "point_coords" and "point_labels", where
          1) "point_coords" has [B, P, 2] shape and float32 dtype and contains the
             absolute pixel-unit coordinate in (x, y) format of the P input points
          2) "point_labels" has shape [B, P] and int32 dtype, where 1 means
             positive clicks, 0 means negative clicks, and -1 means padding
        - mask_inputs: a mask of [B, 1, H*16, W*16] shape, float or bool, with the
          same spatial size as the image.
        - high_res_features: either 1) None or 2) or a list of length 2 containing
          two feature maps of [B, C, 4*H, 4*W] and [B, C, 2*H, 2*W] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.
        - multimask_output: if it's True, we output 3 candidate masks and their 3
          corresponding IoU estimates, and if it's False, we output only 1 mask and
          its corresponding IoU estimate.

        Outputs:
        - low_res_multimasks: [B, M, H*4, W*4] shape (where M = 3 if
          `multimask_output=True` and M = 1 if `multimask_output=False`), the SAM
          output mask logits (before sigmoid) for the low-resolution masks, with 4x
          the resolution (1/4 stride) of the input backbone_features.
        - high_res_multimasks: [B, M, H*16, W*16] shape (where M = 3
          if `multimask_output=True` and M = 1 if `multimask_output=False`),
          upsampled from the low-resolution masks, with shape size as the image
          (stride is 1 pixel).
        - ious, [B, M] shape, where (where M = 3 if `multimask_output=True` and M = 1
          if `multimask_output=False`), the estimated IoU of each output mask.
        - low_res_masks: [B, 1, H*4, W*4] shape, the best mask in `low_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `low_res_multimasks`.
        - high_res_masks: [B, 1, H*16, W*16] shape, the best mask in `high_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `high_res_multimasks`.
        - obj_ptr: [B, C] shape, the object pointer vector for the output mask, extracted
          based on the output token from the SAM mask decoder.
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            # low_res_multimasks = torch.where(
            #     is_obj_appearing[:, None, None],
            #     low_res_multimasks,
            #     NO_OBJ_SCORE,
            # )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        # if self.pred_obj_scores:
        #     # Allow *soft* no obj ptr, unlike for masks
        #     if self.soft_no_obj_ptr:
        #         # Only hard possible with gt
        #         assert not self.teacher_force_obj_scores_for_mem
        #         lambda_is_obj_appearing = object_score_logits.sigmoid()
        #     else:
        #         lambda_is_obj_appearing = is_obj_appearing.float()
        #
        #     if self.fixed_no_obj_ptr:
        #         obj_ptr = lambda_is_obj_appearing * obj_ptr
        #     obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )


    def track_step(
            self,
            current_vision_feats: list,
            current_vision_pos_embeds: list,
            feat_sizes: list,
            memory: torch.Tensor,
            num_obj_ptr_tokens: int,
            points: torch.Tensor,
            labels: torch.Tensor,
            box: torch.Tensor,
            original_shape: tuple,
            device: str,
            run_mem_encoder: bool
    ):
        '''
        This track step will work with a memory tensor 
        :param current_vision_feats:
        :param current_vision_pos_embeds:
        :param feat_sizes:
        :param memory:
        :param num_obj_ptr_tokens:
        :return:
        '''
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]

        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        if memory is None:
            pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)

            if points is None:
                points = torch.zeros(0, 2, dtype=torch.float32, device=device)
            elif not isinstance(points, torch.Tensor):
                points = torch.tensor(points, dtype=torch.float32, device=device)
            if labels is None:
                labels = torch.zeros(0, dtype=torch.int32, device=device)
            elif not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.int32, device=device)
            if points.dim() == 2:
                points = points.unsqueeze(0)  # add batch dimension
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)  # add batch dimension

            # If `box` is provided, we add it as the first two points with labels 2 and 3
            # along with the user-provided points (consistent with how SAM 2 is trained).
            if box is not None:
                if not isinstance(box, torch.Tensor):
                    box = torch.tensor(box, dtype=torch.float32, device=device)
                box_coords = box.reshape(1, 2, 2)
                box_labels = torch.tensor([2, 3], dtype=torch.int32, device=device)
                box_labels = box_labels.reshape(1, 2)
                points = torch.cat([box_coords, points], dim=1)
                labels = torch.cat([box_labels, labels], dim=1)
            video_H, video_W = original_shape
            points = points / torch.tensor([video_W, video_H]).to(points.device)
            # scale the (normalized) coordinates by the model's internal image size
            points = points * self.image_size

            point_inputs = concat_points(None, points, labels)
            mask_inputs = None

        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat_with_mem = self._prepare_memory_conditioned_features(
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                memory=memory,
                num_obj_ptr_tokens=num_obj_ptr_tokens
            )
            point_inputs = None
            mask_inputs = None

        sam_outputs = self._forward_sam_heads(
            backbone_features=pix_feat_with_mem,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            high_res_features=high_res_features,
            multimask_output=False,
        )
        (
            _,
            _,
            _,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        if run_mem_encoder:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                is_mask_from_pts=False,
            )
        else:
            maskmem_features, maskmem_pos_enc = None, None
        return Sam2Output(high_res_masks=high_res_masks, obj_ptr=obj_ptr, object_score_logits=object_score_logits,
                          maskmem_features=maskmem_features, maskmem_pos_enc=maskmem_pos_enc)

    def _prepare_memory_conditioned_features(
            self,
            current_vision_feats: list,
            current_vision_pos_embeds: list,
            feat_sizes: list,
            memory: torch.Tensor,
            num_obj_ptr_tokens: int
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=torch.zeros(memory.shape).to(device),
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        # reshape the output (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem
