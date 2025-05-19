from typing import Optional, Tuple, Union, Dict
import numpy as np
from dataclasses import dataclass
from functools import partial, reduce
from PIL import Image
import torch
import torch.utils.checkpoint
from torch import nn
import os
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from transformers.utils import ModelOutput
from llava.utils import rank0_print
import math
import re
import time
import torch
import torch.nn as nn

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random

outputs_prumerge_plus = {}


def siglip_attention_forward( self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    if self.layer_idx == 24:    
        self.attn_weights = None

    batch_size, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    raw_key_states = key_states.float().clone()           
    value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    k_v_seq_len = key_states.shape[-2]
    attn_weights = torch.matmul(query_states.float(), key_states.transpose(2, 3).float()) * self.scale 

    if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
        raise ValueError(f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is" f" {attn_weights.size()}")

    if attention_mask is not None:
        if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
            raise ValueError(f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}")
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)

    if self.layer_idx == 24:             
        self.attn_weights = attn_weights
    
    attn_weights = attn_weights.to(query_states.dtype)     
    attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
        raise ValueError(f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is" f" {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights, raw_key_states.mean(1)


def siglip_EncoderLayer_forward(self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        if self.layer_idx == 24:
            self.metric = None
            self.hidden_states = None

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights, metric = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)


        if self.layer_idx == 24:   

            self.metric = metric
            self.hidden_states = hidden_states          

        return outputs


def siglip_vision_tower_forward(self, images):
    if type(images) is list:
        raise NotImplementedError("VisionZip does not support batch inference")

    else:
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=False, output_attentions=False)
        attn_weights = self.vision_tower.vision_model.encoder.layers[-2].self_attn.attn_weights    
        hidden_states = self.vision_tower.vision_model.encoder.layers[-2].hidden_states
        metric = self.vision_tower.vision_model.encoder.layers[-2].metric   # [3, 729, 72]
        per_patch_token_num = metric.shape[1]
        dominant_num =  max(1, int(self.budgets * per_patch_token_num * 5.4 / 6.4))
        contextual_num = max(1, int(self.budgets * per_patch_token_num * 1 / 6.4))

        ## Dominant Visual Tokens
        attention_sum = attn_weights.mean(dim=1).mean(dim=1)  
        all_indices = attention_sum.topk(dominant_num, dim=1).indices

        mask = torch.ones_like(hidden_states[:, :, 0], dtype=torch.bool, device=metric.device).scatter_(1, all_indices, False) 

        dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(hidden_states.shape[0], dominant_num, hidden_states.shape[2])
        
        ### Filter
        metric_filtered = metric[mask].view(hidden_states.shape[0], hidden_states.shape[1] - dominant_num, metric.shape[2])  
        hidden_states_filtered = hidden_states.masked_select(mask.unsqueeze(-1)).view(hidden_states.shape[0], hidden_states.shape[1] - dominant_num, hidden_states.shape[2])  # [3, 675, 1152]
        metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True) 

        ## Contextual Visual Tokens
        step = max(1, metric_normalized.shape[1] // contextual_num)

        target_indices = torch.arange(0, metric_normalized.shape[1], step, device=metric_normalized.device)[:contextual_num] 
        target_tokens = metric_normalized[:, target_indices, :]  


        tokens_to_merge = metric_normalized[:, ~torch.isin(torch.arange(metric_normalized.shape[1], device=metric_normalized.device), target_indices), :] 
        similarity = torch.bmm(tokens_to_merge.float(), target_tokens.transpose(1, 2).float()) 
        assign_one_hot = torch.zeros(tokens_to_merge.shape[0], tokens_to_merge.shape[1], contextual_num, dtype=hidden_states_filtered.dtype, device=metric_normalized.device)
        assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)

        counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)      

        hidden_to_merge = hidden_states_filtered[:, ~torch.isin(torch.arange(hidden_states_filtered.shape[1], device=hidden_states_filtered.device), target_indices), :]
        aggregated_hidden = (torch.bmm(assign_one_hot.transpose(1, 2).float(), hidden_to_merge.float()) / counts).to(torch.bfloat16) 
        target_hidden = hidden_states_filtered[:, target_indices, :]  
        contextual_tokens = target_hidden + aggregated_hidden

        # Merge with target hidden states and concatenate
        hidden_states_save = torch.cat([dominant_tokens, contextual_tokens], dim=1).to(images.dtype)  

    return hidden_states_save, all_indices

def encode_images_visionzip(self, images):
    image_features, keep_idx = self.get_model().get_vision_tower()(images)
    # image_features = self.get_model().vision_resampler(image_features, images=images)
    image_features = self.get_model().mm_projector(image_features)
    return image_features, keep_idx

def encode_images_visionzip_simple(self, images):
    image_features, keep_idx = self.get_model().get_vision_tower()(images)
    # image_features = self.get_model().vision_resampler(image_features, images=images)
    image_features = self.get_model().mm_projector(image_features)
    return image_features


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor

def prepare_inputs_labels_for_multimodal_visionzip(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):
    vision_tower = self.get_vision_tower()
    # rank_print(modalities)
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels

    if isinstance(modalities, str):
        modalities = [modalities]

    # import pdb; pdb.set_trace()
    if type(images) is list or images.ndim == 5:
        if type(images) is list:
            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

        video_idx_in_batch = []
        for _ in range(len(modalities)):
            if modalities[_] == "video":
                video_idx_in_batch.append(_)

        images_list = []
        for image in images:
            if image.ndim == 4:
                images_list.append(image)
            else:
                images_list.append(image.unsqueeze(0))

        concat_images = torch.cat([image for image in images_list], dim=0)
        split_sizes = [image.shape[0] for image in images_list]   # num_patches for every img
        encoded_image_features, keep_idx = self.encode_images_visionzip(concat_images)    
        

        encoded_image_features = torch.split(encoded_image_features, split_sizes)
        keep_idx = torch.split(keep_idx, split_sizes)
        image_features = []
        contextual_features = []
        for idx, image_feat in enumerate(encoded_image_features):   
            if idx in video_idx_in_batch:
                image_features.append(self.get_2dPool(image_feat))
            else:
                current_keep_idx = keep_idx[idx]
                num_patches = image_feat.shape[0]
                total_token_num = image_feat.shape[1]
                hidden_size = image_feat.shape[2]
                dominant_num = current_keep_idx.shape[1]
                contextual_num = total_token_num - dominant_num
                contextual_feature = image_feat[:, -contextual_num:, :]
                dominant_feature = image_feat[:, :dominant_num, :]
                contextual_features.append(contextual_feature.flatten(0,1))

                cur_keep_idx_sorted_restore , _ = current_keep_idx.sort(dim=1)
                fake_image = torch.zeros((num_patches, 729, hidden_size), device = image_feat.device, dtype = image_feat.dtype)    # make a fake image with the same size
                
                mask = torch.zeros(num_patches, 729, dtype=torch.bool, device=image_feat.device)
                mask.scatter_(1, cur_keep_idx_sorted_restore, True) 

                fake_image[mask] = dominant_feature.reshape(-1, hidden_size)  

                image_features.append(fake_image)  


        mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")   # spatial_unpad
        image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")   # anyres_max_9
        mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")   # one_token

        if mm_patch_merge_type == "flat":  
            image_features = [x.flatten(0, 1) for x in image_features]

        elif mm_patch_merge_type.startswith("spatial"):
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                # FIXME: now assume the image is square, and split to 2x2 patches
                # num_patches = h * w, where h = w = sqrt(num_patches)
                # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                # we want to first unflatten it to (2, 2, h, w, hidden_size)
                # rank0_print("At least we are reaching here")
                # import pdb; pdb.set_trace()
                if image_idx in video_idx_in_batch:  # video operations
                    # rank0_print("Video")
                    if mm_newline_position == "grid":
                        # Grid-wise
                        image_feature = self.add_token_per_grid(image_feature)
                        if getattr(self.config, "add_faster_video", False):
                            faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                            # Add a token for each frame
                            concat_slow_fater_token = []
                            # import pdb; pdb.set_trace()
                            for _ in range(image_feature.shape[0]):
                                if _ % self.config.faster_token_stride == 0:
                                    concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                else:
                                    concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                            # import pdb; pdb.set_trace()
                            image_feature = torch.cat(concat_slow_fater_token)
                    
                        new_image_features.append(image_feature)
                    elif mm_newline_position == "frame":
                        # Frame-wise
                        image_feature = self.add_token_per_frame(image_feature)
                        new_image_features.append(image_feature.flatten(0, 1))
                        
                    elif mm_newline_position == "one_token":
                        # one-token
                        image_feature = image_feature.flatten(0, 1)
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                        new_image_features.append(image_feature)      
                    elif mm_newline_position == "no_token":
                        new_image_features.append(image_feature.flatten(0, 1))
                    else:
                        raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                    # rank0_print("Single-images")
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]
                    height = width = self.get_vision_tower().num_patches_per_side  # 27
                    assert height * width == base_image_feature.shape[0]

                    if "anyres_max" in image_aspect_ratio:
                        matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                        if matched_anyres_max_num_patches:
                            max_num_patches = int(matched_anyres_max_num_patches.group(1))   # 9

                    if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                        if hasattr(self.get_vision_tower(), "image_size"):
                            vision_tower_image_size = self.get_vision_tower().image_size  # 384
                        else:
                            raise ValueError("vision_tower_image_size is not found in the vision tower.")
                        try:
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)   # 3, 3
                        except Exception as e:
                            rank0_print(f"Error: {e}")
                            num_patch_width, num_patch_height = 2, 2
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)   
                    else:
                        image_feature = image_feature.view(2, 2, height, width, -1)

                    if "maxpool2x2" in mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = nn.functional.max_pool2d(image_feature, 2)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                        unit = image_feature.shape[2]
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        c, h, w = image_feature.shape
                        times = math.sqrt(h * w / (max_num_patches * unit**2))
                        if times > 1.1:
                            image_feature = image_feature[None]
                            image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    elif "unpad" in mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    else:
                        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                        image_feature = image_feature.flatten(0, 3)
                    if "nobase" in mm_patch_merge_type:
                        pass
                    else:
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        mask_here = (image_feature != 0.0).any(dim=1)
                        image_feature = image_feature[mask_here]
                        image_feature = torch.cat((image_feature, contextual_features[image_idx]), dim=0)
                    new_image_features.append(image_feature)
                else:  # single image operations
                    image_feature = image_feature[0]
                    if "unpad" in mm_patch_merge_type:
                        image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                    new_image_features.append(image_feature)
            image_features = new_image_features
        else:
            raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
    else:
        image_features = self.encode_images_visionzip_simple(images)

    # TODO: image start / end is not implemented here to support pretraining.
    if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
        raise NotImplementedError
    # rank_print(f"Total images : {len(image_features)}")

    # Let's just add dummy tensors if they do not exist,
    # it is a headache to deal with None all the time.
    # But it is not ideal, and if you have a better idea,
    # please open an issue / submit a PR, thanks.
    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # remove the padding using attention_mask -- FIXME
    _input_ids = input_ids
    input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0

    # rank_print("Inserting Images embedding")
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        # rank0_print(num_images)
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        cur_new_input_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                try:
                    cur_image_features = image_features[cur_image_idx]
                except IndexError:
                    cur_image_features = image_features[cur_image_idx - 1]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

        cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

        # import pdb; pdb.set_trace()
        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)

    # Truncate sequences to max length as image embeddings can make the sequence longer
    tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
    # rank_print("Finishing Inserting")

    new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
    new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
    # TODO: Hard code for control loss spike
    # if tokenizer_model_max_length is not None:
    #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
    #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

    # Combine them
    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
    # rank0_print("Prepare pos id")

    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        if getattr(self.config, "tokenizer_padding_side", "right") == "left":
            new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
            if cur_len > 0:
                new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
        else:
            new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
    # rank0_print("tokenizer padding")

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None
    if getattr(self.config, "use_pos_skipping", False) and self.training:
        position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
        split_position = random.randint(0, new_input_embeds.size(1))
        left_add = random.randint(0, self.config.pos_skipping_range)
        right_add = random.randint(left_add, self.config.pos_skipping_range)
        position_ids[:, :split_position] += left_add
        position_ids[:, split_position:] += right_add
    # import pdb; pdb.set_trace()
    # rank0_print("Finish preparing")
    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels  


def siglip_vision_tower_forward_prumerge_plus(self, images , multi_images):
    if type(images) is list:
        raise NotImplementedError("Prumerge+ does not support batch inference")
    else:
        # token_prune_merge_advanced_plus
        image_features = token_prune_merge_advanced_prumerge_plus(self,images, multi_images=multi_images, if_adaptive=True) 

    return image_features

def complement_idx_prumerge_plus(idx, dim):
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl

def hook_k_prumerge_plus(module, input, output):
    outputs_prumerge_plus['desired_k'] = output

def hook_q_prumerge_plus(module, input, output):
    outputs_prumerge_plus['desired_q'] = output

def outlier_dectection_prumerge_plus(attn):

    attn_shape = attn.shape
    image_num = attn_shape[0]
    ratios = []
    
    # for each image (multi_images) or for base image (single_image)
    for i in range(image_num):
        cur_attn = attn[i].to(dtype=torch.float32).cpu().numpy().flatten()
        
        Q1 = np.percentile(cur_attn, 25)
        Q3 = np.percentile(cur_attn, 75)
        IQR = Q3 - Q1
        
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_indices = np.where((cur_attn > upper_bound))[0]
        ratio = len(outlier_indices) / len(cur_attn)
        ratios.append(ratio)
    
    return sum(ratios) / len(ratios)

def token_prune_merge_advanced_prumerge_plus(self, images, multi_images, if_adaptive=True):

    #set hooks for extracting desired layer's k and q
    hook_handle_k = self.vision_tower.vision_model.encoder.layers[25].self_attn.k_proj.register_forward_hook(hook_k_prumerge_plus)
    hook_handle_q = self.vision_tower.vision_model.encoder.layers[25].self_attn.q_proj.register_forward_hook(hook_q_prumerge_plus)

    #forward pass
    image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
    image_features = image_forward_outs.hidden_states[-1].to(images.dtype)
    assert image_features.shape[-2] == 729
    B, N, C = image_features.shape

    #extract desired layer's k and q and remove hooks; calculate attention
    desired_layer_k = outputs_prumerge_plus["desired_k"]
    desired_layer_q = outputs_prumerge_plus["desired_q"]

    hook_handle_k.remove()
    hook_handle_q.remove()

    attn = (desired_layer_q.float() @ desired_layer_k.transpose(-2, -1).float()) * C ** -0.5
    attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32)

    # Regardless of whether it is a single image use anyres or multiple images, take the average
    cls_attn = torch.mean(attn, dim=1)
    
    assert cls_attn.ndim == 2

    if if_adaptive:
        reduction_ratio = outlier_dectection_prumerge_plus(cls_attn)#*3.5
    
    # Maintaining the preset budget
    budgets_token = max(int(self.budgets * N), 1)
    iqr_token = max(int(N*reduction_ratio), 1)
    image_num = cls_attn.shape[0]

    if budgets_token > iqr_token:
        _, iqr_idx = torch.topk(cls_attn, iqr_token, dim=1, largest=True)  # [B, left_tokens]
        idx = torch.zeros((image_num, budgets_token), dtype=iqr_idx.dtype, device=self.device)
        
        for i in range(image_num):

            remaining_tokens = budgets_token - iqr_token
            
            # Sampling by arithmetic progression
            step_length = max(1, int(N / budgets_token))
            arithmetic_sequence = torch.arange(0, N, step_length).to(device=self.device)
            
            original_tensor_1d = iqr_idx[i].flatten()
            filtered_sequence = torch.tensor([x for x in arithmetic_sequence if x not in original_tensor_1d]).to(device=self.device)
            
            # If the filtered sequence is too long, truncate it
            if len(filtered_sequence) > remaining_tokens:
                filtered_sequence = filtered_sequence[:remaining_tokens]
            # If the filtered sequence is too short, randomly select additional indices
            elif len(filtered_sequence) < remaining_tokens:
                # code will not reach here
                available_indices = torch.tensor([x for x in range(N) if x not in original_tensor_1d and x not in filtered_sequence], 
                                            device=self.device)
                if len(available_indices) > 0:
                    extra_indices = available_indices[torch.randperm(len(available_indices))[:remaining_tokens - len(filtered_sequence)]]
                    filtered_sequence = torch.cat([filtered_sequence, extra_indices])
            
            # make sure the length of idx is budgets_token
            concatenated_tensor = torch.cat([iqr_idx[i], filtered_sequence])[:budgets_token]
            idx[i] = concatenated_tensor    
    else:
        _, idx = torch.topk(cls_attn, budgets_token, dim=1, largest=True)  
    
    index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]
    x_others = torch.gather(image_features, dim=1, index=index)  # [B, left_tokens, C]
    Key_others = torch.gather(desired_layer_k, dim=1, index=index)  # [B, left_tokens, C]
    x_others_attn = torch.gather(cls_attn, dim=1, index=idx)  
    compl = complement_idx_prumerge_plus(idx, N)  # [B, N-1-left_tokens]
    non_topk = torch.gather(image_features, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]
    non_topk_Key = torch.gather(desired_layer_k, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
    non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]

    Key_others_norm = nn.functional.normalize(Key_others, p=2, dim=-1)
    non_topk_Key_norm = nn.functional.normalize(non_topk_Key, p=2, dim=-1)
    B, left_tokens, C = x_others.size()
    updated_x_others = torch.zeros_like(x_others)

    for b in range(B):
        for i in range(left_tokens):
            key_others_norm = Key_others_norm[b,i,:].unsqueeze(0).unsqueeze(0)

            before_i_Key = Key_others_norm[b, :i, :].unsqueeze(0)  
            after_i_Key = Key_others_norm[b, i+1:, :].unsqueeze(0) 

            before_i_x_others = x_others[b, :i, :].unsqueeze(0)  
            after_i_x_others = x_others[b, i+1:, :].unsqueeze(0)   
            rest_x_others = torch.cat([before_i_x_others, after_i_x_others, non_topk[b,:,:].unsqueeze(0)], dim=1)   
            before_i_x_others_attn = x_others_attn[b, :i].unsqueeze(0)  
            after_i_x_others_attn = x_others_attn[b, i+1:].unsqueeze(0)  
            rest_x_others_attn = torch.cat([before_i_x_others_attn, after_i_x_others_attn, non_topk_attn[b,:].unsqueeze(0)], dim=1)  

            rest_Keys = torch.cat([before_i_Key, after_i_Key, non_topk_Key_norm[b,:,:].unsqueeze(0)], dim=1)
            cos_sim_matrix = torch.bmm(key_others_norm, rest_Keys.transpose(1, 2))

            cos_sim_num = max(min(int(32), cos_sim_matrix.shape[2]), 1)
            _, cluster_indices = torch.topk(cos_sim_matrix, k=cos_sim_num, dim=2, largest=True)

            cluster_tokens = rest_x_others[:,cluster_indices.squeeze(),:]
            weights = rest_x_others_attn[:,cluster_indices.squeeze()].unsqueeze(-1)

            # update cluster centers
            weighted_avg = torch.sum(cluster_tokens * weights, dim=1) #/ torch.sum(weights)
            updated_center = x_others[b, i, :]  + weighted_avg 
            updated_x_others[b, i, :] = updated_center 
        
    extra_one_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
    updated_x_others = torch.cat([updated_x_others, extra_one_token],dim=1)
    image_features = updated_x_others

    image_features = image_features.to(dtype=self.dtype)
    return image_features, idx

def prepare_inputs_labels_for_multimodal_prumerge_plus(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):
    vision_tower = self.get_vision_tower()
    # rank_print(modalities)
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels

    if isinstance(modalities, str):
        modalities = [modalities]

    # import pdb; pdb.set_trace()
    if type(images) is list or images.ndim == 5:
        if type(images) is list:
            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

        video_idx_in_batch = []
        for _ in range(len(modalities)):
            if modalities[_] == "video":
                video_idx_in_batch.append(_)

        images_list = []
        for image in images:
            if image.ndim == 4:
                images_list.append(image)
            else:
                images_list.append(image.unsqueeze(0))

        concat_images = torch.cat([image for image in images_list], dim=0)
        split_sizes = [image.shape[0] for image in images_list]   # num_patches for every img
        encoded_image_features, keep_idx = self.encode_images_prumerge_plus(concat_images) 
        

        encoded_image_features = torch.split(encoded_image_features, split_sizes)
        keep_idx = torch.split(keep_idx, split_sizes)
        image_features = []
        contextual_features = []
        for idx, image_feat in enumerate(encoded_image_features):  
            if idx in video_idx_in_batch:
                image_features.append(self.get_2dPool(image_feat))
            else:
                current_keep_idx = keep_idx[idx]
                num_patches = image_feat.shape[0]
                total_token_num = image_feat.shape[1]
                hidden_size = image_feat.shape[2]
                dominant_num = current_keep_idx.shape[1]
                contextual_num = total_token_num - dominant_num
                # dominant_feature is the selected tokens
                dominant_feature = image_feat[:, :dominant_num, :]
                
                # contextual_feature is the last additional token added
                if contextual_num > 0:
                    contextual_feature = image_feat[:, -contextual_num:, :]
                    contextual_features.append(contextual_feature.flatten(0,1))
                else:
                    empty_contextual_feature = torch.empty((0, hidden_size), device=image_feat.device, dtype=image_feat.dtype)
                    contextual_features.append(empty_contextual_feature)

                cur_keep_idx_sorted_restore , _ = current_keep_idx.sort(dim=1)
                fake_image = torch.zeros((num_patches, 729, hidden_size), device = image_feat.device, dtype = image_feat.dtype)    # make a fake image with the same size
                
                mask = torch.zeros(num_patches, 729, dtype=torch.bool, device=image_feat.device)
                mask.scatter_(1, cur_keep_idx_sorted_restore, True) 

                fake_image[mask] = dominant_feature.reshape(-1, hidden_size)  

                image_features.append(fake_image)   


        mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")   # spatial_unpad
        image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")   # anyres_max_9
        mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")   # one_token

        if mm_patch_merge_type == "flat":  
            image_features = [x.flatten(0, 1) for x in image_features]

        elif mm_patch_merge_type.startswith("spatial"):
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                # FIXME: now assume the image is square, and split to 2x2 patches
                # num_patches = h * w, where h = w = sqrt(num_patches)
                # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                # we want to first unflatten it to (2, 2, h, w, hidden_size)
                # rank0_print("At least we are reaching here")
                # import pdb; pdb.set_trace()
                if image_idx in video_idx_in_batch:  # video operations
                    # rank0_print("Video")
                    if mm_newline_position == "grid":
                        # Grid-wise
                        image_feature = self.add_token_per_grid(image_feature)
                        if getattr(self.config, "add_faster_video", False):
                            faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                            # Add a token for each frame
                            concat_slow_fater_token = []
                            # import pdb; pdb.set_trace()
                            for _ in range(image_feature.shape[0]):
                                if _ % self.config.faster_token_stride == 0:
                                    concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                else:
                                    concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                            # import pdb; pdb.set_trace()
                            image_feature = torch.cat(concat_slow_fater_token)

                        new_image_features.append(image_feature)
                    elif mm_newline_position == "frame":
                        # Frame-wise
                        image_feature = self.add_token_per_frame(image_feature)

                        new_image_features.append(image_feature.flatten(0, 1))
                        
                    elif mm_newline_position == "one_token":
                        # one-token
                        image_feature = image_feature.flatten(0, 1)
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                        new_image_features.append(image_feature)      
                    elif mm_newline_position == "no_token":
                        new_image_features.append(image_feature.flatten(0, 1))
                    else:
                        raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                    # rank0_print("Single-images")
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]
                    height = width = self.get_vision_tower().num_patches_per_side  # 27
                    assert height * width == base_image_feature.shape[0]

                    if "anyres_max" in image_aspect_ratio:
                        matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                        if matched_anyres_max_num_patches:
                            max_num_patches = int(matched_anyres_max_num_patches.group(1))   # 9

                    if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                        if hasattr(self.get_vision_tower(), "image_size"):
                            vision_tower_image_size = self.get_vision_tower().image_size  # 384
                        else:
                            raise ValueError("vision_tower_image_size is not found in the vision tower.")
                        try:
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)   # 3, 3
                        except Exception as e:
                            rank0_print(f"Error: {e}")
                            num_patch_width, num_patch_height = 2, 2
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1) 
                    else:
                        image_feature = image_feature.view(2, 2, height, width, -1)

                    if "maxpool2x2" in mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = nn.functional.max_pool2d(image_feature, 2)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                        unit = image_feature.shape[2]
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        c, h, w = image_feature.shape
                        times = math.sqrt(h * w / (max_num_patches * unit**2))
                        if times > 1.1:
                            image_feature = image_feature[None]
                            image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    elif "unpad" in mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    else:
                        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                        image_feature = image_feature.flatten(0, 3)
                    if "nobase" in mm_patch_merge_type:
                        pass
                    else:
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        mask_here = (image_feature != 0.0).any(dim=1)
                        image_feature = image_feature[mask_here]
                        image_feature = torch.cat((image_feature, contextual_features[image_idx]), dim=0)
                    new_image_features.append(image_feature)
                else:  # single image operations
                    image_feature = image_feature[0]
                    if "unpad" in mm_patch_merge_type:
                        image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                    new_image_features.append(image_feature)
            image_features = new_image_features
        else:
            raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
    else:
        image_features = self.encode_images_prumerge_plus_simple(images)

    # TODO: image start / end is not implemented here to support pretraining.
    if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
        raise NotImplementedError
    # rank_print(f"Total images : {len(image_features)}")

    # Let's just add dummy tensors if they do not exist,
    # it is a headache to deal with None all the time.
    # But it is not ideal, and if you have a better idea,
    # please open an issue / submit a PR, thanks.
    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    _input_ids = input_ids
    input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0

    # rank_print("Inserting Images embedding")
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        # rank0_print(num_images)
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        cur_new_input_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                try:
                    cur_image_features = image_features[cur_image_idx]
                except IndexError:
                    cur_image_features = image_features[cur_image_idx - 1]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

        cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

        # import pdb; pdb.set_trace()
        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)

    # Truncate sequences to max length as image embeddings can make the sequence longer
    tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
    # rank_print("Finishing Inserting")

    new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
    new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
    # TODO: Hard code for control loss spike
    # if tokenizer_model_max_length is not None:
    #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
    #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

    # Combine them
    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
    # rank0_print("Prepare pos id")

    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        if getattr(self.config, "tokenizer_padding_side", "right") == "left":
            new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
            if cur_len > 0:
                new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
        else:
            new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
    # rank0_print("tokenizer padding")

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None
    if getattr(self.config, "use_pos_skipping", False) and self.training:
        position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
        split_position = random.randint(0, new_input_embeds.size(1))
        left_add = random.randint(0, self.config.pos_skipping_range)
        right_add = random.randint(left_add, self.config.pos_skipping_range)
        position_ids[:, :split_position] += left_add
        position_ids[:, split_position:] += right_add
    # import pdb; pdb.set_trace()
    # rank0_print("Finish preparing")
    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels  

def encode_images_prumerge_plus(self, images , multi_images=False):
    image_features, keep_idx = self.get_model().get_vision_tower()(images, multi_images=multi_images)
    # image_features = self.get_model().vision_resampler(image_features, images=images)
    image_features = self.get_model().mm_projector(image_features)
    return image_features, keep_idx

def encode_images_prumerge_plus_simple(self, images , multi_images=True):
    image_features, keep_idx = self.get_model().get_vision_tower()(images, multi_images=multi_images)
    # image_features = self.get_model().vision_resampler(image_features, images=images)
    image_features = self.get_model().mm_projector(image_features)
    return image_features