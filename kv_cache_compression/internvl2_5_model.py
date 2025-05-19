import warnings
from typing import List, Optional, Tuple, Union

import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
import numpy as np

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))



@torch.no_grad()
def internvl_generate_with_mask(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        **generate_kwargs,
) -> torch.LongTensor:

    assert self.img_context_token_id is not None
    if pixel_values is not None:
        if visual_features is not None:
            vit_embeds = visual_features
        else:
            vit_embeds = self.extract_feature(pixel_values)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        assert selected.sum() != 0

        text_image_mask = (input_ids != self.img_context_token_id)
        assert text_image_mask.sum() != 0
        for layer in self.language_model.model.layers:
            layer.self_attn.text_image_mask = [text_image_mask.tolist()]
        self.language_model.model.text_image_mask = [text_image_mask.tolist()]

        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

        input_embeds = input_embeds.reshape(B, N, C)
    else:
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

    outputs = self.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
        output_hidden_states=output_hidden_states,
        use_cache=True,
        **generate_kwargs,
    )

    return outputs


def internvl_naive_attn_4B_visionzip(self, x):
        B, N, C = x.shape   # batchsize， seqlen， hidden_size
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        origin_dtype = q.dtype
        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        if self.layer_idx == 22:
            attn = ((q.float() * self.scale) @ k.transpose(-2, -1).float())
            self.attn_weights = attn[:,:,1:,1:]   
            self.metric = k.mean(dim=1)[:,1:,:]
            attn = attn.to(origin_dtype)
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)             
        attn = self.attn_drop(attn).to(origin_dtype)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def internvl_attention_forward_4B_visionzip(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self._naive_attn(hidden_states)  
        return x

def internvl_extract_feature_4B_visionzip(self, pixel_values):  
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)   # [num_patches, 256, 2048]

        attn_weights = self.vision_model.encoder.layers[-2].attn.attn_weights
        metric = self.vision_model.encoder.layers[-2].attn.metric
        metric = metric.view(metric.shape[0], 32, 32, -1)
        metric = metric.view(metric.shape[0], 16, 2, 16, 2, -1).mean(dim=(2, 4))
        metric = metric.flatten(1,2)
        hidden_states = vit_embeds

        per_patch_token_num = metric.shape[1]
        dominant_num =  max(1, int(self.budgets * per_patch_token_num * 5.4 / 6.4))
        contextual_num = max(1, int(self.budgets * per_patch_token_num * 1 / 6.4))

        attention_sum = attn_weights.mean(dim=1).mean(dim=1)
        attention_sum = attention_sum.view(attention_sum.shape[0], 32, 32)
        attention_sum = attention_sum.view(attention_sum.shape[0], 16, 2, 16, 2).mean(dim=(2, 4))
        attention_sum = attention_sum.flatten(1)

        all_indices = attention_sum.topk(dominant_num, dim=1).indices
        mask = torch.ones_like(hidden_states[:, :, 0], dtype=torch.bool, device=metric.device).scatter_(1, all_indices, False)  
        dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(hidden_states.shape[0], dominant_num, hidden_states.shape[2])
        
        ### Filter
        metric_filtered = metric[mask].view(hidden_states.shape[0], hidden_states.shape[1] - dominant_num, metric.shape[2])  
        hidden_states_filtered = hidden_states.masked_select(mask.unsqueeze(-1)).view(hidden_states.shape[0], hidden_states.shape[1] - dominant_num, hidden_states.shape[2])  
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
        hidden_states_save = torch.cat([dominant_tokens, contextual_tokens], dim=1).to(pixel_values.dtype) 

        return hidden_states_save


@torch.no_grad()
def internvl_generate_4B_visionzip(self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,   # 151665 151667 151666
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        **generate_kwargs,
) -> torch.LongTensor:

    assert self.img_context_token_id is not None
    if pixel_values is not None:
        if visual_features is not None:
            vit_embeds = visual_features
        else:
            vit_embeds = self.extract_feature(pixel_values)
        assert input_ids.shape[0] == 1, 'only support batch_size == 1'
        num_visual_tokens = vit_embeds.shape[1] * vit_embeds.shape[0]
        input_ids_flat = input_ids[0]  # shape: [seq_len]
        visual_positions = (input_ids_flat == self.img_context_token_id).nonzero(as_tuple=False).flatten()
        if visual_positions.numel() > num_visual_tokens:
            mask = torch.ones_like(input_ids_flat, dtype=torch.bool)
            mask[visual_positions[num_visual_tokens:]] = False
            input_ids = input_ids_flat[mask].unsqueeze(0)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        assert selected.sum() != 0
        text_image_mask = (input_ids != self.img_context_token_id)
        assert text_image_mask.sum() != 0
        for layer in self.language_model.model.layers:
            layer.self_attn.text_image_mask = [text_image_mask.tolist()]
        self.language_model.model.text_image_mask = [text_image_mask.tolist()]
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
        input_embeds = input_embeds.reshape(B, N, C)
    else:
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

    outputs = self.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
        output_hidden_states=output_hidden_states,
        use_cache=True,
        **generate_kwargs,
    )

    return outputs


def internvl_naive_attn_38B_visionzip(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    origin_dtype = q.dtype
    if self.qk_normalization:
        B_, H_, N_, D_ = q.shape
        q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
        k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

    if self.layer_idx == 43:
        attn = ((q.float() * self.scale) @ k.transpose(-2, -1).float())
        self.attn_weights = attn[:,:,1:,1:]   
        self.metric = k.mean(dim=1)[:,1:,:]
        attn = attn.to(origin_dtype)
    else:
        attn = (q * self.scale) @ k.transpose(-2, -1)

    attn = ((q * self.scale) @ k.transpose(-2, -1))
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def internvl_attention_forward_38B_visionzip(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self._naive_attn(hidden_states)
        return x


def internvl_extract_feature_38B_visionzip(self, pixel_values):
    if self.select_layer == -1:
        vit_embeds = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True).last_hidden_state
    else:
        vit_embeds = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True).hidden_states[self.select_layer]
    vit_embeds = vit_embeds[:, 1:, :]

    h = w = int(vit_embeds.shape[1] ** 0.5)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
    vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
    vit_embeds = self.mlp1(vit_embeds)
    
    attn_weights = self.vision_model.encoder.layers[-2].attn.attn_weights
    metric = self.vision_model.encoder.layers[-2].attn.metric
    metric = metric.view(metric.shape[0], 32, 32, -1)
    metric = metric.view(metric.shape[0], 16, 2, 16, 2, -1).mean(dim=(2, 4))
    metric = metric.flatten(1,2)
    hidden_states = vit_embeds.to(metric.device)

    per_patch_token_num = metric.shape[1]
    dominant_num =  max(1, int(self.budgets * per_patch_token_num * 5.4 / 6.4))
    contextual_num = max(1, int(self.budgets * per_patch_token_num * 1 / 6.4))

    attention_sum = attn_weights.mean(dim=1).mean(dim=1)
    attention_sum = attention_sum.view(attention_sum.shape[0], 32, 32)
    attention_sum = attention_sum.view(attention_sum.shape[0], 16, 2, 16, 2).mean(dim=(2, 4))
    attention_sum = attention_sum.flatten(1)

    all_indices = attention_sum.topk(dominant_num, dim=1).indices
    mask = torch.ones_like(hidden_states[:, :, 0], dtype=torch.bool, device=metric.device).scatter_(1, all_indices, False) 
    dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(hidden_states.shape[0], dominant_num, hidden_states.shape[2])
    
    ### Filter
    metric_filtered = metric[mask].view(hidden_states.shape[0], hidden_states.shape[1] - dominant_num, metric.shape[2])
    hidden_states_filtered = hidden_states.masked_select(mask.unsqueeze(-1)).view(hidden_states.shape[0], hidden_states.shape[1] - dominant_num, hidden_states.shape[2]) 
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
    hidden_states_save = torch.cat([dominant_tokens, contextual_tokens], dim=1).to(pixel_values.dtype)

    return hidden_states_save


@torch.no_grad()
def internvl_generate_38B_visionzip(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

    assert self.img_context_token_id is not None
    if pixel_values is not None:
        if visual_features is not None:
            vit_embeds = visual_features
        else:
            vit_embeds = self.extract_feature(pixel_values)
        assert input_ids.shape[0] == 1, 'only support batch_size == 1'
        num_visual_tokens = vit_embeds.shape[1] * vit_embeds.shape[0]
        input_ids_flat = input_ids[0]  # shape: [seq_len]
        visual_positions = (input_ids_flat == self.img_context_token_id).nonzero(as_tuple=False).flatten()
        if visual_positions.numel() > num_visual_tokens:
            mask = torch.ones_like(input_ids_flat, dtype=torch.bool)
            mask[visual_positions[num_visual_tokens:]] = False
            input_ids = input_ids_flat[mask].unsqueeze(0)

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        assert selected.sum() != 0

        text_image_mask = (input_ids != self.img_context_token_id)
        assert text_image_mask.sum() != 0
        for layer in self.language_model.model.layers:
            layer.self_attn.text_image_mask = [text_image_mask.tolist()]
        self.language_model.model.text_image_mask = [text_image_mask.tolist()]

        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

        input_embeds = input_embeds.reshape(B, N, C)
    else:
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

    outputs = self.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
        output_hidden_states=output_hidden_states,
        use_cache=True,
        **generate_kwargs,
    )

    return outputs

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

def internvl_naive_attn_4B_prumerge_plus(self, x):
    B, N, C = x.shape   # batchsize， seqlen， hidden_size
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    origin_dtype = q.dtype
    if self.qk_normalization:
        B_, H_, N_, D_ = q.shape
        q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
        k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

    # penultimate layer token 
    if self.layer_idx == 22:
        key_prumerger = k.permute(0, 2, 1, 3).reshape(B, N, C)
        query_prumerger = q.permute(0, 2, 1, 3).reshape(B, N, C)
        attn_prumerger = (query_prumerger.float() @ key_prumerger.transpose(-2, -1).float()) * C ** -0.5
        attn_prumerger = attn_prumerger.softmax(dim=-1)
        self.cls_attn_weights = attn_prumerger[:,0,1:]
        self.metric = key_prumerger[:,1:,:]
    
    attn = (q * self.scale) @ k.transpose(-2, -1)

    attn = attn.softmax(dim=-1)             
    attn = self.attn_drop(attn).to(origin_dtype)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def internvl_attention_forward_4B_prumerge_plus(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self._naive_attn(hidden_states) 
        return x

def internvl_extract_feature_4B_prumerge_plus(self, pixel_values):
    if self.select_layer == -1:
        vit_embeds = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True).last_hidden_state
    else:
        vit_embeds = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True).hidden_states[self.select_layer]
    vit_embeds = vit_embeds[:, 1:, :]

    h = w = int(vit_embeds.shape[1] ** 0.5)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
    vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
    vit_embeds = self.mlp1(vit_embeds)   # [num_patches, 256, 2048]

    cls_attn = self.vision_model.encoder.layers[-2].attn.cls_attn_weights # [1, 1024]
    cls_attn = cls_attn.view(cls_attn.shape[0], 32, 32)
    cls_attn = cls_attn.view(cls_attn.shape[0], 16, 2, 16, 2).mean(dim=(2, 4))
    cls_attn = cls_attn.flatten(1,2)

    metric = self.vision_model.encoder.layers[-2].attn.metric
    metric = metric.view(metric.shape[0], 32, 32, -1)
    metric = metric.view(metric.shape[0], 16, 2, 16, 2, -1).mean(dim=(2, 4))
    desired_layer_k = metric.flatten(1,2)

    B, N, C = vit_embeds.shape # anyres, 256, 2048
    B_key, N_key, C_key = desired_layer_k.shape
    assert B == B_key
    assert N == N_key
    
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
        _, idx = torch.topk(cls_attn, budgets_token, dim=1, largest=True)  # [B, left_tokens] , sorted=True
    
    index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

    x_others = torch.gather(vit_embeds, dim=1, index=index)  # [B, left_tokens, C]
    Key_others = torch.gather(desired_layer_k, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, C_key))  # [B, left_tokens, C_key]
    x_others_attn = torch.gather(cls_attn, dim=1, index=idx)  
    compl = complement_idx_prumerge_plus(idx, N)  # [B, N-1-left_tokens]
    non_topk = torch.gather(vit_embeds, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]
    non_topk_Key = torch.gather(desired_layer_k, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C_key))
    non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]

    Key_others_norm = nn.functional.normalize(Key_others, p=2, dim=-1)
    non_topk_Key_norm = nn.functional.normalize(non_topk_Key, p=2, dim=-1)

    # cos_sim = torch.bmm(Key_others_norm, non_topk_Key_norm.transpose(1, 2)) # [B, left_tokens, N-1-left_tokens]

    # _, cluster_indices = torch.topk(cos_sim, k=4, dim=2, largest=True)

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
            # updated_center = x_others[b, i, :]
            updated_x_others[b, i, :] = updated_center 
        
    extra_one_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
    updated_x_others = torch.cat([updated_x_others, extra_one_token],dim=1)
    image_features = updated_x_others

    image_features = image_features.to(dtype=self.dtype)
    return image_features


@torch.no_grad()
def internvl_generate_4B_prumerge_plus(self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,   # 151665 151667 151666
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        **generate_kwargs,
) -> torch.LongTensor:

    assert self.img_context_token_id is not None
    if pixel_values is not None:
        if visual_features is not None:
            vit_embeds = visual_features
        else:
            vit_embeds = self.extract_feature(pixel_values)

        assert input_ids.shape[0] == 1, 'Currently only one batch is supported'
        
        num_visual_tokens = vit_embeds.shape[1] * vit_embeds.shape[0]
        input_ids_flat = input_ids[0]  # shape: [seq_len]
        visual_positions = (input_ids_flat == self.img_context_token_id).nonzero(as_tuple=False).flatten()
        if visual_positions.numel() > num_visual_tokens:
            mask = torch.ones_like(input_ids_flat, dtype=torch.bool)
            mask[visual_positions[num_visual_tokens:]] = False
            input_ids = input_ids_flat[mask].unsqueeze(0)

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        assert selected.sum() != 0

        text_image_mask = (input_ids != self.img_context_token_id)
        assert text_image_mask.sum() != 0
        for layer in self.language_model.model.layers:
            layer.self_attn.text_image_mask = [text_image_mask.tolist()]
        self.language_model.model.text_image_mask = [text_image_mask.tolist()]

        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

        input_embeds = input_embeds.reshape(B, N, C)
    else:
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

    outputs = self.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
        output_hidden_states=output_hidden_states,
        use_cache=True,
        **generate_kwargs,
    )

    return outputs


def internvl_naive_attn_38B_prumerge_plus(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    origin_dtype = q.dtype
    if self.qk_normalization:
        B_, H_, N_, D_ = q.shape
        q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
        k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

    if self.layer_idx == 43:
        key_prumerger = k.permute(0, 2, 1, 3).reshape(B, N, C)
        query_prumerger = q.permute(0, 2, 1, 3).reshape(B, N, C)
        attn_prumerger = (query_prumerger.float() @ key_prumerger.transpose(-2, -1).float()) * C ** -0.5
        attn_prumerger = attn_prumerger.softmax(dim=-1)
        self.cls_attn_weights = attn_prumerger[:,0,1:]
        self.metric = key_prumerger[:,1:,:]
    
    attn = (q * self.scale) @ k.transpose(-2, -1)

    attn = attn.softmax(dim=-1)             
    attn = self.attn_drop(attn).to(origin_dtype)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def internvl_attention_forward_38B_prumerge_plus(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self._naive_attn(hidden_states)
        return x



def internvl_extract_feature_38B_prumerge_plus(self, pixel_values): 

    if self.select_layer == -1:
        vit_embeds = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True).last_hidden_state
    else:
        vit_embeds = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True).hidden_states[self.select_layer]
    vit_embeds = vit_embeds[:, 1:, :]

    h = w = int(vit_embeds.shape[1] ** 0.5)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
    vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
    vit_embeds = self.mlp1(vit_embeds).to(self.device)   # [num_patches, 256, 2048]

    cls_attn = self.vision_model.encoder.layers[-2].attn.cls_attn_weights
    cls_attn = cls_attn.view(cls_attn.shape[0], 32, 32)
    cls_attn = cls_attn.view(cls_attn.shape[0], 16, 2, 16, 2).mean(dim=(2, 4))
    cls_attn = cls_attn.flatten(1,2).to(self.device)

    metric = self.vision_model.encoder.layers[-2].attn.metric
    metric = metric.view(metric.shape[0], 32, 32, -1)
    metric = metric.view(metric.shape[0], 16, 2, 16, 2, -1).mean(dim=(2, 4))
    desired_layer_k = metric.flatten(1,2).to(self.device)

    B, N, C = vit_embeds.shape # anyres, 256, 2048
    B_key, N_key, C_key = desired_layer_k.shape
    assert B == B_key
    assert N == N_key
    
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
            
            original_tensor_1d = iqr_idx[i].flatten().to(device=self.device)
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
        _, idx = torch.topk(cls_attn, budgets_token, dim=1, largest=True)  # [B, left_tokens] , sorted=True
        idx = idx.to(device=self.device)
    
    index = idx.unsqueeze(-1).expand(-1, -1, C).to(self.device)  # [B, left_tokens, C]

    x_others = torch.gather(vit_embeds, dim=1, index=index)  # [B, left_tokens, C]
    Key_others = torch.gather(desired_layer_k, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, C_key))  # [B, left_tokens, C_key]
    x_others_attn = torch.gather(cls_attn, dim=1, index=idx)  
    compl = complement_idx_prumerge_plus(idx, N)  # [B, N-1-left_tokens]
    non_topk = torch.gather(vit_embeds, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]
    non_topk_Key = torch.gather(desired_layer_k, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C_key))
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
    return image_features
    
    

@torch.no_grad()
def internvl_generate_38B_prumerge_plus(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

    assert self.img_context_token_id is not None
    if pixel_values is not None:
        if visual_features is not None:
            vit_embeds = visual_features
        else:
            vit_embeds = self.extract_feature(pixel_values)

        assert input_ids.shape[0] == 1, 'Currently only one batch is supported'

        num_visual_tokens = vit_embeds.shape[1] * vit_embeds.shape[0]
        input_ids_flat = input_ids[0]  # shape: [seq_len]
        visual_positions = (input_ids_flat == self.img_context_token_id).nonzero(as_tuple=False).flatten()
        if visual_positions.numel() > num_visual_tokens:
            mask = torch.ones_like(input_ids_flat, dtype=torch.bool)
            mask[visual_positions[num_visual_tokens:]] = False
            input_ids = input_ids_flat[mask].unsqueeze(0)

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        assert selected.sum() != 0

        text_image_mask = (input_ids != self.img_context_token_id)
        assert text_image_mask.sum() != 0
        for layer in self.language_model.model.layers:
            layer.self_attn.text_image_mask = [text_image_mask.tolist()]
        self.language_model.model.text_image_mask = [text_image_mask.tolist()]

        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

        input_embeds = input_embeds.reshape(B, N, C)
    else:
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

    outputs = self.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
        output_hidden_states=output_hidden_states,
        use_cache=True,
        **generate_kwargs,
    )

    return outputs