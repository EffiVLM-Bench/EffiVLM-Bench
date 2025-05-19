      
import sys
import transformers
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from qwen2vl.modeling_qwen2_vl import Qwen2VLFlashAttention2, Qwen2VLModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
from .qwen_model import (
    qwen_attention_forward_streamingLLM,
    qwen_attention_forward_H2O,
    qwen_model_forward_vlcache,
    qwen_attention_forward_vlcache,
    qwen_attention_forward_LOOK_M,
    qwen_attention_forward_snapkv,
    qwen_model_forward_fastv,
    qwen_attention_forward_fastv,
    qwen_attn_forward_PyramidKV,
    qwen_attention_forward_CSP,
    qwen_attention_forward_random
)

from .qwen_model import (
    qwen_flash_attention_forward_streamingLLM,
    qwen_flash_attention_forward_H2O,
    qwen_flash_attention_forward_PyramidKV,
    qwen_flash_attention_forward_random,
    qwen_flash_attention_forward_snapkv,
    qwen2vl_model_forward_vlcache,
    qwen_flash_attention_forward_vlcache,
    qwen_flash_attention_forward_CSP
)

from .internlm2_model import (
    internlm2_flash_attention_forward_streamingLLM,

)
from .internvl2_5_model import (
    internvl_generate_4B_visionzip,
    internvl_extract_feature_4B_visionzip,
    internvl_attention_forward_4B_visionzip,
    internvl_naive_attn_4B_visionzip,
    internvl_generate_with_mask,
    internvl_generate_38B_visionzip,
    internvl_extract_feature_38B_visionzip,
    internvl_attention_forward_38B_visionzip,
    internvl_naive_attn_38B_visionzip,
    internvl_generate_4B_prumerge_plus,
    internvl_extract_feature_4B_prumerge_plus,
    internvl_attention_forward_4B_prumerge_plus,
    internvl_naive_attn_4B_prumerge_plus,
    internvl_generate_38B_prumerge_plus,
    internvl_extract_feature_38B_prumerge_plus,
    internvl_attention_forward_38B_prumerge_plus,
    internvl_naive_attn_38B_prumerge_plus,
)
import types 

from .kv_cache_utils import VlCacheKVCluster
from .siglip_model import *
import qwen2vl
from .qwen2vl_model import (
    qwen_vl_model_forward_fastv,
    qwen_vl_flash_attention_forward_fastv,
    qwen2vl_vision_flash_attention2_forward_visionzip,
    qwen2vl_vision_tower_forward_visionzip,
    qwen2vl_vision_block_forward_visionzip,
    qwen2vl_generation_forward_visionzip,
    qwen2vl_vision_flash_attention2_forward_prumerge_plus,
    qwen2vl_vision_tower_forward_prumerge_plus,
    qwen2vl_vision_block_forward_prumerge_plus,
    qwen2vl_generation_forward_prumerge_plus,
    qwen_flash_attention_forward_look_m,
)


def replace_qwen(args, model, method):

    if method == "streamingllm":
        print('using streamingllm')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_streamingLLM, module)
                module.budgets = args.budgets
    elif method == "h2o":
        print('using h2o')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_H2O, module)
                module.budgets = args.budgets
                module.h2o_head_adaptive = args.h2o_head_adaptive

    elif method == "vl-cache":
        print('using vlcache')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_vlcache, module)
                module.vlcache_alpha_sparsity = getattr(args,'budgets',1.0)
                module.vlcache_different_window_per_layer = getattr(args,'vlcache_different_window_per_layer',False)
                module.vlcache_head_adaptive = getattr(args,'head_adaptive',False)
                module.vlcache_budget_layer_adaptive = getattr(args, 'layer_adaptive', True)
            if isinstance(module, Qwen2Model):
                module.forward = types.MethodType(qwen_model_forward_vlcache, module)

    elif method == 'look-m':
        print('using look-m')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_LOOK_M, module)
                module.budget = getattr(args, 'budgets', 1.0)
                module.merge = getattr(args, 'merge', False)
                module.hh_ratio = getattr(args, 'hh_ratio', None)
                module.recent_ratio = getattr(args, 'recent_ratio', None)

    elif method == 'snapkv':
        print('using snapkv')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_snapkv, module)
                module.budgets = args.budgets
                module.snapkv_head_adaptive = args.snapkv_head_adaptive
                module.pooling = args.pooling

    elif method == 'fastv':
        print('using fastv')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_fastv, module)
                module.target_layer_idx = getattr(args, 'target_layer_idx', 2)
            if isinstance(module, Qwen2Model):
                module.forward = types.MethodType(qwen_model_forward_fastv, module)
                module.target_layer_idx = getattr(args, 'target_layer_idx', 2)
                module.budgets = getattr(args, 'budgets', 1.0)
                module.origin = getattr(args, 'origin', False)

    # elif method == "csp":
    #     print('using csp')
    #     transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_CSP
    #     transformers.models.qwen2.modeling_qwen2.Qwen2Attention.hh_ratio = getattr(
    #         args, 'hh_ratio', None)
    #     transformers.models.qwen2.modeling_qwen2.Qwen2Attention.recent_ratio = getattr(
    #         args, 'recent_ratio', None)
    #     # The budget ratio allocated to cross-attention. The default value is 0.1
    #     transformers.models.qwen2.modeling_qwen2.Qwen2Attention.cross_ratio = getattr(
    #         args, 'cross_ratio', 0.1)
    #     # The kv_recent_bias setting is how many times you want the recent window to be larger than the original window. The default value is 1, which means no additional increase in the window size.
    #     transformers.models.qwen2.modeling_qwen2.Qwen2Attention.kv_recent_bias = getattr(
    #         args, 'kv_recent_bias', 1)
    #     transformers.models.qwen2.modeling_qwen2.Qwen2Attention.budgets = getattr(
    #         args, 'budgets', None)
    #     transformers.models.qwen2.modeling_qwen2.Qwen2Attention.csp_head_adaptive = args.csp_head_adaptive

    elif method == 'pyramidkv':
        print('using pyramidkv')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attn_forward_PyramidKV, module)
                module.budgets = args.budgets
                module.pyramidkv_head_adaptive = args.pyramidkv_head_adaptive
                module.pooling = args.pooling
    
    elif method == 'random':
        print('using random')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_random, module)
                module.budgets = getattr(args, 'budgets', 1.0)

    elif method == 'visionzip':
        print('using visionzip')
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower, SigLipEncoderLayer, SigLipAttention
        from llava.model.llava_arch import LlavaMetaForCausalLM
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
        # SigLipVisionTower.dominant = getattr(args, 'dominant_num')
        # SigLipVisionTower.contextual = getattr(args, 'contextual_num')
        SigLipVisionTower.budgets = getattr(args, 'budgets', 1.0)
        SigLipVisionTower.forward = siglip_vision_tower_forward
        SigLipEncoderLayer.forward = siglip_EncoderLayer_forward
        SigLipAttention.forward = siglip_attention_forward
        LlavaMetaForCausalLM.encode_images_visionzip = encode_images_visionzip
        LlavaMetaForCausalLM.encode_images_visionzip_simple = encode_images_visionzip_simple
        LlavaQwenForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal_visionzip


    elif method == 'prumerge+':
        print('using prumerge+')
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
        from llava.model.llava_arch import LlavaMetaForCausalLM
        SigLipVisionTower.budgets = getattr(args, 'budgets', 1.0)
        SigLipVisionTower.forward = siglip_vision_tower_forward_prumerge_plus
        LlavaMetaForCausalLM.encode_images_prumerge_plus = encode_images_prumerge_plus
        LlavaMetaForCausalLM.encode_images_prumerge_plus_simple = encode_images_prumerge_plus_simple
        LlavaQwenForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal_prumerge_plus

    elif method == 'sparsevlm':
        print('using sparsevlm')
        from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
        from llava.model.language_model.sparse_llava_qwen import LlavaQwenSparseForCausalLM
        from llava.model.language_model.sparse_modeling_qwen import Qwen2SparseModel
        LlavaQwenSparseForCausalLM.bias = 0
        LlavaQwenSparseForCausalLM.scale = 13.5
        Qwen2SparseModel.ratio = getattr(args, 'r', 1.0)


def replace_qwen2vl(args, model, method):

    if method == 'streamingllm':
        print('using streamingllm')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2VLFlashAttention2):
                module.forward = types.MethodType(qwen_flash_attention_forward_streamingLLM, module)
                module.budgets = args.budgets

    elif method == "h2o":
        print('using h2o')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2VLFlashAttention2):
                module.forward = types.MethodType(qwen_flash_attention_forward_H2O, module)
                module.budgets = args.budgets
                module.h2o_head_adaptive = args.h2o_head_adaptive

    elif method == 'snapkv':
        print('using snapkv')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2VLFlashAttention2):
                module.forward = types.MethodType(qwen_flash_attention_forward_snapkv, module)
                module.budgets = args.budgets
                module.snapkv_head_adaptive = args.snapkv_head_adaptive
                module.pooling = args.pooling

    elif method == 'pyramidkv':
        print('using pyramidkv')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2VLFlashAttention2):
                module.forward = types.MethodType(qwen_flash_attention_forward_PyramidKV, module)
                module.budgets = args.budgets
                module.pyramidkv_head_adaptive = args.pyramidkv_head_adaptive
                module.pooling = args.pooling

    elif method == 'random':
        print('using random')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2VLFlashAttention2):
                module.forward = types.MethodType(qwen_flash_attention_forward_random, module)
                module.budgets = getattr(args, 'budgets', 1.0)

    elif method == 'look-m':
        print('using look-m')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2VLFlashAttention2):
                module.forward = types.MethodType(qwen_flash_attention_forward_look_m, module)
                module.budget = getattr(args, 'budgets', 1.0)
                module.merge = getattr(args, 'merge', True)
                module.hh_ratio = getattr(args, 'hh_ratio', None)
                module.recent_ratio = getattr(args, 'recent_ratio', None)

    elif method == 'fastv':
        print('using fastv')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2VLFlashAttention2):
                module.forward = types.MethodType(qwen_vl_flash_attention_forward_fastv, module)
                module.target_layer_idx = getattr(args, 'target_layer_idx', 2)
            if isinstance(module, Qwen2VLModel):
                module.forward = types.MethodType(qwen_vl_model_forward_fastv, module)
                module.target_layer_idx = getattr(args, 'target_layer_idx', 2)
                module.budgets = getattr(args, 'budgets', 1.0)
                module.origin = getattr(args, 'origin', False)

    elif method == 'visionzip':
        print('using visionzip')
        qwen2vl.modeling_qwen2_vl.VisionFlashAttention2.forward = qwen2vl_vision_flash_attention2_forward_visionzip
        qwen2vl.modeling_qwen2_vl.Qwen2VisionTransformerPretrainedModel.forward = qwen2vl_vision_tower_forward_visionzip
        qwen2vl.modeling_qwen2_vl.Qwen2VLVisionBlock.forward = qwen2vl_vision_block_forward_visionzip
        qwen2vl.modeling_qwen2_vl.Qwen2VisionTransformerPretrainedModel.budgets = getattr(
            args, 'budgets', 1.0)
        qwen2vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = qwen2vl_generation_forward_visionzip

    elif method == "vl-cache":
        print('using vlcache')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2VLFlashAttention2):
                module.forward = types.MethodType(qwen_flash_attention_forward_vlcache, module)
                module.vlcache_alpha_sparsity = getattr(args,'budgets',1.0)
                module.vlcache_different_window_per_layer = getattr(args,'vlcache_different_window_per_layer',False)
                module.vlcache_head_adaptive = getattr(args,'head_adaptive',False)
                module.vlcache_budget_layer_adaptive = getattr(args, 'layer_adaptive', True)
            if isinstance(module, Qwen2VLModel):
                module.forward = types.MethodType(qwen2vl_model_forward_vlcache, module)

    # elif method == "csp":
    #     print('using csp')
    #     qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attention_forward_CSP
    #     qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.hh_ratio = getattr(args, 'hh_ratio', None)
    #     qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.recent_ratio = getattr(args, 'recent_ratio', None)
    #     qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.cross_ratio = getattr(args, 'cross_ratio', 0.1)
    #     qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.kv_recent_bias = getattr(args, 'kv_recent_bias', 1)
    #     qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.budgets = getattr(args, 'budgets', None)
    #     qwen2vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.csp_head_adaptive = args.csp_head_adaptive

    elif method == 'prumerge+':
        print('using prumerge+')
        qwen2vl.modeling_qwen2_vl.VisionFlashAttention2.forward = qwen2vl_vision_flash_attention2_forward_prumerge_plus
        qwen2vl.modeling_qwen2_vl.Qwen2VisionTransformerPretrainedModel.forward = qwen2vl_vision_tower_forward_prumerge_plus
        qwen2vl.modeling_qwen2_vl.Qwen2VLVisionBlock.forward = qwen2vl_vision_block_forward_prumerge_plus
        qwen2vl.modeling_qwen2_vl.Qwen2VisionTransformerPretrainedModel.budgets = getattr(args, 'budgets', 1.0)
        qwen2vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = qwen2vl_generation_forward_prumerge_plus

def replace_internvl2_5(args, model, method):

    module_name = model.__class__.__module__
    if '8B' in module_name and "38B" not in module_name:
        mod = sys.modules.get(
            'transformers_modules.InternVL2_5-8B.modeling_internlm2', None)
    elif '26B' in module_name:
        mod = sys.modules.get(
            'transformers_modules.InternVL2_5-26B.modeling_internlm2', None)
    elif '38B' in module_name:
        return replace_qwen_for_internvl_38B(args, model, method)
    else:
        return replace_qwen_for_internvl(args, model, method)

    raise NotImplementedError('Not implemented yet')
    if method == 'streamingllm':
        print('using streamingllm')

        InternLM2AttnClass = getattr(mod, "InternLM2FlashAttention2", None)
        InternLM2AttnClass.forward = internlm2_flash_attention_forward_streamingLLM
        InternLM2AttnClass.budgets = args.budgets



def replace_qwen_for_internvl(args, model, method):

    module_name = model.__class__.__module__
    if '4B' in module_name:
        mod = sys.modules.get(
            'transformers_modules.InternVL2_5-4B.modeling_internvl_chat', None)
        InternVLChatModel = mod.InternVLChatModel
        model.generate = types.MethodType(internvl_generate_with_mask, model)

    if method == "streamingllm":
        print('using streamingllm')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_streamingLLM
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.budgets = args.budgets
    elif method == "h2o":
        print('using h2o')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_H2O
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.budgets = args.budgets
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.h2o_head_adaptive = args.h2o_head_adaptive

    elif method == "vl-cache":
        print('using vlcache')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_vlcache
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = qwen_model_forward_vlcache
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.vlcache_alpha_sparsity = getattr(args,'budgets',1.0)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.vlcache_different_window_per_layer = getattr(args,'vlcache_different_window_per_layer',False)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.vlcache_head_adaptive = getattr(args,'head_adaptive',False)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.vlcache_budget_layer_adaptive = getattr(args, 'layer_adaptive', True)

    elif method == 'look-m':
        print('using look-m')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.hh_ratio = getattr(
            args, 'hh_ratio', None)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.recent_ratio = getattr(
            args, 'recent_ratio', None)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.budget = getattr(
            args, 'budgets', 1.0)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.merge = getattr(
            args, 'merge', True)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_LOOK_M

    elif method == 'snapkv':
        print('using snapkv')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_snapkv
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.budgets = args.budgets
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.snapkv_head_adaptive = args.snapkv_head_adaptive
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.pooling = args.pooling

    elif method == 'fastv':
        print('using fastv')
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = qwen_model_forward_fastv
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_fastv
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.target_layer_idx = getattr(
            args, 'target_layer_idx', 2)
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.target_layer_idx = getattr(
            args, 'target_layer_idx', 2)
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.budgets = getattr(
            args, 'budgets', 1.0)   # visual part
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.origin = getattr(
            args, 'origin', False)

    elif method == 'pyramidkv':
        print('using pyramidkv')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attn_forward_PyramidKV
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.budgets = args.budgets
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.pyramidkv_head_adaptive = args.pyramidkv_head_adaptive
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.pooling = args.pooling

    elif method == 'visionzip':
        print('using visionzip')
        for idx, layer in enumerate(model.vision_model.encoder.layers):  
            layer.attn.layer_idx = idx
            layer.attn.forward = types.MethodType(internvl_attention_forward_4B_visionzip, layer.attn)
            layer.attn._naive_attn = types.MethodType(internvl_naive_attn_4B_visionzip, layer.attn)
        model.generate = types.MethodType(internvl_generate_4B_visionzip, model)
        model.extract_feature = types.MethodType(internvl_extract_feature_4B_visionzip, model)
        model.budgets = getattr(args, 'budgets', 1.0)

    elif method == 'random':
        print('using random')
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.budgets = getattr(
            args, 'budgets', 1.0)
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen_attention_forward_random

    elif method == 'prumerge+':
        print('using prumerge+')
        for idx, layer in enumerate(model.vision_model.encoder.layers): 
            layer.attn.layer_idx = idx
            layer.attn.forward = types.MethodType(internvl_attention_forward_4B_prumerge_plus, layer.attn)
            layer.attn._naive_attn = types.MethodType(internvl_naive_attn_4B_prumerge_plus, layer.attn)
        model.generate = types.MethodType(internvl_generate_4B_prumerge_plus, model)
        model.extract_feature = types.MethodType(internvl_extract_feature_4B_prumerge_plus, model)
        model.budgets = getattr(args, 'budgets', None)



def replace_qwen_for_internvl_38B(args, model, method):

    module_name = model.__class__.__module__
    if '38B' in module_name:
        mod = sys.modules.get(
            'transformers_modules.InternVL2_5-38B.modeling_internvl_chat', None)
        InternVLChatModel = mod.InternVLChatModel
        model.generate = types.MethodType(internvl_generate_with_mask, model)


    if method == "streamingllm":
        print('using streamingllm')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_streamingLLM, module)
                module.budgets = args.budgets
    
    elif method == "h2o":
        print('using h2o')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_H2O, module)
                module.budgets = args.budgets
                module.h2o_head_adaptive = args.h2o_head_adaptive

    elif method == "vl-cache":
        print('using vlcache')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_vlcache, module)
                module.vlcache_alpha_sparsity = getattr(args,'budgets',1.0)
                module.vlcache_different_window_per_layer = getattr(args,'vlcache_different_window_per_layer',False)
                module.vlcache_head_adaptive = getattr(args,'head_adaptive',False)
                module.vlcache_budget_layer_adaptive = getattr(args, 'layer_adaptive', True)
            if isinstance(module, Qwen2Model):
                module.forward = types.MethodType(qwen_model_forward_vlcache, module)

    elif method == 'look-m':
        print('using look-m')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_LOOK_M, module)
                module.hh_ratio = getattr(args, 'hh_ratio', None)
                module.recent_ratio = getattr(args, 'recent_ratio', None)
                module.budget = getattr(args, 'budgets', 1.0)
                module.merge = getattr(args, 'merge', True)

    elif method == 'snapkv':
        print('using snapkv')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_snapkv, module)
                module.snapkv_head_adaptive = args.snapkv_head_adaptive
                module.pooling = args.pooling
                module.budgets = args.budgets


    elif method == 'fastv':
        print('using fastv')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attention_forward_fastv, module)
                module.target_layer_idx = getattr(args, 'target_layer_idx', 2)
            if isinstance(module, Qwen2Model):
                module.forward = types.MethodType(qwen_model_forward_fastv, module)
                module.target_layer_idx = getattr(args, 'target_layer_idx', 2)
                module.budgets = getattr(args, 'budgets', 1.0)
                module.origin = getattr(args, 'origin', False)


    elif method == 'pyramidkv':
        print('using pyramidkv')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.forward = types.MethodType(qwen_attn_forward_PyramidKV, module)
                module.budgets = args.budgets
                module.pyramidkv_head_adaptive = args.pyramidkv_head_adaptive
                module.pooling = args.pooling

    
    elif method == 'random':
        print('using random')
        for name, module in model.named_modules():
            if isinstance(module, Qwen2Attention):
                module.budgets = getattr(args, 'budgets', 1.0)
                module.forward = types.MethodType(qwen_attention_forward_random, module)
    
    elif method == 'visionzip':
        print('using visionzip')
        for idx, layer in enumerate(model.vision_model.encoder.layers): 
            layer.attn.layer_idx = idx
            layer.attn.forward = types.MethodType(internvl_attention_forward_38B_visionzip, layer.attn)
            layer.attn._naive_attn = types.MethodType(internvl_naive_attn_38B_visionzip, layer.attn)
        model.generate = types.MethodType(internvl_generate_38B_visionzip, model)
        model.extract_feature = types.MethodType(internvl_extract_feature_38B_visionzip, model)
        model.budgets = getattr(args, 'budgets', 1.0)


    elif method == 'prumerge+':
        print('using prumerge+')
        for idx, layer in enumerate(model.vision_model.encoder.layers):
            layer.attn.layer_idx = idx
            layer.attn.forward = types.MethodType(internvl_attention_forward_38B_prumerge_plus, layer.attn)
            layer.attn._naive_attn = types.MethodType(internvl_naive_attn_38B_prumerge_plus, layer.attn)
        model.generate = types.MethodType(internvl_generate_38B_prumerge_plus, model)
        model.extract_feature = types.MethodType(internvl_extract_feature_38B_prumerge_plus, model)
        model.budgets = getattr(args, 'budgets', 1.0)

def replace_mistral(method):
    pass


def replace_llama(method):
    pass

    