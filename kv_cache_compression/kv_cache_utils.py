import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math

from typing import List


from typing import List, Optional, Tuple
from transformers.cache_utils import Cache
from flash_attention_softmax_n import softmax_n


def key_pruner_query_driven(kv_states, q_states, recent_size=128, ratio=0.3):
    _, _, seqlen, head_dim = kv_states.shape
    k = max(1, int(head_dim * ratio))
    k = max(1, int(head_dim * ratio))
    # new efficient implementation
    queries_norm = torch.pow(q_states[..., -32:, :], 2).mean(dim=2)
    keys_norm = torch.pow(kv_states, 2).mean(dim=2)
    key = queries_norm * keys_norm
    _, indices = torch.topk(key, k, dim=-1, largest=False)
    keep_idx = indices.sort().values
    mask = torch.zeros(key.shape, dtype=torch.bool).to(kv_states.device)
    mask = mask.scatter_(-1, keep_idx, 1)
    mask_k = mask.unsqueeze(2).expand(-1, -1, seqlen - recent_size, -1)

    return kv_states[:, :, :seqlen - recent_size, :][~mask_k].reshape(1, -1, seqlen - recent_size, head_dim-k), kv_states[:, :, seqlen - recent_size:, :], ~mask


class DynamicCacheSplitHeadFlatten(Cache):
    '''
    adapt from https://github.com/FFY0/AdaKV.
    '''

    def __init__(self) -> None:
        # Token wise List[]  Head wise KV List[torch.Tensor]
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0

    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (tuple(self.key_cache[layer_idx]), tuple(self.value_cache[layer_idx]))

    def __getitem__(self, layer_idx: int) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (tuple(self.key_cache[layer_idx]), tuple(self.value_cache[layer_idx]))
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            assert self.key_cache[layer_idx].dim() == 2
            bs, head, seqlen, dim = key_states.shape
            assert bs == 1 and seqlen == 1
            head_lens = cache_kwargs["head_lens"]
            cu_klen = cache_kwargs["cu_klen"]

            import nvtx
            copy_old_rng = nvtx.start_range("copy old")
            from tiny_api_cuda import update_flatten_view
            new_key_cache = update_flatten_view(
                self.key_cache[layer_idx].view(-1, dim), key_states.view(-1, dim), head_lens, cu_klen)
            new_value_cache = update_flatten_view(
                self.value_cache[layer_idx].view(-1, dim), value_states.view(-1, dim), head_lens, cu_klen)

            nvtx.end_range(copy_old_rng)

            self.key_cache[layer_idx] = new_key_cache
            self.value_cache[layer_idx] = new_value_cache

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        # TODO: return 1 to means has content for now
        return 1

    def get_max_length(self) -> Optional[int]:
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx],
                             self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCacheEachHead":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

# perform qk calculation and get indices
# this version will not update in inference mode

# Copied from transformers.models.llama.modeling_llama.repeat_kv


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def merge_kv(key_states, value_states, indices, window_size, merge):
    # merge methods in LOOK-M

    bsz, num_heads, k_len, head_dim = key_states.shape

    # kv-selected
    # [bsz, num_heads, topk_len, head_dim]
    selected_keys = key_states.gather(dim=2, index=indices)
    # [bsz, num_heads, topk_len, head_dim]
    selected_values = value_states.gather(dim=2, index=indices)

    # kv-drop
    all_indices = torch.arange(k_len, device=key_states.device).unsqueeze(
        0).unsqueeze(0).expand(bsz, num_heads, k_len)
    # [bsz * num_heads * (k_len-window_size)]
    all_indices_flattened = all_indices.flatten()
    # [bsz * num_heads * topk_len]
    selected_indices_flattened = indices.flatten()
    is_selected = torch.isin(all_indices_flattened, selected_indices_flattened)
    drop_indices_flattened = all_indices_flattened[~is_selected]
    drop_len = drop_indices_flattened.shape[0] // (
        all_indices.shape[0] * all_indices.shape[1])
    # [bsz * num_heads * (k_len-window_size-topk_len)]
    drop_indices = drop_indices_flattened.reshape(
        all_indices.shape[0], all_indices.shape[1], drop_len)
    # [bsz, num_heads, (k_len-window_size-topk_len), head_dim]
    drop_indices = drop_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    drop_keys = key_states.gather(dim=2, index=drop_indices)
    drop_values = value_states.gather(dim=2, index=drop_indices)

    # kv-recent
    recent_keys = key_states[:, :, -window_size:, :]

    ##### apply merge #####
    # prepare for merge
    # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    k_hh_pruned = drop_keys
    # [bsz, num_heads, topk_len+window_size, head_dim]
    k_hh_recent = torch.cat([recent_keys, selected_keys], dim=2)
    # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    v_hh_pruned = drop_values
    # [bsz, num_heads, topk_len+window_size, head_dim]
    v_hh_recent = torch.cat(
        [selected_values, value_states[:, :, -window_size:, :]], dim=2)
    # similarity matrix
    similarity = (k_hh_pruned / torch.norm(k_hh_pruned, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)) @ (
        (k_hh_recent / (torch.norm(k_hh_recent, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128))).transpose(-1, -2))  # cosin
    max_values, max_indices = similarity.max(dim=-1)

    # pivot merge
    if merge == "pivot":
        print("Pivot merge")
        merged_indices = max_indices.unsqueeze(-1).repeat(1, 1, 1, 128)
        k_hh_selected = torch.gather(
            input=k_hh_recent, dim=2, index=merged_indices)
        k_hh_merged = (k_hh_pruned + k_hh_selected)/2
        k_hh_recent = torch.scatter_reduce(input=k_hh_recent, dim=2, index=merged_indices, src=k_hh_merged,
                                           reduce='mean', include_self=True)  # include_self=True seems decrease the performance
        v_hh_selected = torch.gather(
            input=v_hh_recent, dim=2, index=merged_indices)
        v_hh_merged = (v_hh_pruned + v_hh_selected)/2
        v_hh_recent = torch.scatter_reduce(
            input=v_hh_recent, dim=2, index=merged_indices, src=v_hh_merged, reduce='mean', include_self=True)
    else:
        raise ValueError('Merge method not supported')

    # TODO: other merge strategies
    # average merge
    # weight merge

    return k_hh_recent, v_hh_recent


class StreamingLLMKVCluster():
    def __init__(self, query_len, budgets, window_size_budgets=0.1, merge=None):
        self.query_len = query_len
        self.budgets = budgets 
        self.max_capacity_prompt = max(1, int(query_len * budgets))
        self.window_size_budgets = window_size_budgets 

        self.window_size = max(1, int(self.max_capacity_prompt * window_size_budgets))
        self.merge = merge

    def reset(self, budgets, window_size_budgets=0.1, merge=None):
        self.query_len = None

        self.budgets = budgets
        self.window_size_budgets = window_size_budgets

        self.window_size = None
        self.max_capacity_prompt = None
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):

        bsz, num_kv_heads, q_len, head_dim = key_states.shape

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            indices = torch.tensor(range(
                self.max_capacity_prompt - self.window_size), dtype=torch.int64).to(key_states.device)
            indices = indices.unsqueeze(0).unsqueeze(
                0).unsqueeze(-1).repeat(bsz, num_kv_heads, 1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(
                    key_states, value_states, indices, self.window_size, self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-
                                         self.window_size, :].gather(dim=2, index=indices)
            v_past_compress = value_states[:, :, :-
                                           self.window_size, :].gather(dim=2, index=indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            return key_states, value_states


class H2OKVCluster():
    def __init__(self, query_len, budgets, window_size_budgets=0.1, head_adaptive=True, merge=None):
        self.query_len = query_len
        
        self.budgets = budgets 
        self.max_capacity_prompt = max(1, int(query_len * budgets))
        self.window_size_budgets = window_size_budgets 

        self.window_size = max(1, int(self.max_capacity_prompt * window_size_budgets))
        self.head_adaptive = head_adaptive
        self.merge = merge

    def reset(self, budgets, window_size_budgets=0.1, head_adaptive=True, merge=None):
        self.query_len = None

        self.budgets = budgets 
        self.window_size_budgets = window_size_budgets

        self.window_size = None
        self.max_capacity_prompt = None
        self.merge = merge
        self.head_adaptive = head_adaptive

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):

        bsz, num_heads, q_len, head_dim = query_states.shape
        assert key_states.shape[1] == num_heads // num_key_value_groups
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        key_states_repeat = repeat_kv(key_states, num_key_value_groups)
        attn_weights = torch.matmul( 
            query_states.float(), key_states_repeat.transpose(-2, -1).float()) / math.sqrt(head_dim)   # float32

        mask = torch.triu(
            torch.ones((q_len, q_len), dtype=torch.bool, device=query_states.device),
            diagonal=1
        )
        attn_weights = attn_weights.masked_fill(mask, float('-inf'))
        del mask
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.view(
            bsz, num_heads // num_key_value_groups, num_key_value_groups, q_len, -1)

        if self.head_adaptive:
            attn_weights_sum = attn_weights[:, :, :,
                                            :, :-self.window_size].sum(dim=[-3, -2])
        else:
            attn_weights_sum = attn_weights[:, :, :, :, :-self.window_size].sum(
                dim=[1, 2, 3]).unsqueeze(1).expand(-1, num_heads // num_key_value_groups, -1)

        indices = attn_weights_sum.topk(
            self.max_capacity_prompt - self.window_size, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        del attn_weights
        
        k_past_compress = key_states[:, :, :-
                                     self.window_size, :].gather(dim=2, index=indices)
        v_past_compress = value_states[:, :, :-
                                       self.window_size, :].gather(dim=2, index=indices)

        k_cur = key_states[:, :, -self.window_size:, :]
        v_cur = value_states[:, :, -self.window_size:, :]
        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)

        return key_states, value_states


class VlCacheKVCluster():

    def __init__(self, vlcache_alpha_sparsity, vlcache_different_window_per_layer, vlcache_head_adaptive, vlcache_budget_layer_adaptive):

        self.vlcache_alpha_sparsity = vlcache_alpha_sparsity
        self.vlcache_different_window_per_layer = vlcache_different_window_per_layer
        self.vlcache_head_adaptive = vlcache_head_adaptive
        self.sparsity_layer_tuple = ()
        self.attn_weights_importance_tuple = ()
        self.isprefill = True
        self.vlcache_budget_layer_adaptive = vlcache_budget_layer_adaptive

    def allocate_budget_and_update_kv(
            self,
            buget_layers: torch.Tensor,
            sorted_attn_kv_indices: torch.Tensor,
            layer_idx: int,
            past_key_value: Cache
    ):
        """
        Called in the forward function of Qwen2Attention, after the prefill phase ends,
        Crop the kv cache based on the reallocated budget and token importance.
        """
        kv_cache_image_num = buget_layers[layer_idx] * \
            past_key_value[layer_idx][0].shape[-2]
        if self.vlcache_different_window_per_layer:
            # different window size for each layer
            kv_cache_window_num = buget_layers[layer_idx] * \
                past_key_value[layer_idx][0].shape[-2] * 0.1
        else:
            # same window size for each layer , 10% budget for each layer
            kv_cache_window_num = past_key_value[layer_idx][0].shape[-2] * \
                self.vlcache_alpha_sparsity * 0.1
        kv_cache_image_num_int = max(1, int(kv_cache_image_num.item()))
        kv_cache_window_num_int = max(1, int(kv_cache_window_num))
        # choose window
        sorted_attn_kv_select_window = torch.arange(
            past_key_value[layer_idx][0].shape[-2] - kv_cache_window_num_int,
            past_key_value[layer_idx][0].shape[-2],
            device=sorted_attn_kv_indices.device
        )

        # choose other tokens
        if self.vlcache_head_adaptive:
            assert len(sorted_attn_kv_indices.shape) == 4
            # [layer, batch_size, head_num , kv_len]
            sorted_attn_kv_indices_layer = sorted_attn_kv_indices[layer_idx, 0, :, :]
            sorted_attn_kv_select_window = sorted_attn_kv_select_window.repeat(
                sorted_attn_kv_indices.shape[2], 1)
            mask = ~ torch.isin(sorted_attn_kv_indices_layer,
                                sorted_attn_kv_select_window)
            sorted_attn_kv_select_image = torch.stack([
                row[m] for row, m in zip(sorted_attn_kv_indices_layer, mask)
            ])
            sorted_attn_kv_select_image = sorted_attn_kv_select_image[:,
                                                                      :kv_cache_image_num_int]
        else:
            assert len(sorted_attn_kv_indices.shape) == 2
            # [layer,kv_len]
            sorted_attn_kv_indices_layer = sorted_attn_kv_indices[layer_idx]
            mask = ~ torch.isin(sorted_attn_kv_indices_layer,
                                sorted_attn_kv_select_window)
            sorted_attn_kv_select_image = sorted_attn_kv_indices_layer[mask]
            sorted_attn_kv_select_image = sorted_attn_kv_select_image[:kv_cache_image_num_int]

        sorted_attn_kv_select = torch.cat(
            [sorted_attn_kv_select_image, sorted_attn_kv_select_window], dim=-1)
        sorted_attn_kv_select = torch.sort(sorted_attn_kv_select, dim=-1)[0]
        sorted_attn_kv_select = sorted_attn_kv_select.to(past_key_value.key_cache[layer_idx].device)

        if self.vlcache_head_adaptive:
            self.prefill_update_head(
                past_key_value, layer_idx, sorted_attn_kv_select)
        else:
            past_key_value.key_cache[layer_idx] = past_key_value.key_cache[layer_idx][:,
                                                                                      :, sorted_attn_kv_select, :]
            past_key_value.value_cache[layer_idx] = past_key_value.value_cache[layer_idx][:,
                                                                                          :, sorted_attn_kv_select, :]

    def get_sparsity_layer(
        self,
        attn_weights_postvison: torch.Tensor,
        q_len: int,
        post_vision_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Called in the forward function of Qwen2Attention, Calculate the sparsity of this layer based on attn_weights_postvision
        '''
        # # get max scores from one row
        # max_scores, _ = torch.max(
        #     attn_weights_postvison, axis=-1, keepdims=True)
        # # filter the attn_weights_postvison
        # filtered_attn_weights = (
        #     attn_weights_postvison >= 0.01 * max_scores).int()
        # # get nonzero entries
        # nonzero_entries = (filtered_attn_weights > 0).sum(
        #     dim=(-2, -1)).to(torch.float32)
        # matrix_tril = torch.tril(torch.ones(
        #     (q_len, q_len), device=attn_weights_postvison.device))[-post_vision_size:, :]
        # num_elements_denominator = matrix_tril.count_nonzero().to(torch.float32)
        # # get sparsity of each layer
        # sparsity = (num_elements_denominator - nonzero_entries) / \
        #     num_elements_denominator
        # sparsity_layer = sparsity.mean()
        # sparsity_layer_tuple = (sparsity_layer,)
        # return sparsity_layer_tuple

        max_scores = torch.max(attn_weights_postvison, dim=-1, keepdim=True)[0]
        filtered_attn_weights = (attn_weights_postvison >= 0.01 * max_scores).float()
        nonzero_entries = filtered_attn_weights.sum(dim=(-2, -1))
        num_elements_denominator = torch.tensor(
            sum(range(q_len - post_vision_size + 1, q_len + 1)), 
            device=attn_weights_postvison.device,
            dtype=torch.float32
        )
        sparsity = (num_elements_denominator - nonzero_entries) / num_elements_denominator
        sparsity_layer = sparsity.mean()
        
        return (sparsity_layer,)

    def get_budget_layer_and_sorted_attn_kv_indices(
        self,
        vlcache_sparsity_layer_tuple: Tuple[torch.Tensor],
        vlcache_attn_weights_importance_tuple: Tuple[torch.Tensor],
        layers: int,
        past_key_value: Cache,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Called in the forward function of Qwen2Model, after the prefill phase ends, the sparsity of each layer is aggregated 
        to reallocate according to the budget, and the token importance is calculated
        '''
        _, num_k_heads, _, _ = past_key_value.key_cache[0].shape
        sparsity_tensor = torch.stack(
            [layer_sparsity[0].to(past_key_value.key_cache[0].device) for layer_sparsity in vlcache_sparsity_layer_tuple])
        non_sparsity_sum = (1 - sparsity_tensor).sum()
        buget_layers = torch.zeros(layers)

        # different budget for each layer
        if self.vlcache_budget_layer_adaptive:
            # Support different window sizes for each layer
            if self.vlcache_different_window_per_layer:
            # different window size for each layer
                buget_layers = torch.clamp(
                    (1.0 - sparsity_tensor) / non_sparsity_sum * self.vlcache_alpha_sparsity * layers, 
                    min=0.01, 
                    max=1.0)
            else:
            # same window size for each layer , 10% budget for each layer
                buget_layers = torch.clamp(
                    (1.0 - sparsity_tensor) / non_sparsity_sum * self.vlcache_alpha_sparsity * 0.9 * layers, 
                    min=0.01 * 0.9, 
                    max=1.0)
        # same budget for each layer
        else:
            if self.vlcache_different_window_per_layer:
                # different window size for each layer
                buget_layers = torch.full((layers,), self.vlcache_alpha_sparsity, device=sparsity_tensor.device)
            else:
                # same window size for each layer , 10% budget for each layer
                buget_layers = torch.full((layers,), self.vlcache_alpha_sparsity * 0.9, device=sparsity_tensor.device)

        # [ layers , batch_size , head_num , all_kv_len]
        stacked_attn_weights_importance = torch.stack(
            [attn_weights_importance[0].to(past_key_value.key_cache[0].device) for attn_weights_importance in vlcache_attn_weights_importance_tuple])

        if not self.vlcache_head_adaptive:
            # if not head adaptive , sum over head and batch.
            attn_weights_importance_sum_layer = stacked_attn_weights_importance.sum(dim=[
                                                                                    1, 2])
            sorted_attn_kv, sorted_indices = torch.sort(
                attn_weights_importance_sum_layer, dim=-1, descending=True)
        else:
            layers, bsz, num_heads, _ = stacked_attn_weights_importance.shape
            stacked_attn_weights_importance = stacked_attn_weights_importance.view(layers, bsz, num_k_heads, num_heads//num_k_heads,-1).sum(dim=3)
            sorted_attn_kv, sorted_indices = torch.sort(
                stacked_attn_weights_importance, dim=-1, descending=True)
        return buget_layers, sorted_indices

    def get_token_importance(
        self,
        attn_weights_postvison: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Called in the forward function of Qwen2Attention , Calculate the token importance based on the attn_weights_postvisa of the current header
        '''
        attn_weights_importance = attn_weights_postvison.sum(dim=-2)
        attn_weights_importance_tuple = (attn_weights_importance,)

        return attn_weights_importance_tuple

    def prefill_update_head(
        self,
        past_key_value: Cache,
        layer_idx: int,
        sorted_attn_kv_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Create new tensors to store the reordered states
        past_key_states = past_key_value.key_cache[layer_idx]
        past_value_states = past_key_value.value_cache[layer_idx]
        batch_size = past_key_states.shape[0]
        num_heads = past_key_states.shape[1]
        head_dim = past_key_states.shape[-1]
        seq_len = sorted_attn_kv_indices.shape[-1]

        # # Initialize new tensors with the right shape
        # new_key_states = torch.zeros((batch_size, num_heads, seq_len, head_dim),
        #                              dtype=past_key_states.dtype,
        #                              device=past_key_states.device)
        # new_value_states = torch.zeros((batch_size, num_heads, seq_len, head_dim),
        #                                dtype=past_value_states.dtype,
        #                                device=past_value_states.device)

        # # Reorder each head separately
        # for head_idx in range(num_heads):
        #     new_key_states[:, head_idx] = past_key_states[:,
        #                                                   head_idx, sorted_attn_kv_indices[head_idx]]
        #     new_value_states[:, head_idx] = past_value_states[:,
        #                                                       head_idx, sorted_attn_kv_indices[head_idx]]

        # # Update the cache
        # past_key_value.key_cache[layer_idx] = new_key_states
        # past_key_value.value_cache[layer_idx] = new_value_states

        expanded_indices = sorted_attn_kv_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, num_heads, seq_len, head_dim)

        new_key_states = torch.gather(past_key_states, dim=2, index=expanded_indices)
        new_value_states = torch.gather(past_value_states, dim=2, index=expanded_indices)

        past_key_value.key_cache[layer_idx] = new_key_states
        past_key_value.value_cache[layer_idx] = new_value_states


class LOOK_MCluster():
    def __init__(self, hh_size=4, recent_size=512, k_seq_dim=2, v_seq_dim=2, hh_ratio=None, recent_ratio=None, layer_idx=None, budget=None, merge=True):
        self.hh_size = hh_size
        self.recent_size = recent_size
        # self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.importance = None
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        # self.image_save_ratio = None
        self.budget = budget
        self.layer_idx = layer_idx
        if self.budget is not None and self.hh_ratio is not None and self.recent_ratio is not None:
            raise ValueError(
                'budget, hh_ratio, recent_ratio can not be set at the same time!')
        self.merge = merge

    def reset(self):
        self.importance = None
        self.seq_len = None

    def get_importance(self, attn_score_cache, num_key_value_groups):

        num_new_tokens = attn_score_cache.shape[2]
        total_heads = attn_score_cache.shape[1]
        kv_heads = total_heads // num_key_value_groups

        if self.importance is None:
            origin = attn_score_cache.sum(0).sum(1)   
            self.importance = origin.view(
                kv_heads, total_heads // kv_heads, -1).mean(dim=1)  
        else:
            # breakpoint() # check here
            raise NotImplementedError('Look-m only processes the KV-cache during the prefill stage. Check the external code.')
         
    def update(self, attn_score_cache, num_key_value_groups):   
        if self.hh_ratio is not None and self.recent_ratio is not None and attn_score_cache.shape[-2] > 1:
            self.hh_size = max(1, int(attn_score_cache.shape[-1] * self.hh_ratio))
            self.recent_size = max(1, int(
                attn_score_cache.shape[-1] * self.recent_ratio))  
            self.budget = self.hh_size + self.recent_size
            
        elif self.budget is not None:
            self.hh_size = max(1, int(attn_score_cache.shape[-1] * self.budget * 0.9))
            self.recent_size = max(1, int(
                attn_score_cache.shape[-1] * self.budget * 0.1))
            self.budget = self.hh_size + self.recent_size
        self.get_importance(attn_score_cache, num_key_value_groups)

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups, text_mask, head_dim, dropout, training, origin_key_states, origin_value_states, merge=True):
        assert len(text_mask) == 1, 'only support batch_size == 1'
        q_len = query_states.shape[-2]

        '''
        key_states is repeated. If key_states is None, we use origin_key_states here.
        '''
        if key_states is None:
            key_states = repeat_kv(origin_key_states, num_key_value_groups)
            
        attn_score_cache = torch.matmul(
            query_states.float(), key_states.transpose(2, 3).float()) / math.sqrt(head_dim)

        if attention_mask is None:
            mask = torch.triu(
            torch.ones((q_len, q_len), dtype=torch.bool, device=query_states.device),
            diagonal=1
        )
            attn_score_cache = attn_score_cache.masked_fill(mask, float('-inf'))
            del mask        
        else:
            attn_score_cache = attn_score_cache + attention_mask
            del attention_mask
        attn_score_cache = nn.functional.softmax(
            attn_score_cache, dim=-1, dtype=torch.float32)
        if dropout is not None:
            attn_score_cache = nn.functional.dropout(
            attn_score_cache, p=dropout, training=training)

        self.update(attn_score_cache, num_key_value_groups)
        del attn_score_cache
        seq_len = origin_key_states.size(self.k_seq_dim)
        if seq_len <= self.budget:  
            return origin_key_states, origin_value_states, None, None
        bsz, num_heads, _, head_dim = origin_key_states.shape

        ################################# only-image######################################
        make_image_less_important = torch.full(
            (self.importance.shape[0], self.importance.shape[1]), 0) 
        if type(text_mask[0]) == torch.Tensor:
            text_image_mask = text_mask[0].tolist()
        else:
            text_image_mask = text_mask[0]
        image_mask = [False if item is True else True for item in text_image_mask]
        make_image_less_important[:, image_mask] = -10000
        make_image_less_important = make_image_less_important.to(
            device=self.importance.device, dtype=self.importance.dtype)
        make_image_less_important[:, -self.recent_size:] = -10000
        self.importance = self.importance + \
            make_image_less_important  
        _, keep_topk = torch.topk(
            self.importance[:, :-self.recent_size], self.hh_size, dim=-1)  
        keep_topk = keep_topk.sort().values
        # mask those keeping tok
        self.importance.scatter_(1, keep_topk, 10000)  
        self.importance[:, -self.recent_size:] = 10000 
        mask = self.importance == 10000 
        expanded_mask = mask.unsqueeze(
            0).unsqueeze(-1).expand_as(origin_key_states) 
        k_hh_recent = torch.masked_select(
            origin_key_states, expanded_mask).view(bsz, num_heads, -1, head_dim)
        v_hh_recent = torch.masked_select(
            origin_value_states, expanded_mask).view(bsz, num_heads, -1, head_dim)
        ############################### only-image-merge###################################
        # applying merge here
        if self.merge:
            k_hh_pruned = torch.masked_select(origin_key_states, ~expanded_mask).view(
                bsz, num_heads, -1, head_dim)  
            k_hh_pruned = k_hh_pruned.float()
            k_hh_recent = k_hh_recent.float()
            similarity = (k_hh_pruned / torch.norm(k_hh_pruned, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)) @ ((k_hh_recent / (torch.norm(
                k_hh_recent, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128))).transpose(-1, -2)) 
            max_values, max_indices = similarity.max(
                dim=-1) 
            merged_indices = max_indices.unsqueeze(-1).repeat(1, 1, 1, 128)
            k_hh_selected = torch.gather(
                input=k_hh_recent, dim=2, index=merged_indices)
            k_hh_merged = (k_hh_pruned + k_hh_selected)/2
            # include_self=True seems decrease the performance
            k_hh_recent = torch.scatter_reduce(
                input=k_hh_recent.float(), dim=2, index=merged_indices, src=k_hh_merged.float(), reduce='mean', include_self=True).to(torch.bfloat16)
            
            v_hh_pruned = origin_value_states.squeeze(
            )[~mask].view(bsz, num_heads, -1, head_dim)
            v_hh_selected = torch.gather(
                input=v_hh_recent, dim=2, index=merged_indices)
            v_hh_merged = (v_hh_pruned + v_hh_selected)/2
            v_hh_recent = torch.scatter_reduce(
                input=v_hh_recent.float(), dim=2, index=merged_indices, src=v_hh_merged.float(), reduce='mean', include_self=True).to(torch.bfloat16)
        return k_hh_recent, v_hh_recent

    def update_kv_only_merge_visual(self, key_states, query_states, value_states, attention_mask, num_key_value_groups, text_mask, head_dim, dropout, training, origin_key_states, origin_value_states, merge=True):
        assert len(text_mask) == 1, 'only support batch_size == 1'
        
        q_len = query_states.shape[-2]

        '''
        key_states is repeated. If key_states is None, we use origin_key_states here.
        '''
        if key_states is None:
            key_states = repeat_kv(origin_key_states, num_key_value_groups)
            
        attn_score_cache = torch.matmul(
            query_states.float(), key_states.transpose(2, 3).float()) / math.sqrt(head_dim)

        if attention_mask is None:
            mask = torch.triu(
            torch.ones((q_len, q_len), dtype=torch.bool, device=query_states.device),
            diagonal=1
        )
            attn_score_cache = attn_score_cache.masked_fill(mask, float('-inf'))
            del mask        
        else:
            attn_score_cache = attn_score_cache + attention_mask
            del attention_mask
        attn_score_cache = nn.functional.softmax(
            attn_score_cache, dim=-1, dtype=torch.float32)
        if dropout is not None:
            attn_score_cache = nn.functional.dropout(
            attn_score_cache, p=dropout, training=training)

        self.update(attn_score_cache)
        del attn_score_cache
        seq_len = origin_key_states.size(self.k_seq_dim)

        if seq_len <= self.budget: 
            return origin_key_states, origin_value_states, None, None
        
        bsz, num_heads, _, head_dim = origin_key_states.shape
        ################################# only-image######################################
        make_image_less_important = torch.full(
            (self.importance.shape[0], self.importance.shape[1]), 0) 
        if type(text_mask[0]) == torch.Tensor:
            text_image_mask = text_mask[0].tolist()
        else:
            text_image_mask = text_mask[0]
        image_mask = [False if item is True else True for item in text_image_mask]
        make_image_less_important[:, image_mask] = -10000
        make_image_less_important = make_image_less_important.to(
            device=self.importance.device, dtype=self.importance.dtype)
        make_image_less_important[:, -self.recent_size:] = -10000
        self.importance = self.importance + \
            make_image_less_important 
        _, keep_topk = torch.topk(
            self.importance[:, :-self.recent_size], self.hh_size, dim=-1) 
        keep_topk = keep_topk.sort().values  
        self.importance.scatter_(1, keep_topk, 10000)  
        self.importance[:, -self.recent_size:] = 10000  
        mask = self.importance == 10000 
        expanded_mask = mask.unsqueeze(
            0).unsqueeze(-1).expand_as(origin_key_states)  
        k_hh_recent = torch.masked_select(
            origin_key_states, expanded_mask).view(bsz, num_heads, -1, head_dim)
        v_hh_recent = torch.masked_select(
            origin_value_states, expanded_mask).view(bsz, num_heads, -1, head_dim)
        ############################### only-image-merge###################################
        k_hh_pruned = torch.masked_select(origin_key_states, ~expanded_mask).view(
            bsz, num_heads, -1, head_dim)  # pruned tokens
        v_hh_pruned = torch.masked_select(origin_value_states, ~expanded_mask).view(
            bsz, num_heads, -1, head_dim)
        
        visual_tokens_mask = torch.tensor(image_mask, device=mask.device)[keep_topk]  
        num_heads = visual_tokens_mask.shape[0]
        padding = torch.tensor(image_mask[-self.recent_size:], device=mask.device).unsqueeze(0).repeat(num_heads, 1) 
        visual_tokens_mask = torch.cat([visual_tokens_mask, padding], dim=1)   
        visual_tokens_mask = visual_tokens_mask.unsqueeze(0).unsqueeze(-1).expand(bsz, num_heads, -1, head_dim)
        text_tokens_mask = ~visual_tokens_mask
        
        pruned_tokens_indices = (~mask).nonzero(as_tuple=True)[1].view(mask.shape[0], -1)  
        pruned_visual_mask = torch.tensor(image_mask, device=mask.device)[pruned_tokens_indices]
        pruned_visual_mask = pruned_visual_mask.unsqueeze(0).unsqueeze(-1).expand(bsz, num_heads, -1, head_dim)
        pruned_text_mask = ~pruned_visual_mask

        k_hh_recent_visual = torch.masked_select(k_hh_recent, visual_tokens_mask).view(bsz, num_heads, -1, head_dim)
        v_hh_recent_visual = torch.masked_select(v_hh_recent, visual_tokens_mask).view(bsz, num_heads, -1, head_dim)
        k_hh_recent_text = torch.masked_select(k_hh_recent, text_tokens_mask).view(bsz, num_heads, -1, head_dim)
        v_hh_recent_text = torch.masked_select(v_hh_recent, text_tokens_mask).view(bsz, num_heads, -1, head_dim)

        k_hh_pruned_visual = torch.masked_select(k_hh_pruned, pruned_visual_mask).view(bsz, num_heads, -1, head_dim)
        v_hh_pruned_visual = torch.masked_select(v_hh_pruned, pruned_visual_mask).view(bsz, num_heads, -1, head_dim)
        k_hh_pruned_text = torch.masked_select(k_hh_pruned, pruned_text_mask).view(bsz, num_heads, -1, head_dim)
        v_hh_pruned_text = torch.masked_select(v_hh_pruned, pruned_text_mask).view(bsz, num_heads, -1, head_dim)

        if k_hh_recent_visual.shape[2] != 0 and k_hh_pruned_visual.shape[2] != 0:
            k_hh_pruned_visual = k_hh_pruned_visual.float()
            k_hh_recent_visual = k_hh_recent_visual.float()
            
            similarity_visual = (k_hh_pruned_visual / torch.norm(k_hh_pruned_visual, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)) @ (
                (k_hh_recent_visual / torch.norm(k_hh_recent_visual, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)).transpose(-1, -2)
            )
            
            max_indices_visual = similarity_visual.max(dim=-1).indices
            merged_indices_visual = max_indices_visual.unsqueeze(-1).repeat(1, 1, 1, 128)

            k_hh_selected_visual = torch.gather(input=k_hh_recent_visual, dim=2, index=merged_indices_visual)
            k_hh_merged_visual = (k_hh_pruned_visual + k_hh_selected_visual)/2
            k_hh_recent_visual = torch.scatter_reduce(
                input=k_hh_recent_visual.float(), 
                dim=2, 
                index=merged_indices_visual, 
                src=k_hh_merged_visual.float(), 
                reduce='mean', 
                include_self=True
            ).to(torch.bfloat16)
            
            v_hh_selected_visual = torch.gather(input=v_hh_recent_visual, dim=2, index=merged_indices_visual)
            v_hh_merged_visual = (v_hh_pruned_visual + v_hh_selected_visual)/2
            v_hh_recent_visual = torch.scatter_reduce(
                input=v_hh_recent_visual.float(), 
                dim=2, 
                index=merged_indices_visual, 
                src=v_hh_merged_visual.float(), 
                reduce='mean', 
                include_self=True
            ).to(torch.bfloat16)

        if k_hh_recent_text.shape[2] != 0 and k_hh_pruned_text.shape[2] != 0:
            k_hh_pruned_text = k_hh_pruned_text.float()
            k_hh_recent_text = k_hh_recent_text.float()
            
            similarity_text = (k_hh_pruned_text / torch.norm(k_hh_pruned_text, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)) @ (
                (k_hh_recent_text / torch.norm(k_hh_recent_text, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)).transpose(-1, -2)
            )
            
            max_indices_text = similarity_text.max(dim=-1).indices
            merged_indices_text = max_indices_text.unsqueeze(-1).repeat(1, 1, 1, 128)

            k_hh_selected_text = torch.gather(input=k_hh_recent_text, dim=2, index=merged_indices_text)
            k_hh_merged_text = (k_hh_pruned_text + k_hh_selected_text)/2
            k_hh_recent_text = torch.scatter_reduce(
                input=k_hh_recent_text.float(), 
                dim=2, 
                index=merged_indices_text, 
                src=k_hh_merged_text.float(), 
                reduce='mean', 
                include_self=True
            ).to(torch.bfloat16)
            
            v_hh_selected_text = torch.gather(input=v_hh_recent_text, dim=2, index=merged_indices_text)
            v_hh_merged_text = (v_hh_pruned_text + v_hh_selected_text)/2
            v_hh_recent_text = torch.scatter_reduce(
                input=v_hh_recent_text.float(), 
                dim=2, 
                index=merged_indices_text, 
                src=v_hh_merged_text.float(), 
                reduce='mean', 
                include_self=True
            ).to(torch.bfloat16)

        k_hh_recent = torch.masked_scatter(k_hh_recent, visual_tokens_mask, k_hh_recent_visual.flatten())
        k_hh_recent = torch.masked_scatter(k_hh_recent, text_tokens_mask, k_hh_recent_text.flatten())
        v_hh_recent = torch.masked_scatter(v_hh_recent, visual_tokens_mask, v_hh_recent_visual.flatten())
        v_hh_recent = torch.masked_scatter(v_hh_recent, text_tokens_mask, v_hh_recent_text.flatten())
        
        
        return k_hh_recent, v_hh_recent



class SnapKVCluster():
    def __init__(self,
                 query_len,
                 budgets,
                 window_size_budgets=0.1,
                 head_adaptive=True,
                 kernel_size=5,
                 pooling='avgpool',
                 merge=None):
        self.query_len = query_len
        self.budgets = budgets 
        self.max_capacity_prompt = max(1, int(query_len * budgets))

        self.window_size_budgets = window_size_budgets

        self.window_size = max(1, int(self.max_capacity_prompt * window_size_budgets))
        self.head_adaptive = head_adaptive
        self.pooling = pooling
        self.kernel_size = kernel_size
        self.merge = merge

    def reset(self,
              budgets,
              window_size_budgets=0.1,
              head_adaptive=True,
              kernel_size=5,
              pooling='avgpool',
              merge=None):

        self.query_len = None

        self.budgets = budgets 
        self.window_size_budgets = window_size_budgets

        self.window_size = None
        self.max_capacity_prompt = None
        self.merge = merge
        self.pooling = pooling,
        self.kernel_size = kernel_size,
        self.head_adaptive = head_adaptive

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):

        bsz, num_heads, q_len, head_dim = query_states.shape
        assert key_states.shape[1] == num_heads // num_key_value_groups
        key_states_repeat = repeat_kv(key_states, num_key_value_groups)
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(
                query_states[..., -self.window_size:, :].float(), key_states_repeat.transpose(2, 3).float()) / math.sqrt(head_dim)   # float32
            mask = torch.full((self.window_size, self.window_size), torch.finfo(
                attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (
                mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -
                         self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32)
            # bsz, num_heads, window_size, seq_len

            attn_weights = attn_weights.view(
                bsz, num_heads // num_key_value_groups, num_key_value_groups, -1, key_states.shape[-2])

            if self.head_adaptive:
                attn_weights_sum = attn_weights[...,
                                                :-self.window_size].sum(dim=[-3, -2])
            else:
                attn_weights_sum = attn_weights[..., :-self.window_size].sum(
                    dim=[1, 2, 3]).unsqueeze(1).expand(-1, num_heads // num_key_value_groups, -1)
            # attn_weights_sum = attn_weights[:, :, -
            #                                 self.window_size:, : -self.window_size].sum(dim=-2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(
                    attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(
                    attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == "no":
                attn_cache = attn_weights_sum
            else:
                raise ValueError('Pooling method not supported')

            indices = attn_cache.topk(
                self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-
                                         self.window_size, :].gather(dim=2, index=indices)
            v_past_compress = value_states[:, :, :-
                                           self.window_size, :].gather(dim=2, index=indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            return key_states, value_states


class pyramidkvCluster():
    def __init__(self,
                 query_len,
                 budgets,
                 window_size_budgets=0.1,
                 head_adaptive=True,
                 num_hidden_layers=28,
                 kernel_size=5,
                 pooling='avgpool',
                 beta=20,
                 layer_idx=None,
                 merge=None):

        self.query_len = query_len
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        self.steps = -1
        self.beta = beta

        self.max_capacity_prompt = max(1, int(query_len * budgets))
        self.window_size_budgets = window_size_budgets
        self.window_size = max(1, int(self.max_capacity_prompt * window_size_budgets))
        self.head_adaptive = head_adaptive
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def reset(self,
              budgets,
              window_size_budgets=0.1,
              head_adaptive=True,
              num_hidden_layers=28,
              kernel_size=5,
              pooling='avgpool',
              beta=20,
              layer_idx=None,
              merge=None):

        self.query_len = None
        self.budgets = budgets 
        self.window_size_budgets = window_size_budgets 
        self.window_size = None
        self.max_capacity_prompt = None
        self.merge = merge
        self.pooling = pooling,
        self.kernel_size = kernel_size,
        self.head_adaptive = head_adaptive,
        self.layer_idx = layer_idx
        self.beta = beta
        self.num_hidden_layers = num_hidden_layers

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):

        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        min_num = (self.max_capacity_prompt - self.window_size) // self.beta
        max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num

        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (self.max_capacity_prompt -
                       self.window_size) * 2 - max_num

        key_states_repeat = repeat_kv(key_states, num_key_value_groups)

        steps = (max_num - min_num) // (self.num_hidden_layers - 1)
        max_capacity_prompt = max_num - self.layer_idx * steps
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        elif q_len < (self.max_capacity_prompt - self.window_size) * 2:

            attn_weights = torch.matmul(
                query_states[..., -self.window_size:, :].float(), key_states_repeat.transpose(2, 3).float()) / math.sqrt(head_dim)   # float32

            mask = torch.full((self.window_size, self.window_size), torch.finfo(
                attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (
                mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -
                         self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32)
            attn_weights = attn_weights.view(
                bsz, num_heads // num_key_value_groups, num_key_value_groups, -1, key_states.shape[-2])
            if self.head_adaptive:
                attn_weights_sum = attn_weights[...,
                                                :-self.window_size].sum(dim=[-3, -2])
            else:
                attn_weights_sum = attn_weights[..., :-self.window_size].sum(
                    dim=[1, 2, 3]).unsqueeze(1).expand(-1, num_heads // num_key_value_groups, -1)

            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(
                    attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(
                    attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == "no":
                attn_cache = attn_weights_sum
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(
                self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            k_past_compress = key_states[:, :, :-
                                         self.window_size, :].gather(dim=2, index=indices)
            v_past_compress = value_states[:, :, :-
                                           self.window_size, :].gather(dim=2, index=indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            return key_states, value_states
        else:
            attn_weights = torch.matmul(
                query_states[..., -self.window_size:, :].float(), key_states_repeat.transpose(2, 3).float()) / math.sqrt(head_dim)   # float32
            mask = torch.full((self.window_size, self.window_size), torch.finfo(
                attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (
                mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -
                         self.window_size:] += attention_mask
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32)
            attn_weights = attn_weights.view(
                bsz, num_heads // num_key_value_groups, num_key_value_groups, -1, key_states.shape[-2])
            if self.head_adaptive:
                attn_weights_sum = attn_weights[...,
                                                :-self.window_size].sum(dim=[-3, -2])
            else:
                attn_weights_sum = attn_weights[..., :-self.window_size].sum(
                    dim=[1, 2, 3]).unsqueeze(1).expand(-1, num_heads // num_key_value_groups, -1)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(
                    attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(
                    attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == "no":
                attn_cache = attn_weights_sum
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(max_capacity_prompt, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-
                                         self.window_size, :].gather(dim=2, index=indices)
            v_past_compress = value_states[:, :, :-
                                           self.window_size, :].gather(dim=2, index=indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            return key_states, value_states

class CSPCluster():
    def __init__(
        self,
        text_image_mask_csp,
        hh_size=4,
        recent_size=512,
        hh_ratio=None,
        recent_ratio=None,
        cross_ratio=0.1,
        kv_recent_bias=1,
        budget=None,
        csp_head_adaptive=False,
    ):
        self.text_image_mask_csp = text_image_mask_csp
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.cache_size = hh_size + recent_size 
        self.hh_ratio = hh_ratio
        self.recent_ratio = recent_ratio
        self.cross_ratio = cross_ratio
        self.kv_recent_bias = kv_recent_bias
        self.budget = budget
        self.csp_head_adaptive = csp_head_adaptive

    def update_kv(self, key_states, value_states, query_states, attn_score_cache, attn_mask):

        if self.cross_ratio is None:
            self.cross_ratio = 0.1
        if self.kv_recent_bias is None:
            self.kv_recent_bias = 1
        
        if attn_score_cache.shape[-2]>1:
            if self.hh_ratio is not None and self.recent_ratio is not None and self.budget is None:
                self.hh_size = max(1, int(attn_score_cache.shape[-1] * self.hh_ratio))
                self.recent_size = max(1, int(attn_score_cache.shape[-1] * self.recent_ratio))
                self.cache_size = self.hh_size + self.recent_size
            elif self.budget is not None:
                self.hh_size = max(1, int(attn_score_cache.shape[-1] * self.budget * 0.9))
                self.recent_size = max(1, int(attn_score_cache.shape[-1] * self.budget * 0.1))
                self.cache_size = self.hh_size + self.recent_size
            else:
                raise ValueError('The CSP method requires setting hh_ratio and recent_ratio at the same time or you can set budget separately.')
            
            ####### add new here ###########################

            assert key_states.shape[-2] == query_states.shape[-2]

            bsz, num_heads, q_len, head_dim = query_states.shape
            _, kv_num_heads, k_len, _ = key_states.shape

            if q_len < self.cache_size:
                return key_states, value_states

            else:
                
                img_mask = self.text_image_mask_csp.clone() # #ALFRED: tensor([112, 142, 163, 191, 213], hh_score: [32, 2789]
                img_mask = img_mask.to(device=attn_score_cache.device, dtype=torch.bool)
                attn_weights = attn_score_cache
                selection_func = 'one-softmax'

                attn_weights = self.apply_selection_function(attn_weights, selection_func)

                q_observe_recent = self.recent_size
                if self.kv_recent_bias != 1:
                    kv_recent_size = max(1, int(self.recent_size * self.kv_recent_bias))
                else:
                    kv_recent_size = self.recent_size
                k = self.cache_size - kv_recent_size # k: 309 self.cache_size: 618 self.recent_size: 309

                attn_weights_past = attn_weights[:, :, -q_observe_recent:, :-kv_recent_size] 
                
                query_img_mask = img_mask[-q_observe_recent:].view(1, 1, -1).expand(bsz, num_heads, -1)  # [B, H, Lq]
                # query_img_mask = img_mask.view(1, 1, -1).expand(bsz, num_heads, -1)  # [B, H, Lq]
                key_img_mask = img_mask[:-kv_recent_size].view(1, 1, -1).expand(bsz, num_heads, -1)  # [B, H, Lk]

                query_img_mask = query_img_mask.unsqueeze(-1)  # [B, H, Lq, 1]
                key_img_mask = key_img_mask.unsqueeze(2)  # [B, H, 1, Lk]

                # self-attn 和 cross-attn mask
                self_attn_mask = (query_img_mask & key_img_mask) | (~query_img_mask & ~key_img_mask)  # [B, H, Lq, Lk]
                cross_attn_mask = (query_img_mask & ~key_img_mask) | (~query_img_mask & key_img_mask)  # [B, H, Lq, Lk]

                attn_weights_self = attn_weights_past * self_attn_mask  # self-attn 
                attn_weights_cross = attn_weights_past * cross_attn_mask  # cross-attn 
                
                k_cross = max(1, int(k * self.cross_ratio))
                k_self = max(1, int(k * (1-self.cross_ratio)))
                if self.cross_ratio <= 0:
                    k_cross, k_self = 1, 308

                # head adaptive
                if self.csp_head_adaptive:
                    attn_weights_self = attn_weights_self.view(bsz, kv_num_heads, -1, attn_weights_self.shape[-2], attn_weights_self.shape[-1])
                    attn_weights_cross = attn_weights_cross.view(bsz, kv_num_heads, -1, attn_weights_cross.shape[-2], attn_weights_cross.shape[-1])
                    attn_weights_self_sum = attn_weights_self.sum(dim=(-2,-3)) # [B, kv_num_heads, L_self]
                    attn_weights_cross_sum = attn_weights_cross.sum(dim=(-2,-3)) # [B, kv_num_heads, L_cross]
                else:
                # layer adaptive
                    attn_weights_self_sum = attn_weights_self.sum(dim=(-2,-3)).unsqueeze(1).expand(-1, kv_num_heads, -1) # [B, kv_num_heads, L_self]
                    attn_weights_cross_sum = attn_weights_cross.sum(dim=(-2,-3)).unsqueeze(1).expand(-1, kv_num_heads, -1) # [B, kv_num_heads, L_cross]
                
                assert attn_weights_self_sum.shape == attn_weights_cross_sum.shape
                assert len(attn_weights_self_sum.shape) == 3
                assert attn_weights_self_sum.shape[1] == kv_num_heads
                
                self_indices = attn_weights_self_sum.topk(k_self, dim=-1).indices  # indices: [B, H, topk]
                cross_indices = attn_weights_cross_sum.topk(k_cross, dim=-1).indices  # indices: [B, H, topk]

                indices = torch.cat([self_indices, cross_indices], dim=-1)

                indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                k_past_compress = key_states[:, :, :-kv_recent_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-kv_recent_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -kv_recent_size:, :]
                v_cur = value_states[:, :, -kv_recent_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)

                return key_states, value_states
        else:
            return key_states, value_states
    
    def apply_selection_function(self,attn_weights, selection_func, tau_init=1.0, tau_delta=0.1, iter_count=0, temp=1.0, query_states=None):
        """
        Apply a selection function to adjust attention weights based on the specified method.
        
        Parameters:
        - attn_weights (Tensor): The attention weights tensor to be transformed.
        - selection_func (str): The selection function to use ('gumbel', 'TaylorV1', 'TaylorV3', 'temp', 'one-softmax').
        - tau_init (float): Initial temperature for Gumbel softmax.
        - tau_delta (float): Delta for temperature adjustment in Gumbel softmax.
        - iter_count (int): Iteration count, used to adjust temperature in Gumbel softmax.
        - temp (float): Temperature for softmax scaling when using 'temp' selection.
        - query_states (Tensor, optional): Query states tensor, used to determine dtype in certain cases.
        
        Returns:
        - Tensor: The adjusted attention weights.
        """
        
        if selection_func == 'gumbel':
            # Apply Gumbel Softmax with dynamic temperature
            tau = tau_init + (iter_count * tau_delta)
            gumbel_score = nn.functional.gumbel_softmax(attn_weights, tau=tau, hard=False, dim=-1)
            return gumbel_score.to(attn_weights.dtype)
        
        elif selection_func == 'TaylorV1':
            # Apply TaylorSoftmaxV1
            taylor_softmax_v1 = TaylorSoftmaxV1(dim=-1, n=4).to(attn_weights.device)
            return taylor_softmax_v1(attn_weights).to(attn_weights.dtype)
        
        elif selection_func == 'TaylorV3':
            # Apply TaylorSoftmaxV3
            taylor_softmax_v3 = TaylorSoftmaxV3(dim=-1, n=4).to(attn_weights.device)
            return taylor_softmax_v3(attn_weights).to(attn_weights.dtype)
        
        elif selection_func == 'temp':
            # Apply scaled softmax with specified temperature
            return (nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32) / temp).to(query_states.dtype)
        
        elif selection_func == 'one-softmax':
            # Apply one-softmax using custom softmax function 'softmax_n'
            return softmax_n(attn_weights, n=1, dim=-1, dtype=torch.float32)
        
        else:
            # Default to standard softmax
            return nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)


class RandomCluster():
    def __init__(self, budgets):
        self.budgets = budgets
    
    def update_kv(self, origin_key_states, origin_value_states):
        bsz, num_heads, seq_len, head_dim = origin_key_states.shape
        window_size = max(1, int(seq_len * self.budgets * 0.1))
        other_size = max(1, int(seq_len * self.budgets * 0.9))

        select_from_len = seq_len - window_size

        indices = torch.stack([
            torch.sort(torch.randperm(select_from_len, device=origin_key_states.device)[:other_size])[0]
            for _ in range(bsz * num_heads)
        ]).view(bsz, num_heads, other_size)

        last_indices = torch.arange(seq_len - window_size, seq_len, 
                              device=origin_key_states.device)
        last_indices = last_indices.unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, -1)

        indices = torch.cat([indices, last_indices], dim=2)
        
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        selected_key_states = torch.gather(origin_key_states, dim=2, index=indices)
        selected_value_states = torch.gather(origin_value_states, dim=2, index=indices)

        return selected_key_states, selected_value_states


def init_StreamingLLM(self,
                      query_len,
                      window_size_budgets=0.1,
                      budgets=0.3,
                      merge=None,
                      ):

    self.kv_cluster = StreamingLLMKVCluster(
        query_len=query_len,
        budgets=budgets,
        window_size_budgets=window_size_budgets,
        merge=merge,
    )


def init_H2O(self,
             query_len,
             head_adaptive,
             window_size_budgets=0.1,
             budgets=0.3,
             merge=None,
             ):

    self.kv_cluster = H2OKVCluster(
        query_len=query_len,
        budgets=budgets,
        window_size_budgets=window_size_budgets,
        head_adaptive=head_adaptive,
        merge=merge,
    )


def init_Vlcache(self):

    self.kv_cluster = VlCacheKVCluster(
        vlcache_alpha_sparsity=self.vlcache_alpha_sparsity,
        vlcache_different_window_per_layer=self.vlcache_different_window_per_layer,
        vlcache_head_adaptive=self.vlcache_head_adaptive,
        vlcache_budget_layer_adaptive=self.vlcache_budget_layer_adaptive,
    )


def init_LOOK_M(self):
    self.kv_cache = LOOK_MCluster(
        k_seq_dim=2,
        v_seq_dim=2,
        hh_ratio=self.hh_ratio,
        recent_ratio=self.recent_ratio,
        layer_idx=self.layer_idx,
        budget=self.budget,
        merge=self.merge
    )


def init_snapkv(self,
                query_len,
                budgets,
                window_size_budgets=0.1,
                head_adaptive=True,
                kernel_size=5,
                pooling='avgpool',
                merge=None
                ):

    self.kv_cluster = SnapKVCluster(
        query_len=query_len,
        budgets=budgets,
        window_size_budgets=window_size_budgets,
        head_adaptive=head_adaptive,
        kernel_size=kernel_size,
        pooling=pooling,
        merge=merge,
    )


def init_pyramidkv(self,
                   query_len,
                   budgets,
                   window_size_budgets=0.1,
                   head_adaptive=True,
                   num_hidden_layers=28,
                   kernel_size=5,
                   pooling='avgpool',
                   beta=20,
                   layer_idx=None,
                   merge=None
                   ):

    self.kv_cluster = pyramidkvCluster(
        query_len=query_len,
        num_hidden_layers=num_hidden_layers,
        layer_idx=layer_idx,
        budgets=budgets,
        window_size_budgets=window_size_budgets,
        head_adaptive = head_adaptive,
        kernel_size=kernel_size,
        pooling=pooling,
        beta=beta,      
        merge=merge,
    )

def init_CSP(self):
    
    # need mask that text if false and image is true , batch_size = 1
    tensor = torch.tensor(self.text_image_mask[0], dtype=torch.bool)
    text_image_mask_csp = ~ tensor

    self.kv_cluster = CSPCluster(
        
        text_image_mask_csp = text_image_mask_csp,
        hh_size = 4,
        recent_size = 512,
        hh_ratio = self.hh_ratio,
        recent_ratio = self.recent_ratio,
        cross_ratio = self.cross_ratio,
        kv_recent_bias = self.kv_recent_bias,
        budget = self.budgets,
        csp_head_adaptive = self.csp_head_adaptive,
        )

def init_Random(self):
    self.kv_cache = RandomCluster(budgets = self.budgets)