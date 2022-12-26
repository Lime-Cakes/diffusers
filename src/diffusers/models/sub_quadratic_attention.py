# original source:
#   https://github.com/AminRezaei0x443/memory-efficient-attention/blob/1bc0d9e6ac5f82ea43a375135c4e1d3896ee1694/memory_efficient_attention/attention_torch.py
# license:
#   unspecified
# credit:
#   Amin Rezaei (primary author)
#   xloem
#     calc_fn
#     sparse broadcasting for bias, mask, weights
#     flattened conditions for clarity
#   Hyungon Ryu (device arg fix)
#   Alex Birch
#     option to forego checkpointing (not needed during inference)
#     MPS support
#     optimizations (batched matmul, fused multiply) (at the expense of support for mask + bias)
# implementation of:
#   Self-attention Does Not Need O(n2) Memory":
#   https://arxiv.org/abs/2112.05682v2

from functools import partial
import torch
from torch.utils.checkpoint import checkpoint
from ..utils.attention_slicing_utils import dynamic_slice, map_pt, scan
import math
from typing import Optional


def _query_chunk_attention(query_idx, query, key, value,
                           mask, bias,
                           key_chunk_size: Optional[int]=None,
                           mask_calc_fn=None,
                           bias_calc_fn=None,
                           weights_calc_fn=None,
                           calc_fn_data=None,
                           use_checkpoint=True):
    num_kv, num_heads, k_features = key.shape[-3:]
    v_features = value.shape[-1]
    num_q = query.shape[-3]
    key_chunk_size = min(key_chunk_size or int(math.sqrt(num_kv)), num_kv)
    scale = k_features ** -0.5

    def summarize_chunk(key_idx, query, key, value, mask, bias):
        attn_weights = torch.baddbmm(
            torch.empty(1, 1, 1, device=query.device, dtype=query.dtype),
            query,
            key.transpose(1,2),
            alpha=scale,
            beta=0,
        )
        if bias_calc_fn is not None:
            raise "bias_calc_fn no longer supported" # lost support as a result of migrating to 3D tensors; needs to be reimplemented
            bias = bias_calc_fn(query_idx, key_idx, bias, attn_weights, calc_fn_data)
        if bias is not None:
            raise "bias no longer supported" # lost support as a result of migrating to 3D tensors; needs to be reimplemented
            bias = torch.einsum('...hqk->...qhk', bias)
            attn_weights = attn_weights + bias
        if mask_calc_fn is not None:
            raise "mask_calc_fn no longer supported" # lost support as a result of migrating to 3D tensors; needs to be reimplemented
            mask = mask_calc_fn(query_idx, key_idx, mask, attn_weights, calc_fn_data)
        if mask is not None:
            raise "mask no longer supported" # lost support as a result of migrating to 3D tensors; needs to be reimplemented
            big_neg = torch.finfo(attn_weights.dtype).min
            big_neg = torch.tensor(big_neg, device=mask.device, dtype=torch.float32)
            mask = torch.einsum('...hqk->...qhk', mask)
            attn_weights = torch.where(mask, attn_weights, big_neg)
        if weights_calc_fn is not None:
            raise "weights_calc_fn no longer supported" # lost support as a result of migrating to 3D tensors; needs to be reimplemented
            attn_weights = weights_calc_fn(query_idx, key_idx, attn_weights, calc_fn_data)
        max_score, _ = torch.max(attn_weights, -1, keepdim=True)
        max_score = max_score.detach()
        exp_weights = torch.exp(attn_weights - max_score)
        exp_values = torch.bmm(exp_weights, value)
        max_score = max_score.squeeze(-1)
        return exp_values, exp_weights.sum(dim=-1), max_score
    summarizer = partial(checkpoint, summarize_chunk) if use_checkpoint else summarize_chunk

    def chunk_scanner(chunk_idx):
        key_chunk = dynamic_slice(key, tuple([0] * (key.ndim - 3)) + (chunk_idx, 0, 0),
                                  tuple(key.shape[:-3]) + (key_chunk_size, num_heads, k_features))
        value_chunk = dynamic_slice(value, tuple([0] * (value.ndim - 3)) + (chunk_idx, 0, 0),
                                    tuple(value.shape[:-3]) + (key_chunk_size, num_heads, v_features))

        if bias is None:
            bias_chunk = None
        elif bias.shape[-1] == 1:
            bias_chunk = bias
        elif bias.shape[-1] == num_kv:
            bias_chunk = dynamic_slice(bias, tuple([0] * (bias.ndim - 3)) + (0, 0, chunk_idx),
                                       tuple(bias.shape[:-3]) + (bias.shape[-3], bias.shape[-2], key_chunk_size))
        else:
            raise TypeError(f'bias.shape[-1] == {bias.shape[-1]} must broadcast with key.shape[-3] == {num_kv}')

        if mask is None:
            mask_chunk = None
        elif mask.shape[-1] == 1:
            mask_chunk = mask
        elif mask.shape[-1] == num_kv:
            mask_chunk = dynamic_slice(mask, tuple([0] * (mask.ndim - 3)) + (0, 0, chunk_idx),
                                       tuple(mask.shape[:-3]) + (mask.shape[-3], mask.shape[-2], key_chunk_size))
        else:
            raise TypeError(f'bias.shape[-1] == {bias.shape[-1]} must broadcast with key.shape[-3] == {num_kv}')

        return summarizer(chunk_idx, query, key_chunk, value_chunk, mask_chunk, bias_chunk)

    chunk_values, chunk_weights, chunk_max = map_pt(
        chunk_scanner, xs=torch.arange(0, num_kv, key_chunk_size))

    global_max, _ = torch.max(chunk_max, 0, keepdim=True)
    max_diffs = torch.exp(chunk_max - global_max)
    chunk_values *= torch.unsqueeze(max_diffs, -1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(dim=0)
    all_weights = torch.unsqueeze(chunk_weights, -1).sum(dim=0)
    return all_values / all_weights


def efficient_dot_product_attention(query, key, value,
                                    mask=None, bias=None,
                                    query_chunk_size=1024,
                                    key_chunk_size: Optional[int] = None,
                                    bias_calc_fn=None,
                                    mask_calc_fn=None,
                                    weights_calc_fn=None,
                                    calc_fn_data=None,
                                    use_checkpoint=True):
    """Computes efficient dot-product attention given query, key, and value.
      This is efficient version of attention presented in
      https://arxiv.org/abs/2112.05682v2 which comes with O(sqrt(n)) memory requirements.
      Args:
        query: queries for calculating attention with shape of
          `[batch * num_heads, tokens, channels_per_head]`.
        key: keys for calculating attention with shape of
          `[batch * num_heads, tokens, channels_per_head]`.
        value: values to be used in attention with shape of
          `[batch * num_heads, tokens, channels_per_head]`.
        bias: bias for the attention weights. This should be broadcastable to the
          shape `[batch..., num_heads, q_length, kv_length]`.
          This can be used for incorporating padding masks, proximity bias, etc.
        mask: mask for the attention weights. This should be broadcastable to the
          shape `[batch..., num_heads, q_length, kv_length]`.
          Attention weights are masked out if their corresponding mask value
          is `False`.
        query_chunk_size: int: query chunks size
        key_chunk_size: int: key chunks size
        bias_calc_fn: a bias calculation callback for each chunk, of form
          `(q_offset, k_offset, bias_chunk, attn_weights, calc_fn_data) -> bias`.
          This can be used for incorporating causal masks, padding masks,
          proximity bias, etc.
        mask_calc_fn: a mask calculation callback for each chunk, of form
          `(q_offset, k_offset, mask_chunk, attn_weights, calc_fn_data) -> mask`.
          This can be used for incorporating causal or other large masks.
          Attention weights are masked out if their corresponding mask value
          is `False`.
        weights_calc_fn: a general attn_weights callback for each chunk, of form
          `(q_offset, k_offset, attn_weights, calc_fn_data) -> attn_weights`.
          attn_weights has shape of
          `[batch..., q_chunk_size, num_heads, k_chunk_size]`.
          This can be used to implement complex weights processing in a memory
          efficient way.
        calc_fn_data: optional pure data to pass to each per-chunk call of
          bias_calc_fn, mask_calc_fn, and weights_calc_fn.
        weights_calc_data: pure_data to pass with each call to weights_calc_fn
        use_checkpoint: bool: whether to use checkpointing (recommended True for training, False for inference)
      Returns:
        Output of shape `[batch * num_heads, query_tokens, channels_per_head]`.
      """
    num_q, num_heads, q_features = query.shape[-3:]
    num_kv = key.shape[-3]

    def chunk_scanner(chunk_idx, _):
        query_chunk = dynamic_slice(query, tuple([0] * (query.ndim - 3)) + (chunk_idx, 0, 0),
                                    tuple(query.shape[:-3]) + (min(query_chunk_size, num_q), num_heads, q_features))

        if mask is None:
            mask_chunk = None
        elif mask.shape[-2] == 1:
            mask_chunk = mask
        elif mask.shape[-2] == num_q:
            mask_chunk = dynamic_slice(mask, tuple([0] * (mask.ndim - 3)) + (0, chunk_idx, 0),
                                       tuple(mask.shape[:-3]) + (mask.shape[-3], min(query_chunk_size, num_q), mask.shape[-1]))
        else:
            raise TypeError(f'mask.shape[-2] == {mask.shape[-2]} must broadcast with query.shape[-3] == {num_q}')

        if bias is None:
            bias_chunk = None
        elif bias.shape[-2] == 1:
            bias_chunk = bias
        elif bias.shape[-2] == num_q:
            bias_chunk = dynamic_slice(bias, tuple([0] * (bias.ndim - 3)) + (0, chunk_idx, 0),
                                       tuple(bias.shape[:-3]) + (bias.shape[-3], min(query_chunk_size, num_q), bias.shape[-1]))
        else:
            raise TypeError(f'bias.shape[-2] == {bias.shape[-2]} must broadcast with query.shape[-3] == {num_q}')
        return (chunk_idx + query_chunk_size,
                _query_chunk_attention(chunk_idx, query_chunk, key, value, mask_chunk, bias_chunk, key_chunk_size=key_chunk_size,
                                       bias_calc_fn=bias_calc_fn, mask_calc_fn=mask_calc_fn,
                                       weights_calc_fn=weights_calc_fn, calc_fn_data=calc_fn_data, use_checkpoint=use_checkpoint))

    _, res = scan(chunk_scanner, init=0, xs=None, length=math.ceil(num_q / query_chunk_size))
    rl = [res[i] for i in range(res.shape[0])]
    return torch.cat(rl, dim=-3)
