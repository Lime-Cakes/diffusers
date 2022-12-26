# original source:
#   https://github.com/AminRezaei0x443/memory-efficient-attention/blob/1bc0d9e6ac5f82ea43a375135c4e1d3896ee1694/memory_efficient_attention/attention_torch.py
# license:
#   unspecified
# credit:
#   Amin Rezaei (original author)
#   Alex Birch (optimized algorithm for 3D tensors, at the expense of removing bias, masking and callbacks)
# implementation of:
#   Self-attention Does Not Need O(n2) Memory":
#   https://arxiv.org/abs/2112.05682v2

from functools import partial
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint
import math
from typing import Optional, NamedTuple, Protocol, List
from ..utils.dynamic_slice import dynamic_slice

class AttnChunk(NamedTuple):
    exp_values: Tensor
    exp_weights_sum: Tensor
    max_score: Tensor

class SummarizeChunk(Protocol):
    @staticmethod
    def __call__(
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> AttnChunk: ...

def _query_chunk_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_chunk_size: Optional[int] = None,
    use_checkpoint = True,
):
    batch_x_heads, k_tokens, k_channels_per_head = key.shape
    _, _, v_channels_per_head = value.shape
    key_chunk_size = min(key_chunk_size or int(math.sqrt(k_tokens)), k_tokens)
    scale = k_channels_per_head ** -0.5

    def summarize_chunk(
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> AttnChunk:
        attn_weights = torch.baddbmm(
            torch.empty(1, 1, 1, device=query.device, dtype=query.dtype),
            query,
            key.transpose(1,2),
            alpha=scale,
            beta=0,
        )
        max_score, _ = torch.max(attn_weights, -1, keepdim=True)
        max_score = max_score.detach()
        exp_weights = torch.exp(attn_weights - max_score)
        exp_values = torch.bmm(exp_weights, value)
        max_score = max_score.squeeze(-1)
        return AttnChunk(exp_values, exp_weights.sum(dim=-1), max_score)
    summarizer: SummarizeChunk = partial(checkpoint, summarize_chunk) if use_checkpoint else summarize_chunk

    def chunk_scanner(chunk_idx: int) -> AttnChunk:
        key_chunk = dynamic_slice(
            key,
            (0, chunk_idx, 0),
            (batch_x_heads, key_chunk_size, k_channels_per_head)
        )
        value_chunk = dynamic_slice(
            value,
            (0, chunk_idx, 0),
            (batch_x_heads, key_chunk_size, v_channels_per_head)
        )

        return summarizer(query, key_chunk, value_chunk)

    chunks: List[AttnChunk] = [
        chunk_scanner(chunk) for chunk in torch.arange(0, k_tokens, key_chunk_size)
    ]
    acc_chunk = AttnChunk(*map(torch.stack, zip(*chunks)))
    chunk_values, chunk_weights, chunk_max = acc_chunk

    global_max, _ = torch.max(chunk_max, 0, keepdim=True)
    max_diffs = torch.exp(chunk_max - global_max)
    chunk_values *= torch.unsqueeze(max_diffs, -1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(dim=0)
    all_weights = torch.unsqueeze(chunk_weights, -1).sum(dim=0)
    return all_values / all_weights

class ScannedChunk(NamedTuple):
    chunk_idx: int
    attn_chunk: AttnChunk

def efficient_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_chunk_size=1024,
    key_chunk_size: Optional[int] = None,
    use_checkpoint=True,
):
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
        query_chunk_size: int: query chunks size
        key_chunk_size: int: key chunks size
        use_checkpoint: bool: whether to use checkpointing (recommended True for training, False for inference)
      Returns:
        Output of shape `[batch * num_heads, query_tokens, channels_per_head]`.
      """
    batch_x_heads, q_tokens, q_channels_per_head = query.shape

    def chunk_scanner(chunk_idx: int) -> Tensor:
        query_chunk = dynamic_slice(
            query,
            (0, chunk_idx, 0),
            (batch_x_heads, min(query_chunk_size, q_tokens), q_channels_per_head)
        )

        return _query_chunk_attention(
            query_chunk,
            key,
            value,
            key_chunk_size=key_chunk_size,
            use_checkpoint=use_checkpoint,
        )
    
    res = torch.cat([
        chunk_scanner(i * query_chunk_size) for i in range(math.ceil(q_tokens / query_chunk_size))
    ], dim=1)
    return res
