# original source:
#   https://github.com/AminRezaei0x443/memory-efficient-attention/blob/1bc0d9e6ac5f82ea43a375135c4e1d3896ee1694/memory_efficient_attention/utils.py
# license:
#   unspecified
# credit:
#   Amin Rezaei (primary author)
#   Hyungon Ryu (device arg fix)
#   Alex Birch (typings, deleted everything)
# implementation of:
#   Self-attention Does Not Need O(n2) Memory":
#   https://arxiv.org/abs/2112.05682v2
from torch import Tensor
from typing import List

def dynamic_slice(
    x: Tensor,
    starts: List[int],
    sizes: List[int],
) -> Tensor:
    slicing = [slice(start, start + size + 1) for start, size in zip(starts, sizes)]
    return x[slicing]