# original source:
#   https://github.com/AminRezaei0x443/memory-efficient-attention/blob/1bc0d9e6ac5f82ea43a375135c4e1d3896ee1694/memory_efficient_attention/utils.py
# license:
#   unspecified
# credit:
#   Amin Rezaei (primary author)
#   Hyungon Ryu (device arg fix)
# implementation of:
#   Self-attention Does Not Need O(n2) Memory":
#   https://arxiv.org/abs/2112.05682v2
import torch
import numpy as np


def dynamic_slice(x, starts, sizes):
    # start_indices[i] = clamp(start_indices[i], 0, operand.dimension_size[i] - size_indices[i])
    starts = [np.clip(starts[i], 0, x.shape[i] - sizes[i]) for i in range(len(starts))]
    for i, (start, size) in enumerate(zip(starts, sizes)):
        x = torch.index_select(x, i, torch.tensor(range(start, start + size), device=x.device))
    return x


def map_pt(f, xs):
    t = [f(x) for x in xs]
    return tuple(map(torch.stack, zip(*t)))


def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, torch.stack(ys)