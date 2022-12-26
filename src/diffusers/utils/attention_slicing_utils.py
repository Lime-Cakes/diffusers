# original source:
#   https://github.com/AminRezaei0x443/memory-efficient-attention/blob/1bc0d9e6ac5f82ea43a375135c4e1d3896ee1694/memory_efficient_attention/utils.py
# license:
#   unspecified
# credit:
#   Amin Rezaei (primary author)
#   Hyungon Ryu (device arg fix)
#   Alex Birch (typings)
# implementation of:
#   Self-attention Does Not Need O(n2) Memory":
#   https://arxiv.org/abs/2112.05682v2
import torch
from torch import Tensor
from typing import Protocol, NamedTuple, Iterable, Optional, List, Tuple


def dynamic_slice(
    x: Tensor,
    starts: List[int],
    sizes: List[int],
) -> Tensor:
    slicing = [slice(start, start + size + 1) for start, size in zip(starts, sizes)]
    return x[slicing]

class AttnChunk(NamedTuple):
    exp_values: Tensor
    exp_weights_sum: Tensor
    max_score: Tensor

class ChunkScanner(Protocol):
    @staticmethod
    def __call__(chunk_idx: int) -> AttnChunk: ...

def map_pt(f: ChunkScanner, xs: List[int]) -> Tuple[Tensor, ...]:
    t = [f(x) for x in xs]
    return tuple(map(torch.stack, zip(*t)))


class ScanOutput(NamedTuple):
    carry: int
    y: Tensor

class ScanCallback(Protocol):
    @staticmethod
    def __call__(
        carry: int,
        value: Tensor,
    ) -> ScanOutput: ...


def scan(
    f: ScanCallback,
    init: int,
    xs: Optional[Iterable[Tensor]],
    length: Optional[int] = None
):
    if xs is None:
        xs: List[Tensor] = [None] * length
    carry: int = init
    ys: List[Tensor] = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, torch.stack(ys)