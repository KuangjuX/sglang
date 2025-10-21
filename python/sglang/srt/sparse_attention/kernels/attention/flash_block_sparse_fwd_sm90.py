
import math
from types import SimpleNamespace
from typing import Type, Callable, Optional, Tuple
from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
import cutlass.utils.hopper_helpers as sm90_utils_basic

from sglang.srt.sparse_attention.kernels.attention import hopper_helpers as sm90_utils
from sglang.srt.sparse_attention.kernels.attention import utils
from sglang.srt.sparse_attention.kernels.attention import pipeline
from sglang.srt.sparse_attention.kernels.attention.mask import AttentionMask
from sglang.srt.sparse_attention.kernels.attention.softmax import Softmax
from sglang.srt.sparse_attention.kernels.attention.seqlen_info import SeqlenInfoQK
from sglang.srt.sparse_attention.kernels.attention.block_info import BlockInfo
from sglang.srt.sparse_attention.kernels.attention.pack_gqa import PackGQA
from sglang.srt.sparse_attention.kernels.attention.named_barrier import NamedBarrierFwd
from sglang.srt.sparse_attention.kernels.attention.tile_scheduler import TileSchedulerArguments, SingleTileScheduler, SingleTileLPTScheduler, SingleTileVarlenScheduler, ParamsBase
from sglang.srt.sparse_attention.kernels.attention.flash_fwd import FlashAttentionForwardBase

class FlashBlockSparseFwdSm90(FlashAttentionForwardBase):
    
    arch = 90
    