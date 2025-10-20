from typing import Optional
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

class BlockSparseAttnFun(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        m_block_dim,
        n_block_dim,
        head_mask_type,
        streaming_info,
        base_blockmask,
        max_seqlen_q_,
        max_seqlen_k_,
        p_dropout: float = 0.0,
        softmax_scale: float = None,
        causal: bool = False,
        extact_streaming: bool = False,
        return_softmax: bool = False,
        window_size_left: int = 0,
        window_size_right: int = 0,
        deterministic: bool = False,
    ):
        pass

    @staticmethod
    def backward(
        ctx,
        dout: torch.Tensor,
    ):
        pass

def block_streaming_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    head_mask_type,
    streaming_info,
    max_seqlen_q_,
    max_seqlen_k_,
    p_dropout: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    return_softmax_lse: bool = False,
):
    pass