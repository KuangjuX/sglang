# SPDX-License-Identifier: Apache-2.0
"""Attention layer with Dual chunk flash attention and sparse attention.
"""
import functools
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from sgl_kernel.sparse_flash_attn import (
    convert_vertical_slash_indexes,
    convert_vertical_slash_indexes_mergehead,
    sparse_attn_func,
)

from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_rank
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


logger = logging.getLogger(__name__)


@dataclass
class DualChunkFlashAttentionMetadata:
    """Metadata for FlashAttentionBackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """

    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]] = None
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor] = None
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_seq_len: int = None

    # (batch_size,). The orig sequence length per sequence.
    orig_seq_lens: Optional[List[int]] = None

    # orig_seq_lens stored as a tensor.
    orig_seq_lens_tensor: Optional[torch.Tensor] = None

    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor] = None

    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor] = None
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor] = None

    # Length scaling factor
    scaling_factor: Optional[torch.Tensor] = None

    # (batch_size,). Sequence lengths for intra attention.
    seq_lens_intra: Optional[torch.Tensor] = None

    # Max sequence length for intra attention.
    max_seq_len_intra: Optional[int] = None

    # (batch_size, num_blocks). Block table for intra attention.
    block_tables_intra: Optional[torch.Tensor] = None

    # (batch_size,). Sequence lengths for succ attention.
    seq_lens_succ: Optional[torch.Tensor] = None

    # Max sequence length for succ attention.
    max_seq_len_succ: Optional[int] = None

    # (batch_size, num_blocks). Block table for succ attention.
    block_tables_succ: Optional[torch.Tensor] = None

    # (batch_size,). Sequence lengths for inter attention.
    seq_lens_inter: Optional[torch.Tensor] = None

    # Max sequence length for inter attention.
    max_seq_len_inter: Optional[int] = None


class DualChunkFlashAttentionBackend(AttentionBackend):
    def __init__(
        self,
        model_runner: "ModelRunner",
    ) -> None:
        self.forward_metadata: FlashAttentionMetadata = None
        self.device = model_runner.device
        self.max_context_len = model_runner.model_config.context_len
        self.num_heads = model_runner.model_config.get_num_attention_heads(
            model_runner.server_args.tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            model_runner.server_args.tp_size
        )
        self.head_size = model_runner.model_config.head_dim

        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype
        self.page_size = model_runner.page_size

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        dual_chunk_attention_config = getattr(
            model_runner.model_config.hf_config, "dual_chunk_attention_config", None
        )
        assert dual_chunk_attention_config is not None
        self.chunk_size = dual_chunk_attention_config.get("chunk_size", 8192)
        self.local_size = dual_chunk_attention_config.get("local_size", 1024)
        self.original_max_position_embeddings = dual_chunk_attention_config.get(
            "original_max_position_embeddings", 0
        )
        self.sparse_attention_config = dual_chunk_attention_config.get(
            "sparse_attention_config", None
        )
        if not self.sparse_attention_config:
            logger.warning_once(
                "Sparse attention will not be enabled as "
                "sparse attention config is not provided."
            )
        self.sparse_attention_enabled = dual_chunk_attention_config.get(
            "sparse_attention_enabled", self.sparse_attention_config is not None
        )
        self.sparse_attention_threshold = dual_chunk_attention_config.get(
            "sparse_attention_threshold", 32768
        )
        self.sparse_attention_last_q = dual_chunk_attention_config.get(
            "sparse_attention_last_q", 64
        )
        self.dual_chunk_attention_config = dual_chunk_attention_config

        if self.sparse_attention_enabled:
            self.arange = torch.arange(self.sparse_attention_last_q, device="cuda")
            self.last_q_mask = (
                self.arange[None, None, :, None] >= self.arange[None, None, None, :]
            )

    @functools.lru_cache()
    def get_sparse_attention_config(self, layer_idx) -> List[Dict[str, Any]]:
        layer_sparse_attention_config = {
            int(i): j for i, j in self.sparse_attention_config[layer_idx].items()
        }
        start_head = self.num_heads * get_tensor_model_parallel_rank()
        end_head = start_head + self.num_heads
        return [layer_sparse_attention_config[i] for i in range(start_head, end_head)]

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize forward metadata hence all layers in the forward pass can reuse it."""

        forward_mode: ForwardMode = forward_batch.forward_mode
        assert forward_mode.is_prefill() or forward_mode.is_decode()
        batch_size = forward_batch.batch_size

        metadata = DualChunkFlashAttentionMetadata()
        metadata.seq_lens_tensor = forward_batch.seq_lens.to(torch.int32)
        metadata.seq_lens = forward_batch.seq_lens.tolist()
        metadata.max_seq_len = forward_batch.seq_lens.max().item()

        metadata.orig_seq_lens_tensor = forward_batch.orig_seq_lens
        metadata.orig_seq_lens = forward_batch.orig_seq_lens.tolist()

        metadata.block_tables = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, : metadata.max_seq_len
        ]
        # Convert the block table to a strided format.
        if self.page_size > 1:
            strided_indices = torch.arange(
                0, metadata.block_tables.shape[1], self.page_size, device=self.device
            )
            metadata.block_tables = (
                metadata.block_tables[:, strided_indices] // self.page_size
            )

        metadata.query_start_loc = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=metadata.seq_lens_tensor.device
        )
        if forward_mode.is_prefill():
            metadata.query_start_loc[1:] = torch.cumsum(
                forward_batch.extend_seq_lens.to(torch.int32), dim=0, dtype=torch.int32
            )
        else:
            metadata.query_start_loc[1:] = torch.cumsum(
                torch.arange(
                    batch_size,
                    dtype=metadata.query_start_loc.dtype,
                    device=metadata.query_start_loc.device,
                ),
                dim=0,
                dtype=torch.int32,
            )
        metadata.seq_start_loc = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=metadata.seq_lens_tensor.device
        )
        metadata.seq_start_loc[1:] = torch.cumsum(
            metadata.seq_lens_tensor, dim=0, dtype=torch.int32
        )

        if self.original_max_position_embeddings > 0:
            if forward_mode.is_prefill():
                metadata.scaling_factor = (
                    0.1
                    * torch.log(
                        metadata.orig_seq_lens_tensor
                        / self.original_max_position_embeddings
                    )
                    + 1.0
                ).clip(min=1)
            else:
                metadata.scaling_factor = (
                    0.1
                    * torch.log(
                        metadata.orig_seq_lens_tensor
                        / self.original_max_position_embeddings
                    )
                    + 1.0
                ).clip(min=1)

        if forward_mode.is_decode():
            cache_seq_lens = metadata.orig_seq_lens_tensor

            chunk_len = self.chunk_size - self.local_size
            chunk_num_curr = (cache_seq_lens - 1) // chunk_len

            seq_lens_intra = cache_seq_lens - chunk_num_curr * chunk_len
            max_seq_len_intra = seq_lens_intra.max().item()
            metadata.seq_lens_intra = seq_lens_intra
            metadata.max_seq_len_intra = max_seq_len_intra

            block_tables_intra = torch.zeros(
                batch_size,
                (max_seq_len_intra - 1) // self.page_size + 1,
                dtype=metadata.block_tables.dtype,
                device=metadata.block_tables.device,
            )
            for i in range(batch_size):
                st = chunk_num_curr[i] * chunk_len // self.page_size
                ed = min(
                    st + (max_seq_len_intra - 1) // self.page_size + 1,
                    (cache_seq_lens[i] - 1) // self.page_size + 1,
                )
                block_tables_intra[i, : ed - st] = metadata.block_tables[i, st:ed]
            metadata.block_tables_intra = block_tables_intra

            metadata.seq_lens_succ = (
                chunk_num_curr - (chunk_num_curr - 1).clip(min=0)
            ) * chunk_len
            metadata.max_seq_len_succ = metadata.seq_lens_succ.max().item()
            if metadata.max_seq_len_succ:
                block_tables_succ = torch.zeros(
                    batch_size,
                    (metadata.max_seq_len_succ - 1) // self.page_size + 1,
                    dtype=metadata.block_tables.dtype,
                    device=metadata.block_tables.device,
                )
                for i in range(batch_size):
                    start = (
                        (chunk_num_curr[i] - 1).clip(min=0)
                        * chunk_len
                        // self.page_size
                    )
                    end = min(
                        start + (metadata.max_seq_len_succ - 1) // self.page_size + 1,
                        (cache_seq_lens[i] - 1) // self.page_size + 1,
                    )
                    block_tables_succ[i, : end - start] = metadata.block_tables[
                        i, start:end
                    ]
                metadata.block_tables_succ = block_tables_succ

            metadata.seq_lens_inter = (chunk_num_curr - 1).clip(min=0) * chunk_len
            metadata.max_seq_len_inter = metadata.seq_lens_inter.max().item()

        self.forward_metadata = metadata

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # Use precomputed metadata across all layers
        metadata = self.forward_metadata

        (
            query,
            query_succ,
            query_inter,
            query_succ_critical,
            query_inter_critical,
        ) = torch.split(q, q.shape[-1] // 5, dim=-1)

        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        query_succ = query_succ.view(-1, self.num_heads, self.head_size)
        query_inter = query_inter.view(-1, self.num_heads, self.head_size)
        query_succ_critical = query_succ_critical.view(
            -1, self.num_heads, self.head_size
        )
        query_inter_critical = query_inter_critical.view(
            -1, self.num_heads, self.head_size
        )
        key = k.view(-1, self.num_kv_heads, self.head_size)
        value = v.view(-1, self.num_kv_heads, self.head_size)

        # apply DCA scaling
        if self.original_max_position_embeddings > 0:
            assert metadata.scaling_factor is not None
            assert metadata.query_start_loc is not None
            assert metadata.orig_seq_lens is not None
            current_start = 0
            query_start_loc_cpu = metadata.query_start_loc.cpu()
            for i in range(len(metadata.orig_seq_lens)):
                current_end = (
                    current_start
                    + (query_start_loc_cpu[i + 1] - query_start_loc_cpu[i]).item()
                )
                key[current_start:current_end].mul_(metadata.scaling_factor[i])
                current_start = current_end
            assert current_end <= self.max_context_len

        # Do multi-head attention
        key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )
        key_cache = key_cache.view(
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim
        )
        value_cache = value_cache.view(
            -1, self.page_size, layer.tp_v_head_num, layer.head_dim
        )

        if key is not None and value is not None:
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    key,
                    value,
                    layer.k_scale,
                    layer.v_scale,
                )

        if not save_kv_cache:
            # profile run
            o = flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=metadata.seq_start_loc,
                cu_seqlens_k=metadata.seq_start_loc,
                max_seqlen_q=metadata.max_seq_len,
                max_seqlen_k=metadata.max_seq_len,
                softmax_scale=layer.scaling,
                causal=True,
            )
        else:
            # prefill/chunked-prefill
            # get per layer sparse attention config
            if self.sparse_attention_enabled:
                self.layer_sparse_attention_config = self.get_sparse_attention_config(
                    layer.layer_id
                )
            assert metadata.orig_seq_lens is not None
            o = self._dual_chunk_flash_attn_prefill(
                q=query,
                q_succ=query_succ,
                q_inter=query_inter,
                q_succ_critical=query_succ_critical,
                q_inter_critical=query_inter_critical,
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=metadata.query_start_loc,
                cu_seqlens_k=metadata.seq_start_loc,
                orig_seq_lens=metadata.orig_seq_lens,
                scaling_factor=metadata.scaling_factor,
                softmax_scale=layer.scaling,
                causal=True,
                window_size=(-1, -1),
                block_table=metadata.block_tables,
                chunk_size=self.chunk_size,
                local_size=self.local_size,
            )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ) -> torch.Tensor:
        # Use precomputed metadata across all layers
        metadata = self.forward_metadata

        (
            query,
            query_succ,
            query_inter,
            query_succ_critical,
            query_inter_critical,
        ) = torch.split(q, q.shape[-1] // 5, dim=-1)

        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        query_succ = query_succ.view(-1, self.num_heads, self.head_size)
        query_inter = query_inter.view(-1, self.num_heads, self.head_size)
        query_succ_critical = query_succ_critical.view(
            -1, self.num_heads, self.head_size
        )
        query_inter_critical = query_inter_critical.view(
            -1, self.num_heads, self.head_size
        )
        key = k.view(-1, self.num_kv_heads, self.head_size)
        value = v.view(-1, self.num_kv_heads, self.head_size)

        key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )
        key_cache = key_cache.view(
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim
        )
        value_cache = value_cache.view(
            -1, self.page_size, layer.tp_v_head_num, layer.head_dim
        )

        if key is not None and value is not None:
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    key,
                    value,
                    layer.k_scale,
                    layer.v_scale,
                )

        # apply DCA scaling
        if self.original_max_position_embeddings > 0:
            assert metadata.scaling_factor is not None
            scaling_factor = metadata.scaling_factor
            key.mul_(scaling_factor.unsqueeze(-1).unsqueeze(-1))

        o = self._dual_chunk_flash_attn_decoding(
            query.unsqueeze(1),
            query_succ.unsqueeze(1),
            query_inter.unsqueeze(1),
            key_cache,
            value_cache,
            block_table=metadata.block_tables,
            cache_seqlens=metadata.seq_lens_tensor,
            softmax_scale=layer.scaling,
            causal=True,
            chunk_size=self.chunk_size,
            local_size=self.local_size,
            original_max_position_embeddings=self.original_max_position_embeddings,
            decode_meta=metadata,
        ).squeeze(1)
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Initialize CUDA graph state for the attention backend.

        Args:
            max_bs (int): Maximum batch size to support in CUDA graphs

        This creates fixed-size tensors that will be reused during CUDA graph replay
        to avoid memory allocations.
        """
        self.decode_metadata = {
            "seq_lens_tensor": torch.zeros(
                max_bs, dtype=torch.int32, device=self.device
            ),
            "orig_seq_lens_tensor": torch.zeros(
                max_bs, dtype=torch.int32, device=self.device
            ),
            "scaling_factor": torch.zeros(
                max_bs, dtype=torch.float32, device=self.device
            ),
            "block_tables": torch.zeros(
                max_bs,
                (self.max_context_len - 1) // self.page_size + 1,
                dtype=torch.int32,
                device=self.device,
            ),
            "block_tables_intra": torch.zeros(
                max_bs,
                (self.max_context_len - 1) // self.page_size + 1,
                dtype=torch.int32,
                device=self.device,
            ),
            "seq_lens_intra": torch.zeros(
                max_bs, dtype=torch.int32, device=self.device
            ),
            "block_tables_succ": torch.zeros(
                max_bs,
                (self.max_context_len - 1) // self.page_size + 1,
                dtype=torch.int32,
                device=self.device,
            ),
            "seq_lens_succ": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "seq_lens_inter": torch.zeros(
                max_bs, dtype=torch.int32, device=self.device
            ),
        }

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[None],
    ):
        metadata = DualChunkFlashAttentionMetadata()

        if forward_mode.is_decode_or_idle():
            if self.original_max_position_embeddings > 0:
                metadata.scaling_factor = self.decode_metadata["scaling_factor"][:bs]

            metadata.seq_lens_tensor = self.decode_metadata["seq_lens_tensor"][:bs]
            metadata.orig_seq_lens_tensor = self.decode_metadata[
                "orig_seq_lens_tensor"
            ][:bs]
            metadata.max_seq_len = self.max_context_len
            metadata.block_tables = self.decode_metadata["block_tables"][
                req_pool_indices, :
            ]

            # intra
            metadata.max_seq_len_intra = self.max_context_len
            metadata.seq_lens_intra = self.decode_metadata["seq_lens_intra"][:bs]

            metadata.block_tables_intra = self.decode_metadata["block_tables_intra"][
                :bs, :
            ]

            # succ
            metadata.seq_lens_succ = self.decode_metadata["seq_lens_succ"][:bs]
            metadata.max_seq_len_succ = self.max_context_len

            metadata.block_tables_succ = self.decode_metadata["block_tables_succ"][
                :bs, :
            ]

            metadata.seq_lens_inter = self.decode_metadata["seq_lens_inter"][:bs]
            metadata.max_seq_len_inter = self.max_context_len

            self.decode_metadata[bs] = metadata

        self.forward_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[None],
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: torch.Tensor = None,
    ):
        """Initialize forward metadata for replaying CUDA graph."""
        assert forward_mode.is_decode()
        seq_lens = seq_lens[:bs]
        req_pool_indices = req_pool_indices[:bs]
        metadata = self.decode_metadata[bs]

        metadata.seq_lens_tensor.copy_(seq_lens.to(torch.int32))
        metadata.seq_lens = seq_lens.tolist()
        metadata.max_seq_len = seq_lens.max().item()

        metadata.orig_seq_lens_tensor.copy_(seq_lens)
        metadata.orig_seq_lens = seq_lens.tolist()

        block_tables = self.req_to_token[req_pool_indices, : metadata.max_seq_len]
        # Convert the block table to a strided format.
        if self.page_size > 1:
            strided_indices = torch.arange(
                0, block_tables.shape[1], self.page_size, device=self.device
            )
            block_tables = block_tables[:, strided_indices] // self.page_size
        metadata.block_tables.fill_(0)
        metadata.block_tables[: block_tables.shape[0], : block_tables.shape[1]].copy_(
            block_tables
        )

        if self.original_max_position_embeddings > 0:
            scaling_factor = (
                0.1
                * torch.log(
                    metadata.orig_seq_lens_tensor
                    / self.original_max_position_embeddings
                )
                + 1.0
            ).clip(min=1)
            metadata.scaling_factor.copy_(scaling_factor)

        cache_seq_lens = metadata.orig_seq_lens_tensor

        chunk_len = self.chunk_size - self.local_size
        chunk_num_curr = (cache_seq_lens - 1) // chunk_len

        seq_lens_intra = cache_seq_lens - chunk_num_curr * chunk_len
        max_seq_len_intra = seq_lens_intra.max().item()
        metadata.seq_lens_intra.copy_(seq_lens_intra)
        metadata.max_seq_len_intra = max_seq_len_intra

        metadata.block_tables_intra.fill_(0)
        for i in range(bs):
            st = chunk_num_curr[i] * chunk_len // self.page_size
            ed = min(
                st + (max_seq_len_intra - 1) // self.page_size + 1,
                (cache_seq_lens[i] - 1) // self.page_size + 1,
            )
            metadata.block_tables_intra[i, : ed - st] = metadata.block_tables[i, st:ed]

        seq_lens_succ = (chunk_num_curr - (chunk_num_curr - 1).clip(min=0)) * chunk_len
        metadata.seq_lens_succ.copy_(seq_lens_succ)
        metadata.max_seq_len_succ = metadata.seq_lens_succ.max().item()
        if metadata.max_seq_len_succ:
            metadata.block_tables_succ.fill_(0)
            for i in range(bs):
                start = (
                    (chunk_num_curr[i] - 1).clip(min=0) * chunk_len // self.page_size
                )
                end = min(
                    start + (metadata.max_seq_len_succ - 1) // self.page_size + 1,
                    (cache_seq_lens[i] - 1) // self.page_size + 1,
                )
                metadata.block_tables_succ[i, : end - start] = metadata.block_tables[
                    i, start:end
                ]

        seq_lens_inter = (chunk_num_curr - 1).clip(min=0) * chunk_len
        metadata.seq_lens_inter.copy_(seq_lens_inter)
        metadata.max_seq_len_inter = metadata.seq_lens_inter.max().item()

        self.forward_metadata = metadata

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence length in CUDA graph."""
        return 1

    def _dual_chunk_flash_attn_prefill(
        self,
        q,
        q_succ,
        q_inter,
        q_succ_critical,
        q_inter_critical,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        orig_seq_lens: List[int],
        scaling_factor: torch.Tensor,
        softmax_scale: float,
        causal: Optional[bool] = True,
        window_size: Tuple[int, int] = (-1, -1),
        block_table: Optional[torch.Tensor] = None,
        chunk_size: int = 8192,
        local_size: int = 1024,
    ):
        """
        双块Flash Attention预填充函数
        
        该函数实现了双块注意力机制的预填充阶段,将长序列分成多个块进行处理。
        支持稀疏注意力模式,可以处理超长上下文。
        
        参数说明:
            q: 主查询张量 (total_tokens, num_heads, head_dim)
            q_succ: 后继块查询张量,用于关注下一个块
            q_inter: 中间块查询张量,用于关注非相邻块
            q_succ_critical: 后继块的关键token查询
            q_inter_critical: 中间块的关键token查询
            k: 键张量,可能是完整的或分块的
            v: 值张量,可能是完整的或分块的
            cu_seqlens_q: 查询序列的累积长度 (batch_size+1,)
            cu_seqlens_k: 键序列的累积长度 (batch_size+1,)
            orig_seq_lens: 每个序列的原始长度列表
            scaling_factor: 长度缩放因子,用于超长序列
            softmax_scale: softmax的缩放因子
            causal: 是否使用因果掩码(必须为True)
            window_size: 滑动窗口大小(不支持,必须为(-1,-1))
            block_table: 块表,用于分页KV缓存
            chunk_size: 块大小,默认8192
            local_size: 局部注意力窗口大小,默认1024
        
        返回:
            torch.Tensor: 注意力输出 (total_tokens, num_heads, head_dim)
        """
        # 验证参数:双块注意力必须使用因果掩码
        if not causal:
            raise ValueError("Dual Chunk Attention does not support causal=False")
        # 验证参数:不支持滑动窗口
        if window_size != (-1, -1):
            raise ValueError("Dual Chunk Attention does not support window_size")

        # 将累积序列长度转换为CPU列表,便于Python循环处理
        cu_seqlens_q_cpu = cu_seqlens_q.cpu().tolist()
        cu_seqlens_k_cpu = cu_seqlens_k.cpu().tolist()
        all_outputs = []

        # 遍历批次中的每个序列
        for i in range(0, len(cu_seqlens_q_cpu) - 1):
            # 获取当前序列的查询起始和结束位置
            # cu_seqlens_q 是累积查询序列长度数组,形状为 (batch_size+1,)
            # 例如: [0, 4, 10, 15] 表示3个序列,长度分别为4, 6, 5
            # cu_seqlens_q_cpu[i] 获取第i个序列的起始位置
            # cu_seqlens_q_cpu[i+1] 获取第i个序列的结束位置(即第i+1个序列的起始位置)
            # cu_seqlens_q_cpu[i : i + 2][-1] 等价于 cu_seqlens_q_cpu[i+1],取切片的最后一个元素
            qs = cu_seqlens_q_cpu[i]  # 第i个序列的查询起始索引
            qe = cu_seqlens_q_cpu[i + 1]  # 第i个序列的查询结束索引
            
            # cu_seqlens_k 是累积键序列长度数组,形状为 (batch_size+1,)
            # 存储方式与 cu_seqlens_q 相同,但表示的是键/值序列的累积长度
            ks = cu_seqlens_k_cpu[i]  # 第i个序列的键起始索引
            ke = cu_seqlens_k_cpu[i + 1]  # 第i个序列的键结束索引

            # 提取当前序列的所有查询张量
            current_q = q[qs:qe]
            current_q_succ = q_succ[qs:qe]
            current_q_inter = q_inter[qs:qe]
            current_q_succ_critical = q_succ_critical[qs:qe]
            current_q_inter_critical = q_inter_critical[qs:qe]

            # 根据是否使用块表(分页KV缓存)来设置键值张量
            if block_table is None:
                # 不使用分页缓存:直接切片获取当前序列的K和V
                current_k = k[ks:ke]
                current_v = v[ks:ke]
                current_block_table = None
                current_orig_seq_len = orig_seq_lens[i]
            else:
                # 使用分页缓存:K和V是完整的缓存池,通过块表索引
                current_block_table = block_table[i]
                current_orig_seq_len = orig_seq_lens[i]
                current_k = k
                current_v = v
            
            # 判断是否启用稀疏注意力:
            # 1. 全局配置启用稀疏注意力
            # 2. 当前序列长度超过稀疏注意力阈值
            sparse_attn_enabled = (
                self.sparse_attention_enabled
                and current_orig_seq_len > self.sparse_attention_threshold
            )

            # 跳过空查询序列
            if current_q.shape[0] == 0:
                continue

            # 如果键序列为空,返回零张量
            if current_k.shape[0] == 0:
                all_outputs.append(
                    torch.zeros(
                        (current_q.shape[0], current_q.shape[1], v.shape[2]),
                        device=q.device,
                        dtype=q.dtype,
                    )
                )
                continue

            # 创建输出张量
            current_output = torch.empty_like(current_q)
            # 计算分组大小(用于分组查询注意力GQA)
            group_size = int(current_q.size(-2) / current_k.size(-2))

            if sparse_attn_enabled:
                # 稀疏注意力模式:所有头一起处理
                num_device_q_heads = current_q.size(-2)
                # 为每个注意力头准备稀疏模式参数
                heads_vertical_size = torch.empty(
                    size=(num_device_q_heads,), dtype=torch.int32
                )
                heads_slash_size = torch.empty(
                    size=(num_device_q_heads,), dtype=torch.int32
                )
                
                # 从配置中读取每个头的稀疏注意力参数
                for head_id in range(current_q.size(-2)):
                    (
                        ty,
                        vertical_size,
                        slash_size,
                        _,
                    ) = self.layer_sparse_attention_config[head_id]
                    # 验证稀疏模式类型
                    assert ty == "vertical_and_slash", "only support slash mode"

                    # 特殊处理:如果vertical_size为30,增加到130
                    if vertical_size == 30:
                        vertical_size += 100
                    heads_vertical_size[head_id] = vertical_size
                    heads_slash_size[head_id] = slash_size

                # 调用稀疏注意力实现,一次处理所有头
                current_output = self._dual_chunk_flash_attn_prefill_func(
                    current_q,  # 所有头的查询
                    current_q_succ,
                    current_q_inter,
                    current_q_succ_critical,
                    current_q_inter_critical,
                    current_k,
                    current_v,
                    current_block_table,
                    softmax_scale,
                    chunk_size,
                    local_size,
                    scaling_factor[i].item(),
                    ke - ks,
                    sparse_attn_enabled=sparse_attn_enabled,
                    heads_vertical_size=heads_vertical_size,
                    heads_slash_size=heads_slash_size,
                    group_size=group_size,
                )
            else:
                # 密集注意力模式:逐头处理
                for head_id in range(current_q.size(-2)):
                    # 提取单个头的查询张量
                    # current_q 形状: (seq_len, num_heads, head_size)
                    # [:, head_id, :] 选择所有序列位置的第 head_id 个头的所有特征
                    # 结果形状: (seq_len, head_size)
                    # unsqueeze(1) 在第1维插入维度,恢复头维度
                    # 最终形状: (seq_len, 1, head_size)
                    current_q_head = current_q[:, head_id, :].unsqueeze(1)
                    current_q_succ_head = current_q_succ[:, head_id, :].unsqueeze(1)
                    current_q_inter_head = current_q_inter[:, head_id, :].unsqueeze(1)
                    # 对于 critical 查询张量,同样的索引操作
                    # [:, head_id, :] 提取第 head_id 个头
                    # unsqueeze(1) 恢复头维度以保持张量形状一致性
                    current_q_succ_head_critical = current_q_succ_critical[
                        :, head_id, :
                    ].unsqueeze(1)
                    current_q_inter_head_critical = current_q_inter_critical[
                        :, head_id, :
                    ].unsqueeze(1)
                    
                    # 根据是否使用块表提取对应的K和V
                    if block_table is not None:
                        # 使用分组查询注意力(GQA):多个Q头共享一个KV头
                        # current_k 形状: (num_blocks, block_size, num_kv_heads, head_size)
                        # head_id // group_size 计算当前Q头对应的KV头索引
                        # 例如: group_size=4时, Q头0-3都映射到KV头0
                        # [..., head_id // group_size, :] 选择:
                        #   - ... : 保留前面所有维度(num_blocks, block_size)
                        #   - head_id // group_size : 选择对应的KV头
                        #   - : : 保留所有head_size特征
                        # 结果形状: (num_blocks, block_size, head_size)
                        # unsqueeze(2) 在第2维插入维度,恢复头维度
                        # 最终形状: (num_blocks, block_size, 1, head_size)
                        current_k_head = current_k[
                            ..., head_id // group_size, :
                        ].unsqueeze(2)
                        current_v_head = current_v[
                            ..., head_id // group_size, :
                        ].unsqueeze(2)
                    else:
                        # 标准多头注意力(MHA):每个Q头有独立的KV头
                        # current_k 形状: (seq_len, num_heads, head_size)
                        # [:, head_id, :] 选择第 head_id 个头
                        # 结果形状: (seq_len, head_size)
                        # unsqueeze(1) 恢复头维度
                        # 最终形状: (seq_len, 1, head_size)
                        current_k_head = current_k[:, head_id, :].unsqueeze(1)
                        current_v_head = current_v[:, head_id, :].unsqueeze(1)

                    # 调用单头注意力计算
                    # 所有输入张量的头维度都是1,表示单头处理
                    current_out = self._dual_chunk_flash_attn_prefill_func(
                        current_q_head,
                        current_q_succ_head,
                        current_q_inter_head,
                        current_q_succ_head_critical,
                        current_q_inter_head_critical,
                        current_k_head,
                        current_v_head,
                        current_block_table,
                        softmax_scale,
                        chunk_size,
                        local_size,
                        scaling_factor[i].item(),
                        ke - ks,
                        sparse_attn_enabled=sparse_attn_enabled,
                    )
                    # 将单头结果写入输出张量
                    # current_output 形状: (seq_len, num_heads, head_size)
                    # [:, head_id : head_id + 1, :] 选择第 head_id 个头的位置
                    # 使用切片 [head_id : head_id + 1] 而不是索引 [head_id]
                    # 是为了保持维度: (seq_len, 1, head_size)
                    # current_out 形状: (seq_len, 1, head_size)
                    # 赋值操作将单头结果填充到对应位置
                    current_output[:, head_id : head_id + 1, :] = current_out
            
            # 收集当前序列的输出
            all_outputs.append(current_output)
        
        # 将批次中所有序列的输出拼接起来
        return torch.cat(all_outputs, dim=0)

    def _dual_chunk_flash_attn_prefill_func(
        self,
        q,
        q_succ,
        q_inter,
        q_succ_critical,
        q_inter_critical,
        k,
        v,
        block_table,
        softmax_scale: float,
        chunk_size: int,
        local_size: int,
        scaling_factor: float,
        k_length: int,
        sparse_attn_enabled: Optional[bool] = True,
        heads_vertical_size=None,
        heads_slash_size=None,
        group_size=None,
    ):
        """
        双块Flash Attention预填充函数
        
        核心思想:
        1. 将长序列分成多个chunk进行处理,每个chunk包含:
           - intra: 当前chunk内的注意力(因果掩码)
           - succ: 前一个chunk的注意力(非因果)
           - inter: 更早chunk的注意力(非因果)
        
        2. 稀疏注意力优化:
           - vertical: 选择重要的列(token位置)
           - slash: 选择重要的对角线(相对位置模式)
        
        参数说明:
        - q: 查询张量 (seq_len, num_heads, head_dim)
        - q_succ/q_inter: 用于succ/inter chunk的查询
        - q_succ_critical/q_inter_critical: 用于稀疏注意力选择的关键查询
        - k, v: 键值张量
        - block_table: 分页注意力的块表
        - chunk_size: 块大小
        - local_size: 局部窗口大小
        - scaling_factor: RoPE缩放因子
        - sparse_attn_enabled: 是否启用稀疏注意力
        - heads_vertical_size/heads_slash_size: 每个头的稀疏模式大小
        - group_size: GQA的组大小
        """
        flash_results = []
        chunk_len = chunk_size - local_size  # 实际chunk长度(去除local窗口)

        # 验证块大小与chunk长度的兼容性
        if block_table is not None:
            block_size = v.shape[1]
            if chunk_len % block_size != 0:
                raise ValueError("chunk_len must be divisible by block_size.")
        else:
            block_size = 1

        # 应用RoPE缩放因子
        if self.original_max_position_embeddings > 0:
            softmax_scale = softmax_scale * scaling_factor

        # === 初始化处理起点 ===
        # 
        # begin: 当前要处理的chunk在KV缓存中的起始位置
        # 
        # 计算逻辑:
        # - k_length: KV缓存的总长度(包含所有历史token)
        # - q.shape[0]: 当前需要计算注意力的查询token数量
        # - begin = k_length - q.shape[0]: 从KV缓存中需要开始计算注意力的位置
        # 
        # 场景说明:
        # 1. Prefill场景: 
        #    - 假设序列总长度为12000,当前prefill最后2000个token
        #    - k_length = 12000 (完整KV缓存)
        #    - q.shape[0] = 2000 (只需要为这2000个token计算注意力)
        #    - begin = 12000 - 2000 = 10000
        #    - 含义: 从KV缓存的第10000个位置开始,为后续2000个token计算注意力
        # 
        # 2. 为什么从这里开始:
        #    - 前10000个token的注意力输出已经在之前的prefill中计算过
        #    - 只需要为新增的2000个token(位置10000-11999)计算注意力输出
        #    - 这些新token需要attend到所有历史token(0-11999)
        # 
        # 3. 循环处理:
        #    - 从begin开始,每次处理一个chunk,直到覆盖所有需要计算的token
        #    - 每个chunk会计算该chunk内token的注意力输出
        begin = k_length - q.shape[0]
        while begin < k_length:
            flash_per_chunk = []  # 存储当前chunk的所有注意力结果

            # === 计算当前chunk的边界位置 ===
            # 
            # 目标: 确定当前要处理的chunk在整个序列中的位置范围
            # 
            # 1. prev_chunk_end_pos: 当前chunk的起始位置(对齐到chunk边界)
            #    - begin // chunk_len: 计算begin属于第几个chunk(整数除法)
            #    - 乘以chunk_len: 将chunk编号转换回序列位置
            #    - 例如: begin=10000, chunk_len=8192-1024=7168
            #      -> 10000 // 7168 = 1 (第1个chunk)
            #      -> 1 * 7168 = 7168 (该chunk从位置7168开始)
            prev_chunk_end_pos = (begin // chunk_len) * chunk_len
            
            # 2. next_chunk_end_pos: 下一个chunk的起始位置(即当前chunk的理论结束位置)
            #    - 例如: prev_chunk_end_pos=7168, chunk_len=7168
            #      -> next_chunk_end_pos = 7168 + 7168 = 14336
            next_chunk_end_pos = prev_chunk_end_pos + chunk_len
            
            # 3. end: 当前chunk的实际结束位置(不能超过序列总长度)
            #    - 取理论结束位置和序列总长度的较小值
            #    - 例如: next_chunk_end_pos=14336, k_length=12000
            #      -> end = min(14336, 12000) = 12000 (最后一个chunk可能不完整)
            end = min(next_chunk_end_pos, k_length)
            
            # 4. qbegin: 当前chunk在查询张量q中的起始索引
            #    - k_length - q.shape[0]: KV缓存中不需要计算注意力的前缀长度
            #    - begin - (k_length - q.shape[0]): 将KV空间的位置映射到Q空间
            #    - 例如: begin=10000, k_length=12000, q.shape[0]=2000
            #      -> qbegin = 10000 - (12000 - 2000) = 0
            #    - 解释: Q只包含最后2000个token,所以begin=10000对应q的索引0
            qbegin = begin - (k_length - q.shape[0])
            
            # 5. qend: 当前chunk在查询张量q中的结束索引
            #    - 同样的映射逻辑,将end从KV空间映射到Q空间
            #    - 例如: end=12000, k_length=12000, q.shape[0]=2000
            #      -> qend = 12000 - (12000 - 2000) = 2000
            qend = end - (k_length - q.shape[0])
            
            # 总结:
            # - KV空间范围: [prev_chunk_end_pos, end) 表示当前chunk在完整序列中的位置
            # - Q空间范围: [qbegin, qend) 表示当前chunk对应的查询token在q张量中的索引
            # - 这种映射允许我们处理prefill场景,其中q可能只包含新增的token,
            #   而KV缓存包含完整的历史token

            # === 第一部分: Intra-chunk注意力(当前chunk内) ===
            qk_chunks = []  # 用于稀疏注意力选择的QK分数
            q_states_intra = q[qbegin:qend]  # 提取当前chunk对应的查询状态
            
            # 获取当前chunk的KV状态
            if block_table is not None:
                # 分页注意力:通过block_table索引
                block_tables_intra = _get_block(
                    block_table, block_size, prev_chunk_end_pos, end
                )
                k_states_intra = k[block_tables_intra].view(-1, *k.shape[-2:])[
                    : (end - prev_chunk_end_pos)
                ]
                v_states_intra = v[block_tables_intra].view(-1, *v.shape[-2:])[
                    : (end - prev_chunk_end_pos)
                ]
            else:
                # 连续内存:直接切片
                block_tables_intra = None
                k_states_intra = k[prev_chunk_end_pos:end]
                v_states_intra = v[prev_chunk_end_pos:end]

            # 稀疏注意力:计算QK分数用于选择重要token
            if sparse_attn_enabled:
                # 1. 确定用于稀疏注意力选择的查询数量
                #    - self.sparse_attention_last_q: 配置的最后N个查询token数量(默认64)
                #    - qend - qbegin: 当前chunk中实际的查询token数量
                #    - 取较小值,避免超出实际查询范围
                #    - 例如: 如果chunk只有32个新token,则last_q_size=32而不是64
                last_q_size = min(qend - qbegin, self.sparse_attention_last_q)
                
                # 2. 获取KV状态的维度信息
                #    - k_states_intra.shape = (seq_len, num_kv_heads, head_dim)
                #    - num_device_k_heads: 当前设备上的KV头数量(可能因张量并行而小于总头数)
                #    - head_dim: 每个注意力头的维度(通常是128或64)
                _, num_device_k_heads, head_dim = k_states_intra.shape
                
                # 3. GQA(Grouped Query Attention)扩展 - K状态
                #    目的: 将较少的KV头扩展到匹配较多的Q头
                #    
                #    步骤分解:
                #    a) unsqueeze(2): 在第2维插入新维度
                #       形状变化: (seq_len, num_kv_heads, head_dim) 
                #                -> (seq_len, num_kv_heads, 1, head_dim)
                #    
                #    b) repeat(1, 1, group_size, 1): 在新维度上重复group_size次
                #       - group_size = num_q_heads / num_kv_heads (每个KV头对应的Q头数量)
                #       - 例如: 如果有32个Q头和8个KV头,则group_size=4
                #       形状变化: (seq_len, num_kv_heads, 1, head_dim)
                #                -> (seq_len, num_kv_heads, group_size, head_dim)
                #    
                #    c) reshape: 合并KV头和组维度
                #       形状变化: (seq_len, num_kv_heads, group_size, head_dim)
                #                -> (seq_len, num_kv_heads * group_size, head_dim)
                #       - 最终num_kv_heads * group_size = num_q_heads
                #       - 这样每个Q头都有对应的K头用于计算注意力
                k_states_intra = (
                    k_states_intra.unsqueeze(2)
                    .repeat(1, 1, group_size, 1)
                    .reshape(-1, num_device_k_heads * group_size, head_dim)
                )
                
                # 4. GQA扩展 - V状态
                #    与K状态完全相同的扩展逻辑
                #    确保V头数量也匹配Q头数量
                v_states_intra = (
                    v_states_intra.unsqueeze(2)
                    .repeat(1, 1, group_size, 1)
                    .reshape(-1, num_device_k_heads * group_size, head_dim)
                )
                
                # 5. 计算QK注意力分数用于稀疏token选择
                #    
                #    步骤分解:
                #    a) q_states_intra.transpose(0, 1): 
                #       - 原始形状: (seq_len_q, num_heads, head_dim)
                #       - 转置后: (num_heads, seq_len_q, head_dim)
                #       - 目的: 将头维度移到最前面,便于批量矩阵乘法
                #    
                #    b) [:, -last_q_size:]: 
                #       - 只取最后last_q_size个查询token
                #       - 形状: (num_heads, last_q_size, head_dim)
                #       - 原因: 稀疏注意力只需要最近的查询来选择重要的KV token
                #    
                #    c) * softmax_scale: 
                #       - 缩放因子,通常是 1/sqrt(head_dim)
                #       - 防止softmax前的logits过大导致梯度消失
                #    
                #    d) @ k_states_intra.permute(1, 2, 0):
                #       - k_states_intra原始: (seq_len_k, num_heads, head_dim)
                #       - permute后: (num_heads, head_dim, seq_len_k)
                #       - 矩阵乘法: (num_heads, last_q_size, head_dim) @ (num_heads, head_dim, seq_len_k)
                #       - 结果形状: (num_heads, last_q_size, seq_len_k)
                #       - 含义: 每个头的每个查询token对所有键token的注意力分数
                #    
                #    e) qk_chunks.append(...):
                #       - 将当前chunk的QK分数添加到列表
                #       - 后续会用这些分数来选择最重要的token进行稀疏注意力计算
                qk_chunks.append(
                    (q_states_intra.transpose(0, 1)[:, -last_q_size:] * softmax_scale)
                    @ k_states_intra.permute(1, 2, 0)
                )

            # === 第二部分: Succ-chunk注意力(前一个chunk) ===
            if prev_chunk_end_pos - chunk_len >= 0:
                q_states_succ = q_succ[qbegin:qend]
                q_states_succ_critical = q_succ_critical[qbegin:qend]
                
                # 获取前一个chunk的KV状态
                if block_table is not None:
                    block_tables_succ = _get_block(
                        block_table,
                        block_size,
                        prev_chunk_end_pos - chunk_len,
                        prev_chunk_end_pos,
                    )
                    k_states_succ = k[block_tables_succ].view(-1, *k.shape[-2:])[
                        :chunk_len
                    ]
                    v_states_succ = v[block_tables_succ].view(-1, *v.shape[-2:])[
                        :chunk_len
                    ]
                else:
                    # 获取前一个chunk的KV状态
                    k_states_succ = k[
                        prev_chunk_end_pos - chunk_len : prev_chunk_end_pos
                    ]
                    v_states_succ = v[
                        prev_chunk_end_pos - chunk_len : prev_chunk_end_pos
                    ]

                if sparse_attn_enabled:
                    # GQA扩展
                    k_states_succ = (
                        k_states_succ.unsqueeze(2)
                        .repeat(1, 1, group_size, 1)
                        .reshape(-1, num_device_k_heads * group_size, head_dim)
                    )
                    v_states_succ = (
                        v_states_succ.unsqueeze(2)
                        .repeat(1, 1, group_size, 1)
                        .reshape(-1, num_device_k_heads * group_size, head_dim)
                    )
                    # 使用critical查询计算注意力分数
                    qk_chunks.append(
                        (
                            q_states_succ_critical.transpose(0, 1)[:, -last_q_size:]
                            * softmax_scale
                        )
                        @ k_states_succ.permute(1, 2, 0)
                    )

            # === 第三部分: Inter-chunk注意力(更早的chunks) ===
            if prev_chunk_end_pos - chunk_len * 2 >= 0:
                q_states_inter = q_inter[qbegin:qend]
                q_states_inter_critical = q_inter_critical[qbegin:qend]
                
                # 获取所有更早chunk的KV状态
                if block_table is not None:
                    block_tables_inter = _get_block(
                        block_table, block_size, 0, prev_chunk_end_pos - chunk_len
                    )
                    k_states_inter = k[block_tables_inter].view(-1, *k.shape[-2:])[
                        : (prev_chunk_end_pos - chunk_len)
                    ]
                    v_states_inter = v[block_tables_inter].view(-1, *v.shape[-2:])[
                        : (prev_chunk_end_pos - chunk_len)
                    ]
                else:
                    # 获取所有更早chunk的KV状态
                    k_states_inter = k[: prev_chunk_end_pos - chunk_len]
                    v_states_inter = v[: prev_chunk_end_pos - chunk_len]

                if sparse_attn_enabled:
                    # GQA扩展
                    k_states_inter = (
                        k_states_inter.unsqueeze(2)
                        .repeat(1, 1, group_size, 1)
                        .reshape(-1, num_device_k_heads * group_size, head_dim)
                    )
                    v_states_inter = (
                        v_states_inter.unsqueeze(2)
                        .repeat(1, 1, group_size, 1)
                        .reshape(-1, num_device_k_heads * group_size, head_dim)
                    )
                    # 使用critical查询计算注意力分数
                    qk_chunks.append(
                        (
                            q_states_inter_critical.transpose(0, 1)[:, -last_q_size:]
                            * softmax_scale
                        )
                        @ k_states_inter.permute(1, 2, 0)
                    )

            # === 第四部分: 稀疏注意力模式选择 ===
            if sparse_attn_enabled:
                # 合并所有chunk的QK分数(逆序:inter, succ, intra)
                reversed_qk = qk_chunks[::-1]
                qk = torch.cat(reversed_qk, dim=-1)

                # 应用因果掩码到最后的查询
                qk[:, :, -last_q_size:] = torch.where(
                    self.last_q_mask[..., -last_q_size:, -last_q_size:].to(qk.device),
                    qk[:, :, -last_q_size:],
                    -torch.inf,
                )
                qk = F.softmax(qk, dim=-1, dtype=torch.float32)

                # Vertical模式:列求和,找重要的token位置
                vertical = qk.sum(-2, keepdim=True)
                vertical[..., :30] = torch.inf  # 忽略前30个token

                # 准备缓冲区用于存储每个头的稀疏索引
                int32_max = torch.iinfo(torch.int32).max
                int32_min = torch.iinfo(torch.int32).min
                n_heads = qk.size()[0]
                max_slash_topk = torch.max(heads_slash_size).item()
                max_vertical_topk = torch.max(heads_vertical_size).item()
                
                # Vertical topk选择
                vertical = vertical.reshape((n_heads, -1))
                max_vertical_topk = min(vertical.shape[-1], max_vertical_topk)
                vertical_topk_buffer = torch.topk(
                    vertical, max_vertical_topk, -1
                ).indices
                
                # Slash topk选择(对角线模式)
                slash_topk_buffer = torch.empty(
                    size=(n_heads, max_slash_topk), dtype=torch.int64, device=qk.device
                )
                for head_i in range(n_heads):
                    head_score = qk[head_i : head_i + 1, :, :]
                    # 对角线求和:捕获相对位置模式
                    slash_scores = _sum_all_diagonal_matrix(head_score)
                    if head_score.size(1) != 1:
                        slash_scores = slash_scores[..., : -last_q_size + 1]
                    slash_scores[..., -100:] = torch.inf  # 忽略最近的100个位置

                    # 选择topk对角线
                    head_slash_size = heads_slash_size[head_i]
                    head_slash_size = min(head_slash_size, vertical.size(-1))
                    slash_topk = torch.topk(slash_scores, head_slash_size, -1).indices
                    slash_topk_buffer[head_i, :head_slash_size] = slash_topk

                    heads_slash_size[head_i] = head_slash_size
                    heads_vertical_size[head_i] = min(
                        heads_vertical_size[head_i], max_vertical_topk
                    )

                # 为每个chunk创建独立的稀疏索引缓冲区
                # Intra chunk
                vertical_buffer = torch.full(
                    (n_heads, max_vertical_topk),
                    int32_max,
                    dtype=torch.int64,
                    device=q.device,
                )
                slash_buffer = torch.full(
                    (n_heads, max_slash_topk),
                    int32_min,
                    dtype=torch.int64,
                    device=q.device,
                )
                # Succ chunk
                succ_vertical_buffer = torch.full(
                    (n_heads, max_vertical_topk),
                    int32_max,
                    dtype=torch.int64,
                    device=q.device,
                )
                succ_slash_buffer = torch.full(
                    (n_heads, max_slash_topk),
                    int32_min,
                    dtype=torch.int64,
                    device=q.device,
                )
                # Inter chunk
                inter_vertical_buffer = torch.full(
                    (n_heads, max_vertical_topk),
                    int32_max,
                    dtype=torch.int64,
                    device=q.device,
                )
                inter_slash_buffer = torch.full(
                    (n_heads, max_slash_topk),
                    int32_min,
                    dtype=torch.int64,
                    device=q.device,
                )

                # 大小缓冲区
                vertical_size_buffer = torch.empty(
                    size=(n_heads,), dtype=torch.int32, device=q.device
                )
                slash_sizes_buffer = torch.empty(
                    size=(n_heads,), dtype=torch.int32, device=q.device
                )
                succ_vertical_size_buffer = torch.empty(
                    size=(n_heads,), dtype=torch.int32, device=q.device
                )
                succ_slash_sizes_buffer = torch.empty(
                    size=(n_heads,), dtype=torch.int32, device=q.device
                )
                inter_vertical_size_buffer = torch.empty(
                    size=(n_heads,), dtype=torch.int32, device=q.device
                )
                inter_slash_sizes_buffer = torch.empty(
                    size=(n_heads,), dtype=torch.int32, device=q.device
                )

                # 为每个头分配稀疏索引到对应的chunk
                for head_i in range(n_heads):
                    vertical_topk = vertical_topk_buffer[
                        head_i, : heads_vertical_size[head_i]
                    ]
                    
                    # === Intra chunk索引 ===
                    intra_vertical_indices = (
                        vertical_topk[vertical_topk >= prev_chunk_end_pos]
                        - prev_chunk_end_pos
                    )
                    # 如果没有选中的索引,使用均匀采样作为后备
                    if intra_vertical_indices.nelement() == 0:
                        intra_vertical_indices = torch.cat(
                            [
                                intra_vertical_indices,
                                torch.arange(
                                    0,
                                    k_states_intra.size(0),
                                    max(1, k_states_intra.size(0) / 5),
                                    dtype=torch.int32,
                                    device=intra_vertical_indices.device,
                                ),
                            ]
                        )
                    
                    slash_topk = slash_topk_buffer[head_i, : heads_slash_size[head_i]]
                    intra_slash_indices = (qk.size(-1) - 1) - slash_topk[
                        slash_topk >= prev_chunk_end_pos
                    ]
                    
                    # 填充缓冲区
                    v_count = intra_vertical_indices.nelement()
                    s_count = intra_slash_indices.nelement()
                    vertical_size_buffer[head_i] = v_count
                    slash_sizes_buffer[head_i] = s_count
                    vertical_buffer[head_i, :v_count].copy_(intra_vertical_indices)
                    slash_buffer[head_i, :s_count].copy_(intra_slash_indices)
                    
                    # === Succ chunk索引 ===
                    if prev_chunk_end_pos - chunk_len >= 0:
                        succ_vertical_indices = vertical_topk[
                            (vertical_topk < prev_chunk_end_pos)
                            & (vertical_topk >= prev_chunk_end_pos - chunk_len)
                        ] - (prev_chunk_end_pos - chunk_len)
                        
                        if succ_vertical_indices.nelement() == 0:
                            succ_vertical_indices = torch.cat(
                                [
                                    succ_vertical_indices,
                                    torch.arange(
                                        0,
                                        k_states_succ.size(0),
                                        max(1, k_states_succ.size(0) / 5),
                                        dtype=torch.int32,
                                        device=intra_vertical_indices.device,
                                    ),
                                ]
                            )
                        
                        succ_slash_indices = (
                            prev_chunk_end_pos + (qend - qbegin) - 1
                        ) - slash_topk[
                            (
                                (slash_topk >= (prev_chunk_end_pos - chunk_len))
                                & (slash_topk < (prev_chunk_end_pos + (qend - qbegin)))
                            )
                        ]
                        
                        if succ_slash_indices.nelement() == 0:
                            succ_slash_indices = torch.cat(
                                [
                                    succ_slash_indices,
                                    torch.arange(
                                        0,
                                        k_states_succ.size(0),
                                        max(1, k_states_succ.size(0) / 5),
                                        dtype=torch.int32,
                                        device=intra_vertical_indices.device,
                                    ),
                                ]
                            )
                        
                        v_count = succ_vertical_indices.nelement()
                        s_count = succ_slash_indices.nelement()
                        succ_vertical_size_buffer[head_i] = v_count
                        succ_slash_sizes_buffer[head_i] = s_count
                        succ_vertical_buffer[head_i, :v_count].copy_(
                            succ_vertical_indices
                        )
                        succ_slash_buffer[head_i, :s_count].copy_(succ_slash_indices)

                    # === Inter chunk索引 ===
                    if prev_chunk_end_pos - 2 * chunk_len >= 0:
                        inter_vertical_indices = vertical_topk[
                            vertical_topk < prev_chunk_end_pos - chunk_len
                        ]

                        if inter_vertical_indices.nelement() == 0:
                            inter_vertical_indices = torch.cat(
                                [
                                    inter_vertical_indices,
                                    torch.arange(
                                        0,
                                        k_states_inter.size(0),
                                        max(1, k_states_inter.size(0) / 5),
                                        dtype=torch.int32,
                                        device=intra_vertical_indices.device,
                                    ),
                                ]
                            )
                        
                        inter_slash_indices = (
                            prev_chunk_end_pos - chunk_len + (qend - qbegin) - 1
                        ) - slash_topk[
                            slash_topk
                            < (prev_chunk_end_pos - chunk_len + (qend - qbegin))
                        ]
                        
                        if inter_slash_indices.nelement() == 0:
                            inter_slash_indices = torch.cat(
                                [
                                    inter_slash_indices,
                                    torch.arange(
                                        0,
                                        k_states_inter.size(0),
                                        max(1, k_states_inter.size(0) / 5),
                                        dtype=torch.int32,
                                        device=intra_vertical_indices.device,
                                    ),
                                ]
                            )
                        
                        v_count = inter_vertical_indices.nelement()
                        s_count = inter_slash_indices.nelement()
                        inter_vertical_size_buffer[head_i] = v_count
                        inter_slash_sizes_buffer[head_i] = s_count
                        inter_vertical_buffer[head_i, :v_count].copy_(
                            inter_vertical_indices
                        )
                        inter_slash_buffer[head_i, :s_count].copy_(inter_slash_indices)
            else:
                # 非稀疏模式:不使用索引
                intra_vertical_indices, intra_slash_indices = None, None
                succ_vertical_indices, succ_slash_indices = None, None
                inter_vertical_indices, inter_slash_indices = None, None

            # === 第五部分: 执行Flash Attention ===
            # Intra chunk (因果注意力)
            if sparse_attn_enabled:
                flash_result = self._do_flash_attn(
                    q_states_intra,
                    k_states_intra,
                    v_states_intra,
                    softmax_scale=softmax_scale,
                    causal=True,
                    stage="intra",
                    vertical_indices=vertical_buffer,
                    slash_indices=slash_buffer,
                    vertical_indices_count=vertical_size_buffer,
                    slash_indices_count=slash_sizes_buffer,
                    mergehead_softmax_scale=softmax_scale,
                    sparse_attn_enabled=sparse_attn_enabled,
                )
            else:
                flash_result = self._do_flash_attn(
                    q_states_intra,
                    k_states_intra,
                    v_states_intra,
                    softmax_scale=softmax_scale,
                    causal=True,
                    stage="intra",
                    vertical_indices=intra_vertical_indices,
                    slash_indices=intra_slash_indices,
                    sparse_attn_enabled=sparse_attn_enabled,
                )
            flash_per_chunk.append(flash_result)

            # Succ chunk (非因果注意力)
            if prev_chunk_end_pos - chunk_len >= 0:
                if sparse_attn_enabled:
                    flash_result = self._do_flash_attn(
                        q_states_succ,
                        k_states_succ,
                        v_states_succ,
                        softmax_scale=softmax_scale,
                        causal=False,
                        stage="succ",
                        vertical_indices=succ_vertical_buffer,
                        slash_indices=succ_slash_buffer,
                        vertical_indices_count=succ_vertical_size_buffer,
                        slash_indices_count=succ_slash_sizes_buffer,
                        mergehead_softmax_scale=softmax_scale,
                        sparse_attn_enabled=sparse_attn_enabled,
                    )
                else:
                    flash_result = self._do_flash_attn(
                        q_states_succ,
                        k_states_succ,
                        v_states_succ,
                        softmax_scale=softmax_scale,
                        causal=False,
                        stage="succ",
                        vertical_indices=succ_vertical_indices,
                        slash_indices=succ_slash_indices,
                        sparse_attn_enabled=sparse_attn_enabled,
                    )
                flash_per_chunk.append(flash_result)

            # Inter chunk (非因果注意力)
            if prev_chunk_end_pos - chunk_len * 2 >= 0:
                if sparse_attn_enabled:
                    flash_result = self._do_flash_attn(
                        q_states_inter,
                        k_states_inter,
                        v_states_inter,
                        softmax_scale=softmax_scale,
                        causal=False,
                        stage="inter",
                        vertical_indices=inter_vertical_buffer,
                        slash_indices=inter_slash_buffer,
                        vertical_indices_count=inter_vertical_size_buffer,
                        slash_indices_count=inter_slash_sizes_buffer,
                        mergehead_softmax_scale=softmax_scale,
                        sparse_attn_enabled=sparse_attn_enabled,
                    )
                else:
                    flash_result = self._do_flash_attn(
                        q_states_inter,
                        k_states_inter,
                        v_states_inter,
                        softmax_scale=softmax_scale,
                        causal=False,
                        stage="inter",
                        vertical_indices=inter_vertical_indices,
                        slash_indices=inter_slash_indices,
                        sparse_attn_enabled=sparse_attn_enabled,
                    )
                flash_per_chunk.append(flash_result)

            flash_results.append(flash_per_chunk)
            begin = end

        # === 第六部分: 合并所有chunk的结果 ===
        attn_output = self._merge_attn_outputs(flash_results)
        del flash_results
        return attn_output

    def _do_flash_attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        softmax_scale: float,
        causal: bool = True,
        max_seqlen_k: Optional[int] = None,
        stage: str = "intra",
        vertical_indices: Optional[torch.Tensor] = None,
        slash_indices: Optional[torch.Tensor] = None,
        vertical_indices_count: Optional[torch.Tensor] = None,
        slash_indices_count: Optional[torch.Tensor] = None,
        mergehead_softmax_scale: Optional[float] = None,
        sparse_attn_enabled: Optional[bool] = False,
    ):
        if max_seqlen_k is None:
            max_seqlen_k = key_states.shape[0]

        q_len = query_states.shape[0]
        q_heads = query_states.shape[1]
        h_dim = query_states.shape[-1]

        if sparse_attn_enabled:
            assert slash_indices is not None
            if stage == "intra":
                assert causal
            else:
                assert not causal

            query_states = query_states.unsqueeze(0).transpose(1, 2)
            key_states = key_states.unsqueeze(0).transpose(1, 2)
            value_states = value_states.unsqueeze(0).transpose(1, 2)

            q = query_states
            k = key_states
            v = value_states

            if vertical_indices_count is not None and slash_indices_count is not None:
                assert mergehead_softmax_scale is not None

                res, s_lse = _vertical_slash_sparse_attention(
                    q,
                    k,
                    v,
                    vertical_indices,
                    slash_indices,
                    mergehead_softmax_scale,
                    causal=causal,
                    stage=stage,
                    vertical_indices_count=vertical_indices_count,
                    slash_indices_count=slash_indices_count,
                )
                res = res.view(q_heads, q_len, h_dim).transpose(
                    0, 1
                )  # (qlen,nhead,h_dim)
                s_lse = (
                    s_lse.view(q_heads, q_len, 1).squeeze(-1).unsqueeze(0).float()
                )  # (1, nhead,qlen)
            else:
                res, s_lse = _vertical_slash_sparse_attention(
                    q,
                    k,
                    v,
                    vertical_indices,
                    slash_indices,
                    softmax_scale,
                    causal=causal,
                    stage=stage,
                )
                res = res.view(q_len, q_heads, h_dim)
                s_lse = s_lse.view(q_len, q_heads, 1).transpose(0, 2).float()
            return res, s_lse

        output, softmax_lse, *rest = flash_attn_varlen_func(
            q=query_states,
            k=key_states,
            v=value_states,
            softmax_scale=softmax_scale,
            cu_seqlens_q=torch.tensor(
                [0, query_states.shape[0]],
                dtype=torch.int32,
                device=query_states.device,
            ),
            max_seqlen_q=query_states.shape[0],
            cu_seqlens_k=torch.tensor(
                [0, max_seqlen_k], dtype=torch.int32, device=query_states.device
            ),
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            return_softmax_lse=True,
        )
        softmax_lse = softmax_lse.view(q_len, q_heads, 1).transpose(0, 2).float()
        return output, softmax_lse

    def _merge_attn_outputs(
        self,
        flash_results: List[List[Tuple[torch.Tensor, torch.Tensor]]],
        return_lse: Optional[bool] = False,
    ) -> torch.Tensor:
        attn_outputs_all = []
        logits_all = []

        for flash_per_chunk in flash_results:
            if len(flash_per_chunk) == 1:
                attn_outputs_all.append(flash_per_chunk[0][0])
                if return_lse:
                    logits_all.append(flash_per_chunk[0][1])
                continue

            attn_outputs = torch.stack(
                [flash_attn_output[0] for flash_attn_output in flash_per_chunk]
            )
            logits = torch.stack(
                [flash_attn_output[1] for flash_attn_output in flash_per_chunk]
            )
            logits = logits.to(torch.float32)

            if return_lse:
                max_val = torch.max(logits, dim=0).values
                diff = torch.abs(logits[0] - logits[1])
                log_sum_exp = max_val + torch.log1p(torch.exp(-diff))
                logits_all.append(log_sum_exp)

            max_logits = torch.max(logits, dim=0).values
            stable_logits = logits - max_logits.unsqueeze(0)
            lse_s = torch.exp(stable_logits).detach()
            lse_sum = torch.sum(lse_s, dim=0)
            lse_s /= lse_sum
            attn_outputs *= lse_s.unsqueeze(-1).transpose(2, 3).squeeze(1)
            attn_outputs_all.append(attn_outputs.sum(dim=0))

        if return_lse:
            return (torch.cat(attn_outputs_all, dim=0), torch.cat(logits_all, dim=-1))
        else:
            return torch.cat(attn_outputs_all, dim=0)

    def _dual_chunk_flash_attn_decoding(
        self,
        query: torch.Tensor,
        query_succ: torch.Tensor,
        query_inter: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        softmax_scale: float,
        causal: bool,
        chunk_size: int,
        local_size: int,
        original_max_position_embeddings: int,
        decode_meta: DualChunkFlashAttentionMetadata,
    ):
        if not causal:
            raise ValueError("Dual Chunk Attention does not support causal=False")

        block_size = value_cache.shape[1]
        chunk_len = chunk_size - local_size
        if chunk_len % block_size != 0:
            raise ValueError("chunk_len must be divisible by block_size.")
        if original_max_position_embeddings > 0:
            assert decode_meta.scaling_factor is not None
            scaling_factor = decode_meta.scaling_factor
            query = (query * scaling_factor.view(-1, 1, 1, 1)).to(
                query.dtype
            )  # possible for numerical issue, need to fused in the kernel
            query_succ = (query_succ * scaling_factor.view(-1, 1, 1, 1)).to(query.dtype)
            query_inter = (query_inter * scaling_factor.view(-1, 1, 1, 1)).to(
                query.dtype
            )
        outputs_list = []
        softmax_lses_list = []

        # intra-attention
        intra_output, intra_softmax_lse = (
            self._dual_chunk_flash_attn_decoding_with_exp_sums(
                query,
                key_cache,
                value_cache,
                decode_meta.block_tables_intra,
                decode_meta.seq_lens_intra,
                softmax_scale,
                causal=False,
            )
        )
        outputs_list.append(intra_output)
        softmax_lses_list.append(intra_softmax_lse)

        # succ-attention
        if decode_meta.max_seq_len_succ:
            succ_output, succ_softmax_lse = (
                self._dual_chunk_flash_attn_decoding_with_exp_sums(
                    query_succ,
                    key_cache,
                    value_cache,
                    decode_meta.block_tables_succ,
                    decode_meta.seq_lens_succ,
                    softmax_scale,
                    causal=False,
                )
            )
            outputs_list.append(succ_output)
            softmax_lses_list.append(succ_softmax_lse)

        # inter-attention
        if decode_meta.max_seq_len_inter:
            inter_output, inter_softmax_lse = (
                self._dual_chunk_flash_attn_decoding_with_exp_sums(
                    query_inter,
                    key_cache,
                    value_cache,
                    block_table,
                    decode_meta.seq_lens_inter,
                    softmax_scale,
                    causal=False,
                )
            )
            outputs_list.append(inter_output)
            softmax_lses_list.append(inter_softmax_lse)
        outputs = torch.stack(outputs_list, dim=0)
        del outputs_list
        softmax_lses = torch.stack(softmax_lses_list, dim=0).to(torch.float32)
        del softmax_lses_list
        max_logits = torch.max(softmax_lses, dim=0).values
        stable_logits = softmax_lses - max_logits.unsqueeze(0)
        lse_s = torch.exp(stable_logits).detach()
        lse_sum = torch.sum(lse_s, dim=0)
        lse_s /= lse_sum
        outputs *= lse_s.unsqueeze(-1).transpose(2, 3)
        return outputs.sum(0)

    def _dual_chunk_flash_attn_decoding_with_exp_sums(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        softmax_scale: float,
        causal: bool,
    ):
        out, softmax_lse, *rest_expand = flash_attn_with_kvcache(
            q=query,
            k_cache=key_cache,
            v_cache=value_cache,
            page_table=block_table,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            causal=causal,
            return_softmax_lse=True,
        )
        mask = cache_seqlens == 0
        out[mask] = 0
        softmax_lse[mask] = -float("inf")
        return out, softmax_lse


def _vertical_slash_sparse_attention(
    query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,  # [BATCH, N_HEADS, N_KV_CTX, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_HEADS, N_KV_CTX, D_HEAD]
    v_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
    s_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
    softmax_scale: float,
    causal: bool = True,
    stage: str = "intra",
    block_size_M: int = 64,
    block_size_N: int = 64,
    vertical_indices_count: torch.Tensor = None,  # [N_HEADS,]
    slash_indices_count: torch.Tensor = None,
):
    if stage == "intra":
        assert causal
    else:
        assert not causal

    batch_size, num_heads, context_size, head_dim = query.shape
    _, _, kv_seq_len, _ = key.shape

    if head_dim not in [16, 32, 64, 128, 256, 512]:
        target_dim = 2 ** math.ceil(math.log2(head_dim)) - head_dim
        query = F.pad(query, [0, target_dim, 0, 0, 0, 0, 0, 0])
        key = F.pad(key, [0, target_dim, 0, 0, 0, 0, 0, 0])
        value = F.pad(value, [0, target_dim, 0, 0, 0, 0, 0, 0])

    v_idx = (
        v_idx.to(torch.int32)
        .reshape((batch_size, num_heads, -1))
        .sort(dim=-1, descending=False)[0]
    )
    s_idx = (
        s_idx.to(torch.int32)
        .reshape((batch_size, num_heads, -1))
        .sort(dim=-1, descending=True)[0]
    )
    q_seqlens = torch.tensor([context_size], dtype=torch.int32, device=query.device)
    kv_seqlens = torch.tensor([kv_seq_len], dtype=torch.int32, device=query.device)

    if vertical_indices_count is not None and slash_indices_count is not None:
        (
            block_count,
            block_offset,
            column_count,
            column_index,
        ) = convert_vertical_slash_indexes_mergehead(
            q_seqlens,
            kv_seqlens,
            v_idx,
            s_idx,
            vertical_indices_count,
            slash_indices_count,
            context_size,
            block_size_M,
            block_size_N,
            causal,
        )
    else:
        (
            block_count,
            block_offset,
            column_count,
            column_index,
        ) = convert_vertical_slash_indexes(
            q_seqlens,
            kv_seqlens,
            v_idx,
            s_idx,
            context_size,
            block_size_M,
            block_size_N,
            causal,
        )

    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()
    out, lse = sparse_attn_func(
        q,
        k,
        v,
        block_count,
        block_offset,
        column_count,
        column_index,
        causal=causal,
        softmax_scale=softmax_scale,
        return_softmax_lse=True,
    )
    out = out.transpose(1, 2).contiguous()
    softmax_lse = lse.reshape(*lse.shape, 1)
    return (out[..., :context_size, :head_dim], softmax_lse[..., :context_size, :])


def _sum_all_diagonal_matrix(mat: torch.tensor):
    h, n, m = mat.shape
    # Zero matrix used for padding
    zero_mat = torch.zeros((h, n, n), device=mat.device)
    # pads the matrix on left and right
    mat_padded = torch.cat((zero_mat, mat, zero_mat), -1)
    # Change the strides
    mat_strided = mat_padded.as_strided(
        (1, n, n + m), (n * (2 * n + m), 2 * n + m + 1, 1)
    )
    # Sums the resulting matrix's columns
    sum_diags = torch.sum(mat_strided, 1)
    return sum_diags[:, 1:]  # drop left bottom corner


def _get_block(block_table: torch.Tensor, block_size: int, begin: int, end: int):
    begin_block = begin // block_size
    end_block = (end - 1) // block_size + 1
    return block_table[begin_block:end_block]
