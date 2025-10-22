
import math
from types import SimpleNamespace
from typing import Type, Callable, Optional, Tuple
from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
import cutlass.utils.hopper_helpers as sm90_utils_basic

from python.sglang.srt.sparse_attention.kernels.attention.flash_fwd import FlashAttentionForwardBase
from sglang.srt.sparse_attention.kernels.attention import hopper_helpers as sm90_utils
from sglang.srt.sparse_attention.kernels.attention import utils
from sglang.srt.sparse_attention.kernels.attention import pipeline
from sglang.srt.sparse_attention.kernels.attention.mask import AttentionMask
from sglang.srt.sparse_attention.kernels.attention.softmax import Softmax
from sglang.srt.sparse_attention.kernels.attention.seqlen_info import SeqlenInfoQK
from sglang.srt.sparse_attention.kernels.attention.block_info import BlockInfo
from sglang.srt.sparse_attention.kernels.attention.pack_gqa import PackGQA
from sglang.srt.sparse_attention.kernels.attention.named_barrier import NamedBarrierFwd
from sglang.srt.sparse_attention.kernels.attention.tile_scheduler import (
    TileSchedulerArguments, 
    SingleTileScheduler, 
    SingleTileLPTScheduler, 
    SingleTileVarlenScheduler, 
    ParamsBase,
    BlockSparseTileScheduler,
    BlockSparseMaskIterator,
)

class FlashBlockSparseFwdSm90(FlashAttentionForwardBase):
    
    arch = 90
    
    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        mbar_ptr_Q: cutlass.Pointer,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        """
        块稀疏版本的 load 函数
        
        与标准 load 函数的关键区别：
        1. 每个 CUDA block 被分配一个固定的 Q 块行 (m_block, head, batch)
        2. 使用掩码迭代器只加载该行的有效 K 块
        3. 不存在持久化调度循环（一个 CUDA block = 一个 Q 块行）
        
        工作流程：
        - BlockSparseTileScheduler.create() 分配 Q 块行
        - 加载对应的 Q 块（一次性）
        - 遍历该行的有效 K 块，按需加载
        
        Args:
            mQ, mK, mV: Q/K/V 的全局内存张量
            sQ, sK, sV: Q/K/V 的共享内存张量
            tma_atom_*: TMA copy atoms
            pipeline_k, pipeline_v: 异步 pipeline
            mbar_ptr_Q: Q 的 barrier 指针
            block_info: 块信息
            SeqlenInfoCls: 序列长度信息类
            TileSchedulerCls: Tile 调度器类（应该是 BlockSparseTileScheduler）
        """
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        
        # 只有 warp 0 执行加载任务
        if warp_idx_in_wg == 0:
            kv_producer_state = pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, self.num_stages
            )
            
            # 创建块稀疏调度器，获取分配的 Q 块行
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            
            # 对于块稀疏调度器，is_valid_tile 表示这个 CUDA block 是否有工作
            if work_tile.is_valid_tile:
                # 获取分配的 Q 块行坐标
                m_block, head_idx, batch_idx = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                
                # ==================== 准备 Q/K/V 张量 ====================
                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                head_idx_kv = head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
                mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
                mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]
                
                gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
                gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (None, 0))
                
                # ==================== 准备 Q 的 TMA partition ====================
                if const_expr(self.use_tma_Q):
                    gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
                    tQsQ, tQgQ = cpasync.tma_partition(
                        tma_atom_Q,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sQ, 0, 2),
                        cute.group_modes(gQ, 0, 2),
                    )
                
                # ==================== 准备 K/V 的 TMA partition ====================
                tKsK, tKgK = cpasync.tma_partition(
                    tma_atom_K,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sK, 0, 2),
                    cute.group_modes(gK, 0, 2),
                )
                tVsV, tVgV = cpasync.tma_partition(
                    tma_atom_V,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sV, 0, 2),
                    cute.group_modes(gV, 0, 2),
                )
                
                load_K = partial(self.load_K, tma_atom_K, tKgK, tKsK, pipeline_k)
                load_V = partial(self.load_K, tma_atom_V, tVgV, tVsV, pipeline_v)
                
                # ==================== 块稀疏加载 K/V ====================
                # 获取掩码迭代器
                mask_iterator = tile_scheduler._mask_iterator
                
                # 获取 n_block 的有效范围（用于边界检查）
                n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
                
                # 收集所有有效的 n_block（按掩码值降序）
                valid_n_blocks = []
                for n_block in range(n_block_min, n_block_max):
                    mask_val = mask_iterator.mask_val(n_block)
                    if mask_val >= 0:
                        valid_n_blocks.append(n_block)
                
                num_valid_blocks = len(valid_n_blocks)
                
                if num_valid_blocks > 0:
                    # 第一次迭代：同时加载 Q 和第一个 K 块
                    n_block = valid_n_blocks[num_valid_blocks - 1]
                    pipeline_k.producer_acquire(
                        kv_producer_state,
                        extra_tx_count=self.tma_copy_bytes["Q"] if const_expr(self.use_tma_Q) else 0
                    )
                    if const_expr(self.use_tma_Q):
                        cute.copy(tma_atom_Q, tQgQ, tQsQ, tma_bar_ptr=pipeline_k.producer_get_barrier(kv_producer_state))
                    load_K(block=n_block, producer_state=kv_producer_state)
                    
                    if const_expr(not self.intra_wg_overlap):
                        # 非重叠模式：顺序加载 K 和 V
                        pipeline_v.producer_acquire(kv_producer_state)
                        load_V(block=n_block, producer_state=kv_producer_state)
                        kv_producer_state.advance()
                        
                        # 加载剩余的 K/V 块
                        for i in cutlass.range(num_valid_blocks - 1, unroll=1):
                            n_block = valid_n_blocks[num_valid_blocks - 2 - i]
                            pipeline_k.producer_acquire(kv_producer_state)
                            load_K(block=n_block, producer_state=kv_producer_state)
                            pipeline_v.producer_acquire(kv_producer_state)
                            load_V(block=n_block, producer_state=kv_producer_state)
                            kv_producer_state.advance()
                    else:
                        # 重叠模式：交错加载 K 和 V
                        for i in cutlass.range(num_valid_blocks - 1, unroll=1):
                            n_block_prev = valid_n_blocks[num_valid_blocks - 1 - i]
                            n_block = valid_n_blocks[num_valid_blocks - 2 - i]
                            kv_producer_state_prev = kv_producer_state.clone()
                            kv_producer_state.advance()
                            pipeline_k.producer_acquire(kv_producer_state)
                            load_K(block=n_block, producer_state=kv_producer_state)
                            pipeline_v.producer_acquire(kv_producer_state_prev)
                            load_V(block=n_block_prev, producer_state=kv_producer_state_prev)
                        
                        # 加载最后一个 V 块
                        n_block = valid_n_blocks[0]
                        pipeline_v.producer_acquire(kv_producer_state)
                        load_V(block=n_block, producer_state=kv_producer_state)
                        kv_producer_state.advance()
            
            # 注意：块稀疏调度器没有持久化循环
            # 每个 CUDA block 只处理一个 Q 块行，处理完就结束
    
