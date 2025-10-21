# Copyright (c) 2025, Tri Dao.

from typing import Optional, Tuple
from dataclasses import dataclass, fields

import cutlass
import cutlass.cute as cute
from cutlass import Int32

import sglang.srt.sparse_attention.kernels.attention.utils as utils
from sglang.srt.sparse_attention.kernels.attention.fast_math import FastDivmod, clz


@dataclass
class ParamsBase:
    def __extract_mlir_values__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, cutlass.Constexpr)]
        values, self._values_pos = [], []
        for obj in non_constexpr_fields:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {n: f for n, f in all_fields.items() if isinstance(f, cutlass.Constexpr)}
        non_constexpr_fields = {
            n: f for n, f in all_fields.items() if not isinstance(f, cutlass.Constexpr)
        }
        for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)


@dataclass
class TileSchedulerArguments(ParamsBase):
    num_block: Int32
    num_head: Int32
    num_batch: Int32
    seqlen_k: Int32
    headdim: Int32
    headdim_v: Int32
    total_q: Int32
    tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
    mCuSeqlensQ: Optional[cute.Tensor] = None
    mSeqUsedQ: Optional[cute.Tensor] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    element_size: cutlass.Constexpr[int] = 2
    is_persistent: cutlass.Constexpr[bool] = False
    lpt: cutlass.Constexpr[bool] = False


class SingleTileScheduler:
    """
    单tile调度器：每个CUDA block处理一个tile的工作
    这是最简单的调度策略，每个block独立处理一个tile，不涉及持久化或负载均衡
    """
    
    @dataclass
    class Params(ParamsBase):
        """
        调度器参数类
        存储调度所需的基本维度信息
        """
        num_block: Int32  # Q序列的block数量（按m_block_size划分）
        num_head: Int32   # KV头的数量
        num_batch: Int32  # batch大小

        @staticmethod
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileScheduler.Params":
            """
            从TileSchedulerArguments创建Params实例
            只提取必要的维度信息
            """
            return SingleTileScheduler.Params(args.num_block, args.num_head, args.num_batch)

    def __init__(self, blk_coord: cute.Coord, *, loc=None, ip=None):
        """
        初始化调度器实例
        
        Args:
            blk_coord: CUDA block的坐标 (block_idx, head_idx, batch_idx)
            loc: MLIR location信息
            ip: MLIR insertion point
        """
        self._blk_coord = blk_coord  # 当前block的坐标
        self._is_first_block = True  # 标记是否是第一个block（用于某些初始化逻辑）
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        """
        将TileSchedulerArguments转换为调度器的底层参数
        这是host端调用的接口
        """
        return SingleTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileScheduler":
        """
        在device端创建调度器实例
        获取当前CUDA block的索引作为工作坐标
        """
        blk_coord = cute.arch.block_idx()  # 获取当前block的3D索引
        return SingleTileScheduler(blk_coord, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        """
        计算CUDA grid的维度
        对于SingleTileScheduler，grid维度直接对应工作维度
        
        Returns:
            (num_block, num_head, num_batch): 3D grid维度
        """
        return params.num_block, params.num_head, params.num_batch

    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        """
        获取当前block要处理的工作tile信息
        
        Returns:
            WorkTileInfo: 包含tile坐标和是否是第一个block的信息
        """
        return cutlass.utils.WorkTileInfo(self._blk_coord, self._is_first_block)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        """
        获取初始工作tile信息
        对于SingleTileScheduler，初始工作就是当前工作
        """
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        """
        预取下一个工作tile
        对于SingleTileScheduler，每个block只处理一个tile，所以这是空操作
        """
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        """
        推进到下一个工作tile
        对于SingleTileScheduler，只是标记不再是第一个block
        （实际上每个block只处理一个tile，这个方法主要用于统一接口）
        """
        self._is_first_block = False

    def __extract_mlir_values__(self):
        """
        提取MLIR值用于序列化
        将Python对象转换为MLIR IR中的值
        """
        values, self._values_pos = [], []
        for obj in [self._blk_coord]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))  # 记录每个对象的值数量
        return values

    def __new_from_mlir_values__(self, values):
        """
        从MLIR值重建对象
        用于反序列化，将MLIR IR中的值转换回Python对象
        
        Args:
            values: MLIR值列表
            
        Returns:
            重建的SingleTileScheduler实例
        """
        obj_list = []
        # 根据之前记录的位置信息，重建每个对象
        for obj, n_items in zip([self._blk_coord], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        # 使用重建的blk_coord创建新实例
        return SingleTileScheduler(*(tuple(obj_list)), loc=self._loc)


class StaticPersistentTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block_divmod: FastDivmod
        num_head_divmod: FastDivmod
        total_blocks: Int32

        @staticmethod
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "StaticPersistentTileScheduler.Params":
            total_blocks = args.num_block * args.num_head * args.num_batch
            return StaticPersistentTileScheduler.Params(
                FastDivmod.create(args.num_block), FastDivmod.create(args.num_head), total_blocks
            )

    def __init__(self, params: Params, tile_idx: Int32, *, loc=None, ip=None):
        self.params = params
        self._tile_idx = tile_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return StaticPersistentTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(params: Params, *, loc=None, ip=None) -> "StaticPersistentTileScheduler":
        tile_idx = cute.arch.block_idx()[0]
        return StaticPersistentTileScheduler(params, tile_idx, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        hardware_info = cutlass.utils.HardwareInfo()
        sm_count = hardware_info.get_device_multiprocessor_count()
        return (cutlass.min(sm_count, params.total_blocks), Int32(1), Int32(1))

    # @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        hn_idx, block_idx = self.params.num_block_divmod.divmod(self._tile_idx)
        batch_idx, head_idx = self.params.num_head_divmod.divmod(hn_idx)
        is_valid = self._tile_idx < self.params.total_blocks
        # if cute.arch.thread_idx()[0] == 0:
        #     cute.printf("TileScheduler: tile_idx=%d, hn_idx=%d, block_idx=%d, batch_idx=%d, head_idx=%d, is_valid=%d", self._tile_idx, hn_idx, block_idx, batch_idx, head_idx, is_valid)
        return cutlass.utils.WorkTileInfo(
            (Int32(block_idx), Int32(head_idx), Int32(batch_idx)), is_valid
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._tile_idx += cute.arch.grid_dim()[0]

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._tile_idx], self._values_pos,):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return StaticPersistentTileScheduler(*(tuple(obj_list)), loc=self._loc)


class SingleTileLPTScheduler:
    @dataclass
    class Params(ParamsBase):
        total_blocks: Int32
        num_block_divmod: FastDivmod
        num_head_divmod: FastDivmod
        l2_minor_divmod: FastDivmod
        l2_major_divmod: FastDivmod
        l2_minor_residual_divmod: FastDivmod
        num_hb_quotient: Int32

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileLPTScheduler.Params":
            # cute.printf(args.num_block, args.num_head, args.num_batch, args.seqlen_k, args.headdim, args.headdim_v, args.total_q, args.tile_shape_mn, args.qhead_per_kvhead_packgqa, args.element_size)
            size_one_kv_head = args.seqlen_k * (args.headdim + args.headdim_v) * args.element_size
            size_one_head = size_one_kv_head
            size_l2 = 50 * 1024 * 1024  # 40 MB for K & V
            # Swizzle is the size of each "section". Round swizzle to a power of 2
            # Need to be careful about the case where only one head will fit
            # swizzle is how many heads can fit in L2
            # swizzle = 1 if size_l2 < size_one_head else (size_l2 // size_one_head)
            # Seems faster if swizzle if a power of 2
            log2_floor = lambda n: 31 - clz(n)
            swizzle = 1 if size_l2 < size_one_head else (1 << log2_floor(size_l2 // size_one_head))
            # swizzle = 1 if size_l2 < size_one_head else (size_l2 // size_one_head)
            # If we're in the last section (called residual), we don't want to divide by
            # swizzle. Instead we want to divide by the remainder.
            num_hb_quotient = (args.num_head * args.num_batch) // swizzle
            num_hb_remainder = (args.num_head * args.num_batch) % swizzle
            return SingleTileLPTScheduler.Params(
                total_blocks=args.num_block * args.num_head * args.num_batch,
                num_block_divmod=FastDivmod.create(args.num_block),
                num_head_divmod=FastDivmod.create(args.num_head),
                l2_minor_divmod=FastDivmod.create(swizzle),
                l2_major_divmod=FastDivmod.create(swizzle * args.num_block),
                l2_minor_residual_divmod=FastDivmod.create(
                    max(num_hb_remainder, 1)
                ),  # don't divide by 0
                num_hb_quotient=Int32(num_hb_quotient),
            )

    def __init__(self, params: Params, tile_idx: Int32, *, loc=None, ip=None):
        self.params = params
        self._tile_idx = tile_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return SingleTileLPTScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileLPTScheduler":
        tile_idx = cute.arch.block_idx()[0]
        return SingleTileLPTScheduler(params, tile_idx, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        return (params.total_blocks, Int32(1), Int32(1))

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        params = self.params
        # Implement LPT scheduling coordinate calculation
        bidhb, l2_mod = params.l2_major_divmod.divmod(self._tile_idx)
        # If we're in the last section (called residual), we don't want to divide by
        # swizzle. Instead we want to divide by the remainder.
        block, bidhb_residual = 0, 0
        if bidhb < params.num_hb_quotient:
            block, bidhb_residual = params.l2_minor_divmod.divmod(l2_mod)
        else:
            block, bidhb_residual = params.l2_minor_residual_divmod.divmod(l2_mod)
        bidhb_actual = bidhb * params.l2_minor_divmod.divisor + bidhb_residual
        batch_idx, head_idx = params.num_head_divmod.divmod(bidhb_actual)
        # Longest-processing-time-first
        block = params.num_block_divmod.divisor - 1 - block
        is_valid = self._tile_idx < params.total_blocks
        return cutlass.utils.WorkTileInfo(
            (Int32(block), Int32(head_idx), Int32(batch_idx)), is_valid
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        # Single tile scheduler - set to invalid tile_idx to indicate no more work
        self._tile_idx = self.params.total_blocks

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._tile_idx], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return SingleTileLPTScheduler(*(tuple(obj_list)), loc=self._loc)


class SingleTileVarlenScheduler:


    @dataclass
    class Params(ParamsBase):
        num_head: Int32
        num_batch: Int32
        total_q: Int32
        max_kvblock_in_l2: Int32
        tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
        mCuSeqlensQ: Optional[cute.Tensor] = None
        mSeqUsedQ: Optional[cute.Tensor] = None
        qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
        lpt: cutlass.Constexpr[bool] = False

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileVarlenScheduler.Params":
            size_l2 = 50 * 1024 * 1024  # 50 MB for K & V
            max_kvblock_in_l2 = size_l2 // ((args.headdim + args.headdim_v) * args.element_size * args.tile_shape_mn[1])
            assert args.mCuSeqlensQ is not None or args.mSeqUsedQ is not None, (
                "At least one of mCuSeqlensQ or mSeqUsedQ must be provided"
            )
            return SingleTileVarlenScheduler.Params(
                num_head=args.num_head,
                num_batch=args.num_batch,
                total_q=args.total_q,
                max_kvblock_in_l2=max_kvblock_in_l2,
                tile_shape_mn=args.tile_shape_mn,
                mCuSeqlensQ=args.mCuSeqlensQ,
                mSeqUsedQ=args.mSeqUsedQ,
                qhead_per_kvhead_packgqa=args.qhead_per_kvhead_packgqa,
                lpt=args.lpt,
            )

    def __init__(self, params: Params, tile_idx: Int32, *, loc=None, ip=None):
        self.params = params
        self._tile_idx = tile_idx
        self._is_first_block = True
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return SingleTileVarlenScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileVarlenScheduler":
        tile_idx = cute.arch.block_idx()[0]
        return SingleTileVarlenScheduler(params, tile_idx, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        total_blocks_max = (
            params.total_q + params.num_batch * (params.tile_shape_mn[0] - 1)
        ) // params.tile_shape_mn[0]
        return (total_blocks_max * params.num_head, Int32(1), Int32(1))

    @cute.jit
    def _get_num_m_blocks(self, lane: Int32, bidb_start: Int32) -> Int32:
        params = self.params
        batch_idx = lane + bidb_start
        if cutlass.const_expr(params.mSeqUsedQ is not None):
            seqlen = Int32(0)
            if batch_idx < params.num_batch:
                seqlen = params.mSeqUsedQ[batch_idx]
        else:
            assert params.mCuSeqlensQ is not None
            cur_cu_seqlen = Int32(0)
            if batch_idx <= params.num_batch:
                cur_cu_seqlen = params.mCuSeqlensQ[batch_idx]
            next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
            seqlen = next_cu_seqlen - cur_cu_seqlen
        if cutlass.const_expr(params.qhead_per_kvhead_packgqa > 1):
            seqlen *= params.qhead_per_kvhead_packgqa
        return (
            cute.ceil_div(seqlen, params.tile_shape_mn[0])
            if batch_idx < params.num_batch and lane < cute.arch.WARP_SIZE - 1
            else Int32(0)
        )

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        params = self.params
        lane_idx = cute.arch.lane_idx()
        num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=0)
        num_m_blocks_cumulative = utils.warp_prefix_sum(num_m_blocks, lane_idx)
        # Total number of blocks for the next 31 batches
        m_blocks_in_group = cute.arch.shuffle_sync(num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1)
        # Same for all lanes
        group_end_tile = m_blocks_in_group * params.num_head
        # if cute.arch.thread_idx()[0] == 128 + 31: cute.printf("SingleTileVarlenScheduler: tile_idx=%d, group_end_tile = %d, num_m_blocks=%d, num_m_blocks_cumulative = %d, m_blocks_in_group = %d", self._tile_idx, group_end_tile, num_m_blocks, num_m_blocks_cumulative, m_blocks_in_group)
        block, head_idx, batch_idx = Int32(0), Int32(0), Int32(0)
        next_tile_idx = self._tile_idx
        while group_end_tile <= next_tile_idx:
            batch_idx += cute.arch.WARP_SIZE - 1
            if batch_idx >= params.num_batch:
                batch_idx = Int32(params.num_batch)
                group_end_tile = next_tile_idx + 1
            else:
                num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=batch_idx)
                num_m_blocks_cumulative = utils.warp_prefix_sum(num_m_blocks, lane_idx)
                m_blocks_in_group = cute.arch.shuffle_sync(
                    num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1
                )
                group_end_tile += m_blocks_in_group * params.num_head
        is_valid = False
        if batch_idx >= params.num_batch:
            block, head_idx, batch_idx = Int32(0), Int32(0), Int32(params.num_batch)
        else:
            group_start_tile = group_end_tile - m_blocks_in_group * params.num_head
            # if cute.arch.thread_idx()[0] == 128 + 31: cute.printf("SingleTileVarlenScheduler: tile_idx=%d, group_end_tile = %d, num_m_blocks=%d, batch_idx = %d", self._tile_idx, group_end_tile, num_m_blocks, batch_idx)
            # The next problem to process is the first one that does not have ending tile position
            # that is greater than or equal to tile index.
            batch_idx_in_group = cute.arch.popc(
                cute.arch.vote_ballot_sync(
                    group_start_tile + num_m_blocks_cumulative * params.num_head <= next_tile_idx
                )
            )
            batch_idx += batch_idx_in_group
            num_m_blocks_prev_lane = (
                0
                if batch_idx_in_group == 0
                else cute.arch.shuffle_sync(num_m_blocks_cumulative, batch_idx_in_group - 1)
            )
            num_m_blocks = cute.arch.shuffle_sync(num_m_blocks, batch_idx_in_group)
            mh_block = next_tile_idx - group_start_tile - num_m_blocks_prev_lane * params.num_head
            if cutlass.const_expr(params.lpt):
                # This is a version of the SingleTileLPTScheduler, complicated by the fact that
                # the seqlen can vary per batch.
                # TODO: is there any case where num_m_blocks is 0?
                # TODO: by right we should read the seqlen_kv but we're assuming seqlen_q == seqlen_k here
                num_n_blocks = num_m_blocks * params.tile_shape_mn[0] // params.qhead_per_kvhead_packgqa // params.tile_shape_mn[1]
                # nheads_in_l2 = min(max(self.max_kvblock_in_l2 // num_n_blocks, 1), self.num_head)
                # Seems faster to have this be a power of 2
                nheads_in_l2 = 16 if num_n_blocks * 16 <= params.max_kvblock_in_l2 else (8 if num_n_blocks * 8 <= params.max_kvblock_in_l2 else (4 if num_n_blocks * 4 <= params.max_kvblock_in_l2 else (2 if num_n_blocks * 2 <= params.max_kvblock_in_l2 else 1)))
                nheads_in_l2 = min(nheads_in_l2, params.num_head)
                mh_in_l2 = nheads_in_l2 * num_m_blocks
                section_idx = mh_block // mh_in_l2
                l2_mod = mh_block - section_idx * mh_in_l2
                # Deal with tail section
                nheads_in_this_section = nheads_in_l2 if nheads_in_l2 * (section_idx + 1) <= params.num_head else params.num_head - section_idx * nheads_in_l2
                block = l2_mod // nheads_in_this_section
                head_idx_residual = l2_mod - block * nheads_in_this_section
                head_idx = section_idx * nheads_in_l2 + head_idx_residual
                block = num_m_blocks - 1 - block
            else:
                head_idx = mh_block // num_m_blocks
                block = mh_block - head_idx * num_m_blocks
            is_valid = self._is_first_block and batch_idx < params.num_batch
        # if cute.arch.thread_idx()[0] == 128: cute.printf("SingleTileVarlenScheduler: tile_idx=%d, batch_idx=%d, head_idx=%d, block=%d, is_valid = %d", self._tile_idx, batch_idx, head_idx, block, is_valid)
        return cutlass.utils.WorkTileInfo(
            (Int32(block), Int32(head_idx), Int32(batch_idx)), is_valid
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        # Single tile scheduler - set to invalid tile_idx to indicate no more work
        self._is_first_block = False

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._tile_idx], self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return SingleTileVarlenScheduler(*(tuple(obj_list)), loc=self._loc)



@dataclass
class BlockSparseMaskArguments(ParamsBase):
    """
    块稀疏掩码参数
    
    与 C++ fwdBlockmask 对应，存储预计算的块稀疏掩码信息
    """
    mBlockmask: cute.Tensor  # 块掩码张量 [B*num_heads, M_blocks, N_blocks]，存储优先级值
    m_block_dim: cutlass.Constexpr[int]  # 大块的 M 维度
    n_block_dim: cutlass.Constexpr[int]  # 大块的 N 维度
    tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]  # CUDA kernel 块大小 (kBlockM, kBlockN)
    seqlen_q_rounded: cutlass.Constexpr[int]  # 向上取整的 Q 序列长度
    seqlen_k_rounded: cutlass.Constexpr[int]  # 向上取整的 K 序列长度
    num_blocksparse_heads: cutlass.Constexpr[int]  # 稀疏掩码类型数量


class BlockSparseMaskIterator:
    """
    块稀疏掩码迭代器
    
    <design>
    设计目标：
    - 提供基于预计算掩码的高效块遍历功能
    - 支持优先级调度和二分查找优化
    - 在 device 端运行，最小化内存访问和分支开销
    
    核心概念：
    1. 两级块结构：
       - 大块 (m_block_dim x n_block_dim)：预计算掩码的粒度
       - 小块 (kBlockM x kBlockN)：CUDA kernel tile 的粒度
       - 转换因子：row_factor = m_block_dim / kBlockM, col_factor = n_block_dim / kBlockN
    
    2. 优先级机制：
       - 掩码值 >= 0：该大块需要计算，值越大优先级越高
       - 掩码值 = -1：该大块被屏蔽，不需要计算
       - 子块优先级：col_factor * mask_val + col_factor - 1 - block_col_offset
         （同一大块内，列索引越小的子块优先级越高）
    
    3. 遍历策略：
       - 线性扫描：从 n_block_min 到 n_block_max 顺序查找有效块
       - 二分查找：快速定位优先级阈值对应的块范围（用于 load balancing）
    
    使用场景：
    - 在 BlockSparseTileScheduler 中，每个 CUDA block 持有一个迭代器
    - 迭代器负责遍历分配给该 CUDA block 的 Q 块行中的所有活跃 K 块
    - 通过 find_next_valid_block() 和 advance() 实现增量遍历
    
    性能考虑：
    - 掩码数据已预加载到 shared memory 或寄存器
    - mask_val() 查询为 O(1) 操作
    - 二分查找为 O(log N) 操作，用于快速跳过大段无效块
    </design>
    
    对应 C++ 的 fwdBlockmask 类
    """
    
    def __init__(
        self,
        blockmask_ptr: cute.Tensor,  # 指向当前 Q 块行的掩码指针
        max_block_idx: Int32,  # 最大块索引
        m_block_dim: Int32,  # 大块 M 维度
        n_block_dim: Int32,  # 大块 N 维度
        row_factor: Int32,  # M 维度转换因子
        col_factor: Int32,  # N 维度转换因子
        n_block_min: Int32,  # 需要处理的最小 N 块索引
        n_block_max: Int32,  # 需要处理的最大 N 块索引
        *,
        loc=None,
        ip=None
    ):
        """
        初始化块稀疏掩码迭代器
        
        参数对应 C++ fwdBlockmask 构造函数的计算结果
        """
        self._blockmask_ptr = blockmask_ptr
        self._max_block_idx = max_block_idx
        self._m_block_dim = m_block_dim
        self._n_block_dim = n_block_dim
        self._row_factor = row_factor
        self._col_factor = col_factor
        self._n_block_min = n_block_min
        self._n_block_max = n_block_max
        self._current_n_block = n_block_min
        self._loc = loc
        self._ip = ip
    
    @cute.jit
    def mask_val(self, block_col_idx: Int32, *, loc=None, ip=None) -> Int32:
        """
        查询指定列块的掩码值/优先级
        
        对应 C++ 的 mask_val() 方法
        
        返回值：
            >= 0: 该块需要计算，返回值表示优先级（值越大优先级越高）
            -1: 该块被掩码屏蔽，不需要计算
        """
        # 边界检查
        if block_col_idx > self._max_block_idx or block_col_idx < 0:
            return Int32(-1)
        
        # 将 CUDA kernel 块索引转换为大块索引
        real_block_idx = block_col_idx // self._col_factor
        block_col_offset = block_col_idx % self._col_factor
        
        # 从预计算的掩码中读取该大块的值
        mask_val = self._blockmask_ptr[real_block_idx]
        
        # 计算子块的优先级
        # 公式：col_factor * mask_val + col_factor - 1 - block_col_offset
        return (
            Int32(-1) if mask_val == -1
            else self._col_factor * mask_val + self._col_factor - 1 - block_col_offset
        )
    
    @cute.jit
    def max_no_larger(self, target: Int32, *, loc=None, ip=None) -> Int32:
        """
        二分查找最大的块索引，使得其 mask_val <= target
        
        对应 C++ 的 max_no_larger() 方法
        用于快速定位需要处理的块范围
        """
        # 空范围检查
        if self._max_block_idx == 0:
            return Int32(-1)
        
        # 二分查找
        left = Int32(0)
        right = self._max_block_idx - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            mid_val = self.mask_val(mid, loc=loc, ip=ip)
            
            if mid_val > target:
                left = mid + 1
            else:
                right = mid - 1
        
        # 验证结果
        result_val = self.mask_val(left, loc=loc, ip=ip)
        return left if (left < self._max_block_idx and result_val <= target) else Int32(-1)
    
    @cute.jit
    def find_next_valid_block(self, *, loc=None, ip=None) -> Int32:
        """
        从当前位置查找下一个有效块
        
        返回 -1 表示没有更多有效块
        """
        while self._current_n_block <= self._n_block_max:
            if self.mask_val(self._current_n_block, loc=loc, ip=ip) >= 0:
                return self._current_n_block
            self._current_n_block += 1
        
        return Int32(-1)
    
    def is_done(self, *, loc=None, ip=None) -> bool:
        """检查是否完成遍历"""
        return self._current_n_block > self._n_block_max
    
    def current_n_block(self, *, loc=None, ip=None) -> Int32:
        """获取当前 N 块索引"""
        return self._current_n_block
    
    def advance(self, *, loc=None, ip=None):
        """推进到下一个块"""
        self._current_n_block += 1
    
    def __extract_mlir_values__(self):
        """提取 MLIR 值"""
        values, self._values_pos = [], []
        for obj in [
            self._blockmask_ptr,
            self._max_block_idx,
            self._m_block_dim,
            self._n_block_dim,
            self._row_factor,
            self._col_factor,
            self._n_block_min,
            self._n_block_max,
            self._current_n_block,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values
    
    def __new_from_mlir_values__(self, values):
        """从 MLIR 值重建对象"""
        obj_list = []
        for obj, n_items in zip(
            [
                self._blockmask_ptr,
                self._max_block_idx,
                self._m_block_dim,
                self._n_block_dim,
                self._row_factor,
                self._col_factor,
                self._n_block_min,
                self._n_block_max,
                self._current_n_block,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return BlockSparseMaskIterator(*obj_list, loc=self._loc)


@dataclass
class BlockSparseTileSchedulerArguments(ParamsBase):
    """
    块稀疏 Tile 调度器参数
    
    整合所有调度所需的参数
    """
    # 基础调度参数（继承自 TileSchedulerArguments）
    num_block: Int32
    num_head: Int32
    num_batch: Int32
    seqlen_k: Int32
    tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
    
    # 块稀疏专用参数
    mBlockmask: cute.Tensor  # 块掩码张量
    m_block_dim: cutlass.Constexpr[int]
    n_block_dim: cutlass.Constexpr[int]
    seqlen_q_rounded: cutlass.Constexpr[int]
    seqlen_k_rounded: cutlass.Constexpr[int]
    num_blocksparse_heads: cutlass.Constexpr[int]
    mHeadMaskType: cute.Tensor  # [num_head] 每个头对应的掩码类型ID


class BlockSparseTileScheduler:
    """
    块稀疏 Tile 调度器
    
    结合 LUT (Look-Up Table) 方式和 mask-based iterator 方式
    - 使用 LUT 快速定位活跃的 Q 块行
    - 使用 mask iterator 遍历每行的 K 块
    """
    
    @dataclass
    class Params(ParamsBase):
        """
        调度器参数
        
        Attributes:
            total_active_rows: 活跃 Q 块行总数（grid x 维度）
            mRowIndices: [total_active_rows, 3] 映射 blockIdx.x -> (m, h, b)
            mBlockmaskPtr: 块掩码数据指针
            mHeadMaskType: [num_head] 每个头对应的掩码类型ID
            m_block_dim, n_block_dim: 大块维度
            row_factor, col_factor: 转换因子
            seqlen_k: 实际 K 序列长度
            seqlen_k_rounded: 向上取整的 K 序列长度
            num_blocksparse_heads: 稀疏掩码类型数量
        """
        total_active_rows: Int32
        mRowIndices: cute.Tensor  # [total_active_rows, 3]
        mBlockmaskPtr: cute.Tensor  # 块掩码数据
        mHeadMaskType: cute.Tensor  # [num_head] 每个头对应的掩码类型ID
        m_block_dim: Int32
        n_block_dim: Int32
        row_factor: Int32
        col_factor: Int32
        seqlen_k: Int32
        seqlen_k_rounded: Int32
        num_blocksparse_heads: Int32
        num_batch: Int32
        tile_shape_mn_0: Int32  # tile_shape_mn[0]
        tile_shape_mn_1: Int32  # tile_shape_mn[1]
        
        @staticmethod
        def create(
            args: BlockSparseTileSchedulerArguments, *, loc=None, ip=None
        ) -> "BlockSparseTileScheduler.Params":
            """
            从参数创建调度器参数
            
            预处理掩码数据，构建 LUT
            """
            import numpy as np
            import torch
            
            # 计算转换因子
            kBlockM, kBlockN = args.tile_shape_mn
            row_factor = args.m_block_dim // kBlockM
            col_factor = args.n_block_dim // kBlockN
            
            # 转换掩码到 CPU
            blockmask_host = args.mBlockmask.cpu().numpy()
            head_mask_type_host = args.mHeadMaskType.cpu().numpy()
            
            # blockmask shape: [B * num_blocksparse_heads, M_blocks, N_blocks]
            num_batch = args.num_batch
            num_head = args.num_head
            num_q_blocks = args.seqlen_q_rounded // args.m_block_dim
            
            # 收集所有活跃的 (m, h, b) 组合
            row_indices = []
            
            for b in range(num_batch):
                for h in range(num_head):
                    mask_type = head_mask_type_host[h]
                    
                    # mask_type > 0 表示使用块稀疏掩码
                    if mask_type <= 0:
                        continue
                    
                    mask_idx = b * args.num_blocksparse_heads + (mask_type - 1)
                    
                    for m in range(num_q_blocks):
                        # 检查该行是否有非零元素
                        row_mask = blockmask_host[mask_idx, m, :]
                        if np.any(row_mask >= 0):
                            row_indices.append((m, h, b))
            
            total_active_rows = len(row_indices)
            
            # 处理空掩码情况
            if total_active_rows == 0:
                row_indices.append((0, 0, 0))
                total_active_rows = 1
            
            # 创建张量
            row_indices_tensor = cute.Tensor(
                torch.tensor(row_indices, dtype=torch.int32, device=args.mBlockmask.device),
                name="row_indices"
            )
            
            return BlockSparseTileScheduler.Params(
                total_active_rows=Int32(total_active_rows),
                mRowIndices=row_indices_tensor,
                mBlockmaskPtr=args.mBlockmask,
                mHeadMaskType=args.mHeadMaskType,
                m_block_dim=Int32(args.m_block_dim),
                n_block_dim=Int32(args.n_block_dim),
                row_factor=Int32(row_factor),
                col_factor=Int32(col_factor),
                seqlen_k=args.seqlen_k,
                seqlen_k_rounded=Int32(args.seqlen_k_rounded),
                num_blocksparse_heads=Int32(args.num_blocksparse_heads),
                num_batch=Int32(num_batch),
                tile_shape_mn_0=Int32(kBlockM),
                tile_shape_mn_1=Int32(kBlockN),
            )
    
    def __init__(
        self,
        blk_coord: cute.Coord,  # (m_block, head_idx, batch_idx)
        mask_iterator: BlockSparseMaskIterator,
        *,
        loc=None,
        ip=None
    ):
        """初始化调度器实例"""
        self._blk_coord = blk_coord
        self._mask_iterator = mask_iterator
        self._is_first_block = True
        self._loc = loc
        self._ip = ip
    
    @staticmethod
    def to_underlying_arguments(
        args: BlockSparseTileSchedulerArguments, *, loc=None, ip=None
    ) -> Params:
        """转换为底层参数"""
        return BlockSparseTileScheduler.Params.create(args, loc=loc, ip=ip)
    
    @staticmethod
    @cute.jit
    def create(params: Params, *, loc=None, ip=None) -> "BlockSparseTileScheduler":
        """
        在 device 端创建调度器实例
        
        每个 CUDA block 调用此方法初始化其调度器
        """
        # 获取当前 CUDA block 的 x 维度索引
        block_id_x = cute.arch.block_idx()[0]
        
        # 从 LUT 中获取分配给这个 CUDA block 的工作
        m_block = params.mRowIndices[block_id_x, 0]
        head_idx = params.mRowIndices[block_id_x, 1]
        batch_idx = params.mRowIndices[block_id_x, 2]
        
        blk_coord = cute.make_coord(m_block, head_idx, batch_idx)
        
        # 计算该行对应的掩码指针
        # 类似 C++ 中的 blockmask_ptr 计算
        mask_type = params.mHeadMaskType[head_idx]
        
        # 计算掩码索引
        # blockmask shape: [B * num_blocksparse_heads, M_blocks, N_blocks]
        mask_batch_offset = batch_idx * params.num_blocksparse_heads + (mask_type - 1)
        
        # 定位到当前 Q 块行的掩码
        blockmask_row_ptr = (
            params.mBlockmaskPtr[mask_batch_offset, m_block // params.row_factor, :]
        )
        
        # 计算最大块索引（基于实际序列长度）
        max_block_idx = cute.ceil_div(params.seqlen_k, params.n_block_dim) * params.col_factor
        
        # 创建掩码迭代器
        mask_iterator = BlockSparseMaskIterator(
            blockmask_ptr=blockmask_row_ptr,
            max_block_idx=max_block_idx,
            m_block_dim=params.m_block_dim,
            n_block_dim=params.n_block_dim,
            row_factor=params.row_factor,
            col_factor=params.col_factor,
            n_block_min=Int32(0),
            n_block_max=max_block_idx - 1,
            loc=loc,
            ip=ip
        )
        
        return BlockSparseTileScheduler(blk_coord, mask_iterator, loc=loc, ip=ip)
    
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        """
        计算 CUDA grid 维度
        
        x 维度 = 活跃行总数（每个 CUDA block 处理一个 Q 块行）
        """
        return (params.total_active_rows, Int32(1), Int32(1))
    
    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        """
        获取当前工作 tile 信息
        
        返回 (n_block, head_idx, batch_idx) 和有效性标志
        """
        # 查找下一个有效的 N 块
        n_block = self._mask_iterator.find_next_valid_block(loc=loc, ip=ip)
        is_valid = n_block >= 0 and self._is_first_block
        
        # 如果无效，返回默认值
        if n_block < 0:
            n_block = Int32(0)
        
        return cutlass.utils.WorkTileInfo(
            (n_block, self._blk_coord[1], self._blk_coord[2]), is_valid
        )
    
    def initial_work_tile_info(self, *, loc=None, ip=None):
        """获取初始工作 tile"""
        return self.get_current_work(loc=loc, ip=ip)
    
    def prefetch_next_work(self, *, loc=None, ip=None):
        """预取下一个工作（空操作）"""
        pass
    
    def advance_to_next_work(self, *, loc=None, ip=None):
        """
        推进到下一个 K 块
        
        移动迭代器到下一个活跃的 K 块
        """
        self._mask_iterator.advance(loc=loc, ip=ip)
        self._is_first_block = False
    
    def __extract_mlir_values__(self):
        """提取 MLIR 值"""
        values, self._values_pos = [], []
        for obj in [self._blk_coord, self._mask_iterator]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values
    
    def __new_from_mlir_values__(self, values):
        """从 MLIR 值重建对象"""
        obj_list = []
        for obj, n_items in zip([self._blk_coord, self._mask_iterator], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return BlockSparseTileScheduler(*obj_list, loc=self._loc)

