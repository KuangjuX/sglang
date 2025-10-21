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
class BlockSparseTileSchedulerArguments(ParamsBase):
    """
    Additional parameters for block sparse tile scheduler
    """
    mBlockmask: cute.Tensor  # Input block mask tensor (B, H, M_blocks, K_blocks)
    # Optional: specify the maximum number of K blocks per Q block row for memory allocation
    max_k_blocks_per_row: cutlass.Constexpr[int] = 256 

class BlockSparseRowIterator:
    def __init__(self, k_indices_ptr: cute.Tensor, num_k_blocks: Int32, *, loc=None, ip=None):
        self._k_indices_ptr = k_indices_ptr
        self._num_k_blocks = num_k_blocks
        self._k_block_offset = cute.int32(0)
        self._loc = loc
        self._ip = ip

    def is_empty_row(self, *, loc=None, ip=None) -> bool:
        return self._num_k_blocks == 0

    def current_n_block(self, *, loc=None, ip=None) -> Int32:
        return self._k_indices_ptr[self._k_block_offset]

    def advance(self, *, loc=None, ip=None):
        self._k_block_offset += 1

    def is_done(self, *, loc=None, ip=None) -> bool:
        return self._k_block_offset >= self._num_k_blocks



class BlockSparseTileScheduler:
    """
    Block sparse tile scheduler: handles sparse attention patterns
    Uses precomputed lookup tables to efficiently schedule sparse block computations
    """
    
    @dataclass
    class Params(ParamsBase):
        """
        Scheduler parameter class, stores precomputed lookup tables
        
        Attributes:
            total_active_rows: Total number of non-empty Q block rows, i.e., Grid x dimension
            mRowIndices: [total_active_rows, 3], maps blockIdx.x to (m, h, b)
            mBlockLUT: [B, H, M, max_k + 1], stores valid K block list for each Q block row
        """
        total_active_rows: Int32  # Total number of non-empty Q block rows, i.e., Grid x dimension
        mRowIndices: cute.Tensor  # [total_active_rows, 3], maps blockIdx.x to (m, h, b)
        mBlockLUT: cute.Tensor    # [B, H, M, max_k + 1], stores valid K block list for each Q block row

    @staticmethod
    def to_underlying_argumrnts(args: BlockSparseTileSchedulerArguments, *, loc=None, ip=None) -> Params:
        import numpy as np
        import torch

        # Convert block mask tensor to numpy array on host
        blockmask_host = args.mBlockmask.cpu().numpy()
        num_batch, num_head, num_q_blocks, _ = blockmask_host.shape

        row_indices = []
        max_k = args.max_k_blocks_per_row.value

        # 2. Initialize Block LUT
        block_lut_host = np.zeros((num_batch, num_head, num_q_blocks, max_k + 1), dtype=np.int32)


        for b in range(num_batch):
            for h in range(num_head):
                for m in range(num_q_blocks):
                    k_indices = np.flatnonzero(blockmask_host[b, h, m, :])

                    if k_indices.size > 0:
                        row_indices.append((m, h, b))

                        num_valid_k = min(k_indices.size, max_k)

                        if k_indices.size >  max_k:
                            print(f"Warning: more than {max_k} K blocks per Q block row for batch {b}, head {h}, Q block {m}")
                        
                        block_lut_host[b, h, m, 0] = num_valid_k

        total_active_rows = len(row_indices)
        if total_active_rows == 0:
            row_indices.append((0, 0, 0))
            total_active_rows = 1


        row_indices_tensor = cute.Tensor(
            torch.Tensor(row_indices, dtype=torch.int32, device=args.mBlockmask.device),
            name = "row_indices"
        )

        block_lut_tensor = cute.Tensor(
            torch.from_numpy(block_lut_host).to(args.mBlockmask.device),
            name = "block_lut"
        )

        return BlockSparseTileScheduler.Params(
            total_active_rows=total_active_rows,
            mRowIndices=row_indices_tensor,
            mBlockLUT=block_lut_tensor,
        )

    @staticmethod
    def create(params: Params, *, loc=None, ip=None) -> "BlockSparseTileScheduler":
        """
        Create a BlockSparseTileScheduler instance on the device side.
        
        This method is called by each CUDA block to initialize its scheduler.
        It retrieves the block's assigned work from the row indices lookup table
        and sets up an iterator for the sparse K blocks.
        
        Args:
            params: Scheduler parameters containing row indices and block lookup table
            loc: MLIR location information
            ip: MLIR insertion point
            
        Returns:
            A BlockSparseTileScheduler instance configured for this CUDA block
        """
        # Get the current CUDA block's x-dimension index
        # This serves as the index into the row_indices array
        block_id_x = cute.arch.block_idx_x()

        # Retrieve the Q block row (m_block), head index, and batch index
        # assigned to this CUDA block from the row_indices lookup table
        m_block = params.mRowIndices[block_id_x, 0]
        head_idx = params.mRowIndices[block_id_x, 1]
        batch_idx = params.mRowIndices[block_id_x, 2]
        
        # Create a coordinate tuple representing this block's position
        # in the (Q block, head, batch) space
        blk_coord = cute.make_coord(m_block, head_idx, batch_idx)

        # Get a pointer to this row's entry in the block lookup table (LUT)
        # The LUT stores: [num_k_blocks, k_index_0, k_index_1, ..., k_index_max]
        lut_entry_ptr = params.mBlockLUT.ptr(m_block, head_idx, batch_idx)
        
        # The first element in the LUT entry is the count of valid K blocks
        # for this Q block row
        num_k_blocks = lut_entry_ptr[0]

        # The remaining elements are the indices of the K blocks that are
        # non-zero (active) in the block sparse mask for this Q block row
        k_indices_ptr = lut_entry_ptr[1:]

        # Create an iterator that will traverse the active K blocks
        # for this Q block row during attention computation
        row_iterator = BlockSparseRowIterator(k_indices_ptr, num_k_blocks, loc=loc, ip=ip)

        # Return the scheduler instance with the block coordinate and row iterator
        return BlockSparseTileScheduler(blk_coord, row_iterator, loc=loc, ip=ip)