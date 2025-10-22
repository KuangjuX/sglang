"""
测试 BlockSparseTileScheduler 和 BlockSparseMaskIterator

测试分为三个层级：
1. 单元测试：测试各个组件的功能
2. 集成测试：测试组件间的协作
3. 端到端测试：测试完整的稀疏注意力计算
"""

import pytest
import torch
import numpy as np
from typing import List, Tuple

# 注意：这里假设可以导入 cutlass 和 cute
# 实际测试时需要根据环境调整
try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32
    
    from sglang.srt.sparse_attention.kernels.attention.tile_scheduler import (
        BlockSparseTileScheduler,
        BlockSparseTileSchedulerArguments,
        BlockSparseMaskIterator,
    )
    
    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False
    pytest.skip("CuTe/CUTLASS not available", allow_module_level=True)


class TestBlockSparseMaskIterator:
    """测试块稀疏掩码迭代器"""
    
    def test_mask_val_basic(self):
        """测试基本的掩码值查询"""
        # 创建简单的掩码: [5, -1, 3, -1]
        blockmask = torch.tensor([5, -1, 3, -1], dtype=torch.int32, device='cuda')
        
        # 参数：col_factor = 1 (没有子块划分)
        iterator = BlockSparseMaskIterator(
            blockmask_ptr=cute.Tensor(blockmask),
            max_block_idx=Int32(4),
            m_block_dim=Int32(128),
            n_block_dim=Int32(128),
            row_factor=Int32(1),
            col_factor=Int32(1),
            n_block_min=Int32(0),
            n_block_max=Int32(3),
        )
        
        # 测试每个块的掩码值
        assert iterator.mask_val(Int32(0)) == 5
        assert iterator.mask_val(Int32(1)) == -1
        assert iterator.mask_val(Int32(2)) == 3
        assert iterator.mask_val(Int32(3)) == -1
    
    def test_mask_val_with_subblocks(self):
        """测试有子块划分的掩码值计算"""
        # 大块掩码: [3, -1]
        blockmask = torch.tensor([3, -1], dtype=torch.int32, device='cuda')
        
        # col_factor = 2 (每个大块包含 2 个小块)
        iterator = BlockSparseMaskIterator(
            blockmask_ptr=cute.Tensor(blockmask),
            max_block_idx=Int32(4),
            m_block_dim=Int32(128),
            n_block_dim=Int32(128),
            row_factor=Int32(1),
            col_factor=Int32(2),
            n_block_min=Int32(0),
            n_block_max=Int32(3),
        )
        
        # 大块0 (值=3) 的子块
        # 小块0: 2*3 + 2 - 1 - 0 = 7
        # 小块1: 2*3 + 2 - 1 - 1 = 6
        assert iterator.mask_val(Int32(0)) == 7
        assert iterator.mask_val(Int32(1)) == 6
        
        # 大块1 (值=-1) 的子块
        assert iterator.mask_val(Int32(2)) == -1
        assert iterator.mask_val(Int32(3)) == -1
    
    def test_find_next_valid_block(self):
        """测试查找下一个有效块"""
        blockmask = torch.tensor([5, -1, 3, -1, 8], dtype=torch.int32, device='cuda')
        
        iterator = BlockSparseMaskIterator(
            blockmask_ptr=cute.Tensor(blockmask),
            max_block_idx=Int32(5),
            m_block_dim=Int32(128),
            n_block_dim=Int32(128),
            row_factor=Int32(1),
            col_factor=Int32(1),
            n_block_min=Int32(0),
            n_block_max=Int32(4),
        )
        
        # 第一个有效块
        n_block = iterator.find_next_valid_block()
        assert n_block == 0
        
        # 推进并查找下一个
        iterator.advance()
        n_block = iterator.find_next_valid_block()
        assert n_block == 2
        
        # 再推进
        iterator.advance()
        n_block = iterator.find_next_valid_block()
        assert n_block == 4
        
        # 没有更多
        iterator.advance()
        n_block = iterator.find_next_valid_block()
        assert n_block == -1
    
    def test_is_done(self):
        """测试迭代器完成状态"""
        blockmask = torch.tensor([5, -1], dtype=torch.int32, device='cuda')
        
        iterator = BlockSparseMaskIterator(
            blockmask_ptr=cute.Tensor(blockmask),
            max_block_idx=Int32(2),
            m_block_dim=Int32(128),
            n_block_dim=Int32(128),
            row_factor=Int32(1),
            col_factor=Int32(1),
            n_block_min=Int32(0),
            n_block_max=Int32(1),
        )
        
        assert not iterator.is_done()
        iterator.advance()
        iterator.advance()
        assert iterator.is_done()


class TestBlockSparseTileSchedulerParams:
    """测试调度器参数创建和 LUT 构建"""
    
    def test_params_creation_simple(self):
        """测试简单场景的参数创建"""
        batch_size = 2
        num_heads = 4
        num_mask_types = 1
        num_q_blocks = 4
        num_k_blocks = 4
        
        # 创建简单的局部窗口掩码
        blockmask = torch.full(
            (batch_size * num_mask_types, num_q_blocks, num_k_blocks),
            -1,
            dtype=torch.int32,
            device='cuda'
        )
        
        # 对角线附近为有效块
        for b in range(batch_size):
            for m in range(num_q_blocks):
                for n in range(max(0, m-1), min(num_k_blocks, m+2)):
                    blockmask[b, m, n] = 5
        
        # 所有头使用掩码类型 1
        head_mask_type = torch.ones(num_heads, dtype=torch.int32, device='cuda')
        
        # 创建参数
        args = BlockSparseTileSchedulerArguments(
            num_block=num_q_blocks,
            num_head=num_heads,
            num_batch=batch_size,
            seqlen_k=512,
            tile_shape_mn=(64, 64),
            mBlockmask=cute.Tensor(blockmask, name="blockmask"),
            m_block_dim=128,
            n_block_dim=128,
            seqlen_q_rounded=512,
            seqlen_k_rounded=512,
            num_blocksparse_heads=num_mask_types,
            mHeadMaskType=cute.Tensor(head_mask_type, name="head_mask_type"),
        )
        
        params = BlockSparseTileScheduler.Params.create(args)
        
        # 验证
        # 每个 Q 块行都有有效块，所以 total_active_rows = batch * heads * q_blocks
        expected_rows = batch_size * num_heads * num_q_blocks
        assert params.total_active_rows == expected_rows
        
        # 验证 row_indices 的形状
        assert params.mRowIndices.shape == (expected_rows, 3)
        
        # 验证转换因子
        assert params.row_factor == 128 // 64  # 2
        assert params.col_factor == 128 // 64  # 2
    
    def test_params_with_empty_rows(self):
        """测试有空行的掩码"""
        batch_size = 1
        num_heads = 2
        num_mask_types = 1
        num_q_blocks = 4
        num_k_blocks = 4
        
        # 创建掩码，只有部分行有效
        blockmask = torch.full(
            (batch_size * num_mask_types, num_q_blocks, num_k_blocks),
            -1,
            dtype=torch.int32,
            device='cuda'
        )
        
        # 只有 Q block 0 和 2 有有效块
        blockmask[0, 0, 0:2] = 5
        blockmask[0, 2, 1:3] = 5
        
        head_mask_type = torch.ones(num_heads, dtype=torch.int32, device='cuda')
        
        args = BlockSparseTileSchedulerArguments(
            num_block=num_q_blocks,
            num_head=num_heads,
            num_batch=batch_size,
            seqlen_k=512,
            tile_shape_mn=(64, 64),
            mBlockmask=cute.Tensor(blockmask, name="blockmask"),
            m_block_dim=128,
            n_block_dim=128,
            seqlen_q_rounded=512,
            seqlen_k_rounded=512,
            num_blocksparse_heads=num_mask_types,
            mHeadMaskType=cute.Tensor(head_mask_type, name="head_mask_type"),
        )
        
        params = BlockSparseTileScheduler.Params.create(args)
        
        # 只有 2 个 Q 块行 * 2 个头 = 4 个活跃行
        assert params.total_active_rows == 4
    
    def test_grid_shape_calculation(self):
        """测试 grid 维度计算"""
        batch_size = 2
        num_heads = 4
        num_mask_types = 1
        num_q_blocks = 8
        num_k_blocks = 8
        
        # 创建全局注意力掩码（所有块都有效）
        blockmask = torch.full(
            (batch_size * num_mask_types, num_q_blocks, num_k_blocks),
            5,
            dtype=torch.int32,
            device='cuda'
        )
        
        head_mask_type = torch.ones(num_heads, dtype=torch.int32, device='cuda')
        
        args = BlockSparseTileSchedulerArguments(
            num_block=num_q_blocks,
            num_head=num_heads,
            num_batch=batch_size,
            seqlen_k=1024,
            tile_shape_mn=(64, 64),
            mBlockmask=cute.Tensor(blockmask, name="blockmask"),
            m_block_dim=128,
            n_block_dim=128,
            seqlen_q_rounded=1024,
            seqlen_k_rounded=1024,
            num_blocksparse_heads=num_mask_types,
            mHeadMaskType=cute.Tensor(head_mask_type, name="head_mask_type"),
        )
        
        params = BlockSparseTileScheduler.Params.create(args)
        grid = BlockSparseTileScheduler.get_grid_shape(params)
        
        # Grid = (total_active_rows, 1, 1)
        expected_grid_x = batch_size * num_heads * num_q_blocks
        assert grid == (expected_grid_x, 1, 1)


class TestBlockSparseTileSchedulerIntegration:
    """集成测试：测试调度器在模拟 kernel 环境中的行为"""
    
    def test_work_distribution(self):
        """测试工作分配的正确性"""
        batch_size = 2
        num_heads = 2
        num_q_blocks = 4
        num_k_blocks = 4
        
        # 创建局部窗口掩码
        blockmask = torch.full(
            (batch_size, num_q_blocks, num_k_blocks),
            -1,
            dtype=torch.int32,
            device='cuda'
        )
        
        # 设置局部窗口（窗口大小=2）
        for b in range(batch_size):
            for m in range(num_q_blocks):
                for n in range(max(0, m-1), min(num_k_blocks, m+1)):
                    blockmask[b, m, n] = 10 - abs(m - n)
        
        head_mask_type = torch.ones(num_heads, dtype=torch.int32, device='cuda')
        
        args = BlockSparseTileSchedulerArguments(
            num_block=num_q_blocks,
            num_head=num_heads,
            num_batch=batch_size,
            seqlen_k=512,
            tile_shape_mn=(64, 64),
            mBlockmask=cute.Tensor(blockmask, name="blockmask"),
            m_block_dim=128,
            n_block_dim=128,
            seqlen_q_rounded=512,
            seqlen_k_rounded=512,
            num_blocksparse_heads=1,
            mHeadMaskType=cute.Tensor(head_mask_type, name="head_mask_type"),
        )
        
        params = BlockSparseTileScheduler.Params.create(args)
        
        # 验证 row_indices 的内容
        row_indices_host = params.mRowIndices.cpu().numpy()
        
        # 收集所有分配的 (m, h, b)
        work_assignments = set()
        for i in range(params.total_active_rows):
            m, h, b = row_indices_host[i]
            work_assignments.add((int(m), int(h), int(b)))
        
        # 验证每个非空的 (m, h, b) 都被分配了
        for b in range(batch_size):
            for h in range(num_heads):
                for m in range(num_q_blocks):
                    # 检查该行是否有有效块
                    row_mask = blockmask[b, m, :].cpu().numpy()
                    has_valid_blocks = np.any(row_mask >= 0)
                    
                    if has_valid_blocks:
                        assert (m, h, b) in work_assignments, \
                            f"非空行 ({m}, {h}, {b}) 没有被分配"
    
    def test_mask_iterator_coverage(self):
        """测试掩码迭代器能正确遍历所有有效块"""
        num_k_blocks = 8
        
        # 创建测试掩码：[5, -1, 3, -1, 7, -1, -1, 2]
        test_mask = [5, -1, 3, -1, 7, -1, -1, 2]
        blockmask = torch.tensor(test_mask, dtype=torch.int32, device='cuda')
        
        iterator = BlockSparseMaskIterator(
            blockmask_ptr=cute.Tensor(blockmask),
            max_block_idx=Int32(num_k_blocks),
            m_block_dim=Int32(128),
            n_block_dim=Int32(128),
            row_factor=Int32(1),
            col_factor=Int32(1),
            n_block_min=Int32(0),
            n_block_max=Int32(num_k_blocks - 1),
        )
        
        # 收集所有有效块
        valid_blocks = []
        while not iterator.is_done():
            n_block = iterator.find_next_valid_block()
            if n_block >= 0:
                valid_blocks.append(n_block)
                iterator.advance()
            else:
                break
        
        # 验证
        expected_valid = [i for i, val in enumerate(test_mask) if val >= 0]
        assert valid_blocks == expected_valid, \
            f"Expected {expected_valid}, got {valid_blocks}"


class TestSparsityPatterns:
    """测试不同的稀疏模式"""
    
    def create_local_window_mask(
        self, 
        num_q_blocks: int, 
        num_k_blocks: int, 
        window_size: int
    ) -> torch.Tensor:
        """创建局部窗口掩码"""
        mask = torch.full((num_q_blocks, num_k_blocks), -1, dtype=torch.int32)
        for m in range(num_q_blocks):
            start = max(0, m - window_size // 2)
            end = min(num_k_blocks, m + window_size // 2 + 1)
            for n in range(start, end):
                # 优先级：距离越近越高
                priority = 10 - abs(m - n)
                mask[m, n] = priority
        return mask
    
    def create_strided_mask(
        self, 
        num_q_blocks: int, 
        num_k_blocks: int, 
        stride: int
    ) -> torch.Tensor:
        """创建步长掩码（如 Longformer 的 strided attention）"""
        mask = torch.full((num_q_blocks, num_k_blocks), -1, dtype=torch.int32)
        for m in range(num_q_blocks):
            for n in range(0, num_k_blocks, stride):
                mask[m, n] = 5
        return mask
    
    def create_block_diagonal_mask(
        self, 
        num_q_blocks: int, 
        num_k_blocks: int, 
        block_size: int
    ) -> torch.Tensor:
        """创建块对角掩码"""
        mask = torch.full((num_q_blocks, num_k_blocks), -1, dtype=torch.int32)
        for m in range(num_q_blocks):
            block_idx = m // block_size
            start = block_idx * block_size
            end = min(num_k_blocks, (block_idx + 1) * block_size)
            mask[m, start:end] = 5
        return mask
    
    def test_local_window_pattern(self):
        """测试局部窗口模式"""
        num_q_blocks = 8
        num_k_blocks = 8
        window_size = 3
        
        mask = self.create_local_window_mask(num_q_blocks, num_k_blocks, window_size)
        
        # 计算理论稀疏率
        total_blocks = num_q_blocks * num_k_blocks
        valid_blocks = (mask >= 0).sum().item()
        sparsity = 1 - (valid_blocks / total_blocks)
        
        print(f"\n局部窗口模式 (window={window_size}):")
        print(f"  有效块: {valid_blocks}/{total_blocks}")
        print(f"  稀疏率: {sparsity:.1%}")
        
        # 窗口大小为 3 时，每行大约有 3 个有效块
        # 边界行可能少一些
        assert valid_blocks >= num_q_blocks * 2  # 至少每行 2 个
        assert valid_blocks <= num_q_blocks * window_size  # 最多每行 window_size 个
    
    def test_strided_pattern(self):
        """测试步长模式"""
        num_q_blocks = 8
        num_k_blocks = 8
        stride = 2
        
        mask = self.create_strided_mask(num_q_blocks, num_k_blocks, stride)
        
        total_blocks = num_q_blocks * num_k_blocks
        valid_blocks = (mask >= 0).sum().item()
        sparsity = 1 - (valid_blocks / total_blocks)
        
        print(f"\n步长模式 (stride={stride}):")
        print(f"  有效块: {valid_blocks}/{total_blocks}")
        print(f"  稀疏率: {sparsity:.1%}")
        
        # 每行有 num_k_blocks / stride 个有效块
        expected_per_row = num_k_blocks // stride
        expected_total = num_q_blocks * expected_per_row
        assert valid_blocks == expected_total
    
    def test_block_diagonal_pattern(self):
        """测试块对角模式"""
        num_q_blocks = 8
        num_k_blocks = 8
        block_size = 2
        
        mask = self.create_block_diagonal_mask(num_q_blocks, num_k_blocks, block_size)
        
        total_blocks = num_q_blocks * num_k_blocks
        valid_blocks = (mask >= 0).sum().item()
        sparsity = 1 - (valid_blocks / total_blocks)
        
        print(f"\n块对角模式 (block_size={block_size}):")
        print(f"  有效块: {valid_blocks}/{total_blocks}")
        print(f"  稀疏率: {sparsity:.1%}")
        
        # 每行有 block_size 个有效块
        expected_total = num_q_blocks * block_size
        assert valid_blocks == expected_total


class TestEdgeCases:
    """测试边界情况"""
    
    def test_empty_mask(self):
        """测试完全空的掩码"""
        batch_size = 1
        num_heads = 2
        num_q_blocks = 4
        num_k_blocks = 4
        
        # 所有块都无效
        blockmask = torch.full(
            (batch_size, num_q_blocks, num_k_blocks),
            -1,
            dtype=torch.int32,
            device='cuda'
        )
        
        head_mask_type = torch.ones(num_heads, dtype=torch.int32, device='cuda')
        
        args = BlockSparseTileSchedulerArguments(
            num_block=num_q_blocks,
            num_head=num_heads,
            num_batch=batch_size,
            seqlen_k=512,
            tile_shape_mn=(64, 64),
            mBlockmask=cute.Tensor(blockmask, name="blockmask"),
            m_block_dim=128,
            n_block_dim=128,
            seqlen_q_rounded=512,
            seqlen_k_rounded=512,
            num_blocksparse_heads=1,
            mHeadMaskType=cute.Tensor(head_mask_type, name="head_mask_type"),
        )
        
        params = BlockSparseTileScheduler.Params.create(args)
        
        # 空掩码会创建一个 dummy row
        assert params.total_active_rows == 1
    
    def test_full_mask(self):
        """测试完全密集的掩码"""
        batch_size = 1
        num_heads = 2
        num_q_blocks = 4
        num_k_blocks = 4
        
        # 所有块都有效
        blockmask = torch.full(
            (batch_size, num_q_blocks, num_k_blocks),
            5,
            dtype=torch.int32,
            device='cuda'
        )
        
        head_mask_type = torch.ones(num_heads, dtype=torch.int32, device='cuda')
        
        args = BlockSparseTileSchedulerArguments(
            num_block=num_q_blocks,
            num_head=num_heads,
            num_batch=batch_size,
            seqlen_k=512,
            tile_shape_mn=(64, 64),
            mBlockmask=cute.Tensor(blockmask, name="blockmask"),
            m_block_dim=128,
            n_block_dim=128,
            seqlen_q_rounded=512,
            seqlen_k_rounded=512,
            num_blocksparse_heads=1,
            mHeadMaskType=cute.Tensor(head_mask_type, name="head_mask_type"),
        )
        
        params = BlockSparseTileScheduler.Params.create(args)
        
        # 所有行都活跃
        expected = batch_size * num_heads * num_q_blocks
        assert params.total_active_rows == expected
    
    def test_single_valid_block_per_row(self):
        """测试每行只有一个有效块"""
        batch_size = 1
        num_heads = 1
        num_q_blocks = 4
        num_k_blocks = 4
        
        blockmask = torch.full(
            (batch_size, num_q_blocks, num_k_blocks),
            -1,
            dtype=torch.int32,
            device='cuda'
        )
        
        # 每行只有对角线块有效
        for m in range(num_q_blocks):
            blockmask[0, m, m] = 5
        
        head_mask_type = torch.ones(num_heads, dtype=torch.int32, device='cuda')
        
        args = BlockSparseTileSchedulerArguments(
            num_block=num_q_blocks,
            num_head=num_heads,
            num_batch=batch_size,
            seqlen_k=512,
            tile_shape_mn=(64, 64),
            mBlockmask=cute.Tensor(blockmask, name="blockmask"),
            m_block_dim=128,
            n_block_dim=128,
            seqlen_q_rounded=512,
            seqlen_k_rounded=512,
            num_blocksparse_heads=1,
            mHeadMaskType=cute.Tensor(head_mask_type, name="head_mask_type"),
        )
        
        params = BlockSparseTileScheduler.Params.create(args)
        
        # 所有行都活跃（每行有 1 个块）
        assert params.total_active_rows == num_q_blocks


# ==================== 辅助工具 ====================

def visualize_mask(mask: torch.Tensor, title: str = "Block Mask"):
    """可视化掩码矩阵"""
    print(f"\n{title}:")
    mask_np = mask.cpu().numpy()
    if mask_np.ndim == 3:
        mask_np = mask_np[0]  # 只显示第一个 batch
    
    num_q, num_k = mask_np.shape
    print("    ", end="")
    for n in range(num_k):
        print(f"K{n:2d} ", end="")
    print()
    
    for m in range(num_q):
        print(f"Q{m:2d}: ", end="")
        for n in range(num_k):
            val = mask_np[m, n]
            if val >= 0:
                print(f"{val:3d} ", end="")
            else:
                print("  - ", end="")
        print()
    print()


def compute_sparsity_stats(mask: torch.Tensor) -> dict:
    """计算稀疏度统计信息"""
    if mask.ndim == 3:
        mask = mask[0]
    
    total = mask.numel()
    valid = (mask >= 0).sum().item()
    sparsity = 1 - (valid / total)
    
    # 每行的有效块数
    valid_per_row = (mask >= 0).sum(dim=1).cpu().numpy()
    
    return {
        'total_blocks': total,
        'valid_blocks': valid,
        'sparsity': sparsity,
        'valid_per_row_mean': valid_per_row.mean(),
        'valid_per_row_std': valid_per_row.std(),
        'valid_per_row_min': valid_per_row.min(),
        'valid_per_row_max': valid_per_row.max(),
    }


# ==================== 运行测试 ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


