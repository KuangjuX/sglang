import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils
import torch
import numpy as np
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum


from sglang.srt.sparse_attention.kernels.attention.mask import AttentionMask
from ..test_streaming_attention import construct_streaming_mask



class StreamingMaskTester:
    """Test class for streaming mask functionality following CuteDSL pattern."""
    
    def __init__(
        self,
        m_block_size: int,
        n_block_size: int,
        num_threads: int = 128,
    ):
        # These are now treated as compile-time constants for the kernel
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.num_threads = num_threads
    
    @cute.jit
    def __call__(
        self,
        mOutput: cute.Tensor,
        seqlen_q: cutlass.Int32,
        seqlen_k: cutlass.Int32,
        window_size_left: cutlass.Int32,
        sink_size: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """Launch the streaming mask test kernel."""
        # Do NOT pass m_block_size and n_block_size here.
        # The kernel will access them via `self`.
        self.kernel(
            mOutput,
            seqlen_q,
            seqlen_k,
            window_size_left,
            sink_size,
        ).launch(
            grid=(1, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )
    
    @cute.kernel
    def kernel(
        self,
        mOutput: cute.Tensor,
        seqlen_q: cutlass.Int32,
        seqlen_k: cutlass.Int32,
        window_size_left: cutlass.Int32,
        sink_size: cutlass.Int32,
    ):
        # Access m_block_size and n_block_size via `self`.
        # The JIT compiler treats these as compile-time constants.
        m_block_size = self.m_block_size
        n_block_size = self.n_block_size

        atom_layout_mnk = (1, 1, 1)
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            cutlass.Float16,
            cutlass.Float16,
            LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
            LayoutEnum.ROW_MAJOR.sm90_mma_major_mode(),
            cutlass.Float32,
            atom_layout_mnk,
            tiler_mn=(m_block_size, n_block_size),
        )

        tidx = cute.arch.thread_idx()[0]
        thr_mma = tiled_mma.get_slice(tidx)
        
        m_block = cutlass.Int32(0)
        n_block = cutlass.Int32(0)

        mask = AttentionMask(
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            window_size_left=window_size_left,
            sink_size=sink_size,
            enable_streaming=True
        )

        acc_shape = tiled_mma.partition_shape_C((m_block_size, n_block_size))
        acc_S = cute.make_fragment(acc_shape, cutlass.Float32)
        # cute.fill(acc_S, 0.0)

        mask.apply_streaming_mask(
            acc_S, m_block, n_block, thr_mma, mask_seqlen=True
        )

        # 2. Define the Destination Layout in Global Memory
        gmem_layout_c = cute.make_layout((m_block_size, n_block_size))

        # 3. Create a TiledCopy atom for storing the result
        # This atom knows how to convert from the TiledMma's accumulator layout
        # to the simple global memory layout we just defined.
        tiled_copy_C = sm90_utils.make_tiled_copy_C_atom(
            tiled_mma.tv_layout_C_tiled, gmem_layout_c
        )
        thr_copy_op = tiled_copy_C.get_slice(tidx)

        # 4. Partition Source (Accumulator) and Destination (Global Memory)
        #    using the thread's copy operator.
        
        # This is the "retiling" step. It creates a view of the accumulator
        # with a layout that the copy operation understands.
        thr_S = thr_copy_op.partition_S(acc_S)
        
        # This partitions the global output tensor to get this thread's destination.
        thr_D = thr_copy_op.partition_D(mOutput)

        # 5. Execute the copy
        cute.copy(thr_S, thr_D)



def run_test(
    m_block_size: int,
    n_block_size: int,
    seqlen_q: int,
    seqlen_k: int,
    window_size_left: int,
    sink_size: int,
):
    cache_key = (m_block_size, n_block_size)
    if cache_key not in run_test.tester_cache:
        print(f"Compiling new kernel for block size ({m_block_size}, {n_block_size})...")
        run_test.tester_cache[cache_key] = StreamingMaskTester(
            m_block_size=m_block_size,
            n_block_size=n_block_size,
        )
    
    tester = run_test.tester_cache[cache_key]
    
    output = torch.empty((m_block_size, n_block_size), dtype=torch.float32, device='cuda')
    mOutput = from_dlpack(output.detach(), assumed_align=16)
    
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    
    tester(
        mOutput=mOutput,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        window_size_left=window_size_left,
        sink_size=sink_size,
        stream=current_stream,
    )

    torch.cuda.synchronize()

    result_cpu = output.cpu().numpy()
    result_cpu_sliced = result_cpu[:seqlen_q, :seqlen_k]

    expected_mask_bool = construct_streaming_mask(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        sink_size=sink_size,
        local_size=window_size_left,
        is_causal=True,
        device=torch.device('cpu')
    )

    expected_mask = torch.zeros_like(expected_mask_bool, dtype=torch.float32)
    expected_mask[expected_mask_bool] = -float('inf')
    expected_mask = expected_mask.numpy()

    np.testing.assert_allclose(result_cpu_sliced, expected_mask, atol=1e-6)

    print(f"Test passed for m_block={m_block_size}, n_block={n_block_size}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}!")


# Initialize tester cache
run_test.tester_cache = {}


if __name__ == "__main__":
    # I've also corrected the logic in the dummy `construct_streaming_mask` to better
    # match the typical behavior of streaming attention masks.
    run_test(
        m_block_size=64,
        n_block_size=64,
        seqlen_q=32,
        seqlen_k=32,
        window_size_left=16,
        sink_size=4
    )

    print("\nRunning original test case (sliced to one block)...")
    run_test(
        m_block_size=64,
        n_block_size=64,
        seqlen_q=64,
        seqlen_k=64,
        window_size_left=128, # window > seqlen, so it's fully causal + sink
        sink_size=4
    )