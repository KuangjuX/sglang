import cutlass
import cutlass.cute as cute
import torch
import numpy as np

from sglang.srt.sparse_attention.kernels.attention.mask import AttentionMask
from ..test_streaming_attention import construct_streaming_mask

@cutlass.kernel
def streaming_mask_kernel(
    output_ptr: cutlass.Ptr[cutlass.Float32],
    m_block_size: cutlass.Int32,
    n_block_size: cutlass.Int32,
    seqlen_q: cutlass.Int32,
    seqlen_k: cutlass.Int32,
    window_size_left: cutlass.Int32,
    sink_size: cutlass.Int32
):
    thr_mma = cute.TiledMma.make_tiled_mma(
        cute.GMMA.ss_m64n64k16_m16n8k8_f32f16f16f32_tn(),
        cute.Layout.make_layout((128, 128)),
        cute.Layout.make_layout((2, 2))
    )

    m_block = cute.Int<0>()
    n_block = cute.Int<0>()

    mask = AttentionMask(
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        window_size_left=window_size_left,
        sink_size=sink_size,
        enable_streaming=True
    )

    acc_S = cute.make_tensor(
        cute.make_shape(thr_mma.shape_C),
        cute.Layout.make_layout(thr_mma.shape_C),
        dtype=cutlass.Float32
    )
    cute.clear(acc_S)

    mask.apply_streaming_mask(
        acc_S,
        m_block,
        n_block,
        thr_mma,
        mask_seqlen=cute.true_type()
    )

    gmem_layout = cute.make_layout((m_block_size, n_block_size))
    gmem_thr_layout = thr_mma.partition_C(gmem_layout)

    output_tensor = cute.make_tensor(output_ptr, gmem_thr_layout)

    cute.copy(acc_S, output_tensor)

def run_test(
    M_BLOCK_SIZE: int,
    N_BLOCK_SIZE: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    WINDOW_SIZE_LEFT: int,
    SINK_SIZE: int,
):

    output = torch.empty((M_BLOCK_SIZE, N_BLOCK_SIZE), dtype=torch.float32, device='cuda')

    grid = (1, 1, 1)
    block = (128, 1, 1)

    params = streaming_mask_kernel.Params(
        output_ptr=output.data_ptr(),
        m_block_size=M_BLOCK_SIZE,
        n_block_size=N_BLOCK_SIZE,
        seqlen_q=SEQLEN_Q,
        seqlen_k=SEQLEN_K,
        window_size_left=WINDOW_SIZE_LEFT,
        sink_size=SINK_SIZE
    )
    
    streaming_mask_kernel.launch(grid, block, params)
    torch.cuda.synchronize()

    result_cpu = output.cpu().numpy()


    expected_mask_bool = construct_streaming_mask(
        seqlen_q=SEQLEN_Q,
        seqlen_k=SEQLEN_K,
        sink_size=SINK_SIZE,
        local_size=WINDOW_SIZE_LEFT,
        is_causal=True,
        device=torch.device('cuda')
    )

    expected_mask = torch.zeros_like(expected_mask_bool, dtype=torch.float32)
    expected_mask[expected_mask_bool] = -float('inf')

    np.testing.assert_allclose(result_cpu, expected_mask, atol=1e-6)

    print("Test passed!")

if __name__ == "__main__":
    run_test(
        M_BLOCK_SIZE=64,
        N_BLOCK_SIZE=64,
        SEQLEN_Q=512,
        SEQLEN_K=512,
        WINDOW_SIZE_LEFT=128,
        SINK_SIZE=4
    )
