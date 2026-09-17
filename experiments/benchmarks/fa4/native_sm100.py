# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in native SM100 packed backward integration for hero benchmarks.

Use a fresh process per backend. This factory has a separate persistent cache
identity from production, and does not change production architecture selection.
"""

import importlib
import importlib.metadata
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import jax
import jax.numpy as jnp
from levanter.cutlass_kernel_cache import cute_launcher_factory, cutlass_call, gpu_compute_capability
from levanter.grug.attention import _fa4_cute_backend as backend
from levanter.grug.attention import _fa4_cute_kernels as kernels
from levanter.grug.attention._fa4_cute_config import Flash4CuteKernelConfig

HEAD_DIM = 128
TILE_M = 128
TILE_N = 128
PINNED_FA4_VERSION = "4.0.0b28"


@cute_launcher_factory
def native_sm100_launcher(
    modules: backend._CutlassCuteModules, *, head_dim: int, head_dim_v: int, qhead_per_kvhead: int
) -> Any:
    deps = kernels._import_cute_dependencies(modules)
    cutlass, cute, cuda = deps.cutlass, deps.cute, deps.cuda
    native_module = importlib.import_module("flash_attn.cute.flash_bwd_sm100")
    sparsity_module = importlib.import_module("flash_attn.cute.block_sparsity")
    utils_module = importlib.import_module("flash_attn.cute.utils")
    FlashAttentionBackwardSm100 = native_module.FlashAttentionBackwardSm100
    BlockSparseTensors = sparsity_module.BlockSparseTensors
    kernels._patch_jax_array_list_tvm_ffi_converter()
    AuxData = utils_module.AuxData

    @cute.jit
    def _grug_segment_mask_mod(
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        q_idx: cutlass.Int32,
        kv_idx: cutlass.Int32,
        seqlen_info: Any,
        aux_tensors: Any,
    ) -> Any:
        del head_idx, seqlen_info
        batch_idx = utils_module.ssa_to_scalar(batch_idx)
        q_idx = utils_module.ssa_to_scalar(q_idx)
        kv_idx = utils_module.ssa_to_scalar(kv_idx)
        lower_bounds, valid = aux_tensors
        query_in_bounds = cute.elem_less(q_idx, lower_bounds.shape[1])
        metadata_q_idx = q_idx if query_in_bounds else lower_bounds.shape[1] - 1
        query_valid = valid[batch_idx, metadata_q_idx] != 0
        query_lower_bound = lower_bounds[batch_idx, metadata_q_idx]
        key_after_lower_bound = cute.elem_less(query_lower_bound, kv_idx + 1)
        key_before_query = cute.elem_less(kv_idx, q_idx + 1)
        mask_value = query_in_bounds and query_valid and key_after_lower_bound and key_before_query
        return utils_module.scalar_to_ssa(mask_value, cutlass.Boolean)

    backward = FlashAttentionBackwardSm100(
        head_dim,
        head_dim_v,
        is_causal=False,
        is_local=False,
        qhead_per_kvhead=qhead_per_kvhead,
        tile_m=TILE_M,
        tile_n=TILE_N,
        cluster_size=1,
        use_2cta_instrs=False,
        deterministic=False,
        mask_mod=_grug_segment_mask_mod,
        has_aux_tensors=True,
    )
    if backward.use_2cta_instrs:
        raise RuntimeError("Packed SM100 benchmark must use one CTA")

    class _Float32ZeroFill:
        def __init__(self, num_threads: int):
            self._num_threads = num_threads

        @cute.jit
        def __call__(self, tensor: cute.Tensor, stream: cuda.CUstream):
            self.kernel(tensor).launch(
                grid=[cute.ceil_div(cute.size(tensor), self._num_threads), 1, 1],
                block=[self._num_threads, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(self, tensor: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            flat = cute.make_tensor(tensor.iterator, cute.make_layout(cute.size(tensor)))
            idx = bidx * self._num_threads + tidx
            if idx < cute.size(flat):
                flat[idx] = cutlass.Float32(0.0)

    zero_fill = _Float32ZeroFill(512)

    @cute.jit
    def _as_gmem_tensor(tensor: cute.Tensor) -> cute.Tensor:
        ptr = cute.make_ptr(
            tensor.element_type,
            tensor.iterator.toint(),
            cute.AddressSpace.gmem,
            assumed_align=256,
        )
        return cute.make_tensor(ptr, tensor.layout)

    @cute.jit
    def _launch_native_sm100(
        stream: cuda.CUstream,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        dout: cute.Tensor,
        lse_log2: cute.Tensor,
        dpsum: cute.Tensor,
        lower_bounds: cute.Tensor,
        valid: cute.Tensor,
        mask_block_cnt: cute.Tensor,
        mask_block_idx: cute.Tensor,
        full_block_cnt: cute.Tensor,
        full_block_idx: cute.Tensor,
        dq_accum: cute.Tensor,
        dk_accum: cute.Tensor,
        dv_accum: cute.Tensor,
        *,
        softmax_scale: cutlass.Float32,
    ):
        blocksparse_tensors = BlockSparseTensors(mask_block_cnt, mask_block_idx, full_block_cnt, full_block_idx)
        if cutlass.const_expr(qhead_per_kvhead > 1):
            zero_fill(dq_accum, stream)
            zero_fill(dk_accum, stream)
            zero_fill(dv_accum, stream)
        lse_log2_gmem = _as_gmem_tensor(lse_log2)
        dpsum_gmem = _as_gmem_tensor(dpsum)
        dq_accum_gmem = _as_gmem_tensor(dq_accum)
        dk_accum_gmem = _as_gmem_tensor(dk_accum)
        dv_accum_gmem = _as_gmem_tensor(dv_accum)
        backward(
            q,
            k,
            v,
            dout,
            lse_log2_gmem,
            dpsum_gmem,
            dq_accum_gmem,
            dk_accum_gmem,
            dv_accum_gmem,
            softmax_scale,
            aux_data=AuxData(tensors=(lower_bounds, valid)),
            blocksparse_tensors=blocksparse_tensors,
            stream=stream,
        )

    return _launch_native_sm100


def native_sm100_backward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    out: jax.Array,
    dout: jax.Array,
    lse: jax.Array,
    lower_bounds: jax.Array,
    valid: jax.Array,
    *,
    softmax_scale: float,
    kernel_config: Flash4CuteKernelConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Match the segmented backend contract using native one-CTA SM100."""
    del kernel_config
    backend._validate_forward_inputs(q, k, v, lower_bounds, valid, softmax_scale=softmax_scale)
    backend._validate_backward_inputs(q, k, v, out, dout, lse)
    if q.dtype != jnp.bfloat16 or q.shape[-1] != HEAD_DIM or v.shape[-1] != HEAD_DIM:
        raise NotImplementedError("Native SM100 benchmark requires BF16 D128 Q/K/V")
    ratio = q.shape[2] // k.shape[2]
    if ratio not in (4, 8):
        raise NotImplementedError("Native SM100 benchmark supports only hero GQA ratios 4 and 8")
    if gpu_compute_capability() // 10 != 10:
        raise NotImplementedError("Native SM100 benchmark requires physical SM100-family hardware")
    version = importlib.metadata.version("flash-attn-4")
    if version != PINNED_FA4_VERSION:
        raise RuntimeError(f"Native SM100 benchmark requires FA4 {PINNED_FA4_VERSION}, got {version}")
    modules = backend._import_cutlass_cute()
    tile = (TILE_M, TILE_N)
    sparse = backend._packed_segment_backward_block_sparse_indices_with_full(
        lower_bounds, valid, tile_m=TILE_M, tile_n=TILE_N
    )
    partial_count, partial_index = backend._broadcast_backward_block_sparse_metadata(
        q, sparse.partial_block_cnt, sparse.partial_block_idx
    )
    full_count, full_index = backend._broadcast_backward_block_sparse_metadata(
        q, sparse.full_block_cnt, sparse.full_block_idx
    )
    preprocess_inputs, preprocess_outputs = backend._cutlass_attention_backward_sm90_preprocess_specs(
        modules, vector_elems=8
    )
    preprocess = cutlass_call(
        kernels.segmented_flash_attention_backward_sm90_preprocess_launcher(
            modules, dtype=q.dtype, head_dim=HEAD_DIM, head_dim_v=HEAD_DIM, tile_m=TILE_M
        ),
        output_shape_dtype=backend._cutlass_attention_backward_sm90_preprocess_output_shapes(q, tile),
        input_spec=preprocess_inputs,
        output_spec=preprocess_outputs,
        use_static_tensors=True,
        softmax_scale=softmax_scale,
    )
    dpsum, lse_log2, _ = preprocess(out, dout, lse)
    accum_inputs, accum_outputs = backend._cutlass_attention_backward_sm90_accum_specs(modules, vector_elems=8)
    backward = cutlass_call(
        native_sm100_launcher(modules, head_dim=HEAD_DIM, head_dim_v=HEAD_DIM, qhead_per_kvhead=ratio),
        output_shape_dtype=backend._cutlass_attention_backward_sm90_backward_output_shapes(q, k, v, tile),
        input_spec=accum_inputs,
        output_spec=accum_outputs,
        use_static_tensors=True,
        softmax_scale=softmax_scale,
    )
    accumulators = backward(
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        lower_bounds,
        valid.astype(jnp.int32),
        partial_count,
        partial_index,
        full_count,
        full_index,
    )
    post_inputs, post_outputs = backend._cutlass_attention_backward_sm90_postprocess_specs(modules, vector_elems=8)
    gradients = []
    for tensor, accum, scale in zip((q, k, v), accumulators, (softmax_scale, softmax_scale, 1.0), strict=True):
        postprocess = cutlass_call(
            kernels.flash_attention_backward_postprocess_launcher(
                modules,
                dtype=tensor.dtype,
                head_dim=HEAD_DIM,
                tile_m=TILE_M,
                atom_layout_m=1,
                arch=100,
                num_threads=128,
                cluster_size=1,
                use_2cta_instrs=False,
                accum_is_gmem=True,
            ),
            output_shape_dtype=(jax.ShapeDtypeStruct(tensor.shape, tensor.dtype),),
            input_spec=post_inputs,
            output_spec=post_outputs,
            use_static_tensors=True,
            softmax_scale=scale,
        )
        gradients.append(postprocess(accum)[0])
    return gradients[0], gradients[1], gradients[2]


@contextmanager
def native_sm100_backward_scope() -> Iterator[None]:
    """Select native backward while tracing a benchmark; production is unchanged."""
    with patch.object(backend, "segmented_flash_attention_backward", native_sm100_backward):
        yield
