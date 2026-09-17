# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate native SM100 backward on packed-attention gradients before timing it.

Run control and native in fresh processes. These are local, synthetic kernel
checks; a positive result still needs a restored full-rack training comparison.
"""

import argparse
import contextlib
import dataclasses
import importlib.metadata
import json
import os
import signal

import jax
import jax.numpy as jnp
import numpy as np
from levanter.cutlass_kernel_cache import gpu_compute_capability
from levanter.grug.attention import AttentionMask, reference_attention
from levanter.grug.attention import _fa4_cute as fa4_cute

from experiments.benchmarks.fa4.native_sm100 import native_sm100_backward_scope

ATOL = 7e-2
RTOL = 7e-2
HEAD_DIM = 128
Q_HEADS = 48
REPETITIONS = 3
PROCESS_TIMEOUT = 30 * 60


@dataclasses.dataclass(frozen=True)
class CheckCase:
    sequence: int
    kv_heads: int
    window: int | None


CHECK_CASES = tuple(
    CheckCase(sequence, kv_heads, window)
    for kv_heads in (6, 12)
    for sequence, window in ((257, None), (257, 31), (2305, 2048))
)


def _segment_ids(case: CheckCase, iteration: int) -> jax.Array:
    batch = 2 if case.sequence == 257 else 1
    positions = np.arange(case.sequence)
    rows = []
    for row in range(batch):
        # The long final document crosses the 2048-token window boundary.
        boundaries = [101] if case.sequence > 2048 else [31, 129, 193]
        shifted = np.asarray(boundaries) + iteration + row * 7
        ids = np.searchsorted(shifted, positions).astype(np.int32)
        ids[: 19 + iteration] = -1
        ids[-17:] = -1
        if row == 1 and iteration == REPETITIONS - 1:
            ids[:] = -1
        rows.append(ids)
    return jnp.asarray(np.stack(rows))


def _reference(q, k, v, mask):
    output = reference_attention(q, k, v, mask, logits_dtype=jnp.float32)
    assert mask.segment_ids is not None
    # Fully masked reference rows use a finite softmax sentinel. Their output
    # is irrelevant to training; zero it explicitly to match the FA4 contract.
    return jnp.where((mask.segment_ids[0] >= 0)[..., None, None], output, 0)


def _output_and_gradients(attention, q, k, v, cotangent, mask):
    def loss(q_arg, k_arg, v_arg):
        output = attention(q_arg, k_arg, v_arg, mask)
        return jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32)), output

    (_, output), gradients = jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True)(q, k, v)
    return (output, *gradients)


def check_case(case: CheckCase) -> dict[str, object]:
    """Compare outputs and all gradients while reusing each compiled executable."""
    actual_call = jax.jit(lambda *args: _output_and_gradients(fa4_cute.gpu_fa4_cute_attention, *args))
    reference_call = jax.jit(lambda *args: _output_and_gradients(_reference, *args))
    records = []
    for iteration in range(REPETITIONS):
        ids = _segment_ids(case, iteration)
        mask = AttentionMask.causal(sliding_window=case.window).with_segment_ids(ids)
        keys = jax.random.split(jax.random.key(20260916 + iteration), 4)
        query_shape = (*ids.shape, Q_HEADS, HEAD_DIM)
        kv_shape = (*ids.shape, case.kv_heads, HEAD_DIM)
        q, k, v, cotangent = (
            jax.random.normal(key, shape, dtype=jnp.bfloat16)
            for key, shape in zip(keys, (query_shape, kv_shape, kv_shape, query_shape), strict=True)
        )
        # Leave nonzero cotangents on padding: native backward must ignore them.
        args = (q, k, v, cotangent, mask)
        expected = reference_call(*args)
        actual = actual_call(*args)
        jax.block_until_ready((actual, expected))
        metrics = {}
        invalid = np.asarray(ids) < 0
        for name, got_array, want_array in zip(("out", "dq", "dk", "dv"), actual, expected, strict=True):
            got = np.asarray(got_array, dtype=np.float32)
            want = np.asarray(want_array, dtype=np.float32)
            np.testing.assert_allclose(got, want, atol=ATOL, rtol=RTOL, err_msg=f"{case} {iteration} {name}")
            np.testing.assert_array_equal(got[invalid], 0, err_msg=f"{case} {iteration} padded {name}")
            difference = np.abs(got - want)
            metrics[name] = {
                "max_absolute_difference": float(difference.max()),
                "mean_absolute_difference": float(difference.mean()),
            }
        records.append({"iteration": iteration, "metrics": metrics})
    return {**dataclasses.asdict(case), "repetitions": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("control", "native"), required=True)
    parser.add_argument("--source-sha", required=True, help="Clean source snapshot recorded by the submitting process")
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(PROCESS_TIMEOUT)
    if jax.default_backend() != "gpu" or gpu_compute_capability() != 100:
        raise RuntimeError("This benchmark requires a physical SM100 GPU")
    scope = native_sm100_backward_scope() if args.variant == "native" else contextlib.nullcontext()
    with scope:
        checks = [check_case(case) for case in CHECK_CASES]
    print(
        "HERO_NATIVE_SM100 "
        + json.dumps(
            {
                "kernel": "packed_fa4_backward",
                "implementation": args.variant,
                "dtype": "bfloat16",
                "backend": jax.default_backend(),
                "device_type": jax.devices()[0].device_kind,
                "device_count": 1,
                "visible_device_count": len(jax.local_devices()),
                "block_sizes": [128, 128] if args.variant == "native" else [64, 64],
                "git_sha": args.source_sha,
                "flash_attn_version": importlib.metadata.version("flash-attn-4"),
                "xla_flags": os.environ.get("XLA_FLAGS", ""),
                "backend_env": {
                    key: os.environ.get(key)
                    for key in ("JAX_ENABLE_COMPILATION_CACHE", "CUTE_DSL_KEEP", "CUTE_DSL_DUMP_DIR")
                },
                "tolerance": {"atol": ATOL, "rtol": RTOL},
                "checks": checks,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
