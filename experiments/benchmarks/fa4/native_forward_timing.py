# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Screen native SM100 forward at hero shapes after the reference correctness gate."""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
from levanter.cutlass_kernel_cache import gpu_compute_capability
from levanter.grug.attention import _fa4_cute as fa4_cute
from levanter.grug.attention._fa4_cute_backend import fa4_cute_attention_forward
from levanter.grug.attention._fa4_cute_config import Flash4CuteKernelConfig

from experiments.benchmarks.fa4.tile_sweep import _seconds_per_step, _segment_ids


def _loss(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cotangent: jax.Array,
    lower: jax.Array,
    valid: jax.Array,
    *,
    kernel_config: Flash4CuteKernelConfig,
) -> jax.Array:
    output = fa4_cute_attention_forward(q, k, v, lower, valid, sm_scale=q.shape[-1] ** -0.5, kernel_config=kernel_config)
    return jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("control", "native"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    root = Path(__file__).resolve().parents[3]
    for name, expected in protocol["production_source_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    versions = {name: importlib.metadata.version(name) for name in protocol["packages"]}
    assert versions == protocol["packages"], versions
    assert jax.default_backend() == "gpu" and gpu_compute_capability() == 100
    assert jax.device_count() == jax.local_device_count() == 1
    config = fa4_cute._segmented_kernel_config(128)
    assert config.sm100_backward is not None and config.sm100_forward is not None
    if args.variant == "control":
        config = dataclasses.replace(config, sm100_forward=None)
    results = []
    for shape in protocol["shapes"]:
        batch, sequence, q_heads, kv_heads, head_dim = (
            shape[name] for name in ("batch", "sequence", "query_heads", "kv_heads", "head_dim")
        )
        keys = jax.random.split(jax.random.key(protocol["seed"]), 4)
        q_shape = (batch, sequence, q_heads, head_dim)
        kv_shape = (batch, sequence, kv_heads, head_dim)
        q, k, v, cotangent = (
            jax.random.normal(key, size, dtype=jnp.bfloat16)
            for key, size in zip(keys, (q_shape, kv_shape, kv_shape, q_shape), strict=True)
        )
        ids = _segment_ids(batch, sequence, protocol["documents_per_sequence"])
        lower, valid = fa4_cute._packed_segment_causal_lower_bounds(
            ids, batch_size=batch, seq_len=sequence, sliding_window=shape["window"]
        )

        forward_call = jax.jit(partial(fa4_cute_attention_forward, sm_scale=head_dim**-0.5, kernel_config=config))
        full_call = jax.jit(jax.value_and_grad(partial(_loss, kernel_config=config), argnums=(0, 1, 2)))
        forward_args = (q, k, v, lower, valid)
        full_args = (q, k, v, cotangent, lower, valid)
        outputs = (forward_call(*forward_args), full_call(*full_args))
        assert all(bool(jnp.all(jnp.isfinite(value))) for value in jax.tree.leaves(outputs))
        samples = {}
        for name, call in (
            ("forward", partial(forward_call, *forward_args)),
            ("forward_backward", partial(full_call, *full_args)),
        ):
            samples[name] = [
                _seconds_per_step(call, steps=protocol["steps_per_batch"], warmup=protocol["warmup_steps"])
                for _ in range(protocol["timing_batches"])
            ]
        results.append({"shape": shape, "seconds_per_call": samples})
    result = {
        "variant": args.variant,
        "production_source": protocol["production_source"],
        "production_source_sha256": protocol["production_source_sha256"],
        "packages": versions,
        "device": jax.devices()[0].device_kind,
        "config": dataclasses.asdict(config),
        "results": results,
        "scope": "Single-GPU exploratory timing batches; not independent training replications",
    }
    (Path(os.environ["IRIS_OUTPUT_DIR"]) / f"{args.variant}-timing.json").write_text(json.dumps(result, indent=2) + "\n")
    print("HERO_NATIVE_FORWARD_TIMING " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
