# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Screen all-gather symmetric registration before a restored-training comparison."""

import importlib.metadata
import json
import os
import re
import shlex
import signal
from pathlib import Path

import click
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from marin.testing.moe.ragged_ep import RaggedEpConfig, run_benchmark

from experiments.grug.moe_hero_ep import train as hero_train

DEVICE_COUNT = 4
LOCAL_ROWS = 256
WIDTH = 128
ITERATIONS = 10
CHECK_TIMEOUT = 5 * 60
SCOPE_FLAG = "--xla_enable_nccl_symmetric_buffers_for_collectives"
PJRT_VERSION = "0.11.1+marin.708c3a4ec79c"


def check_all_gathers() -> dict:
    """Compare repeated donated gathers and their transposes with NumPy values."""
    devices = jax.devices()
    if len(devices) != DEVICE_COUNT or jax.process_count() != 1:
        raise ValueError("This gate requires one process with four addressable devices")
    mesh = Mesh(np.asarray(devices), ("rank",), axis_types=(AxisType.Explicit,))
    sharding = NamedSharding(mesh, P("rank", None))
    replicated = NamedSharding(mesh, P())

    @jax.shard_map(
        mesh=mesh,
        in_specs=(P("rank", None), P("rank", None)),
        out_specs=(P("rank", None), P("rank", None), P(), P(), P("rank", None), P("rank", None), P(), P()),
        check_vma=False,
    )
    def update(bf16, fp32):
        def loss(value):
            gathered = jax.lax.all_gather(value, "rank", axis=0, tiled=True)
            # Constrain conversion hoisting; verify actual collective dtypes in GPU HLO below.
            gathered = jax.lax.optimization_barrier(gathered)
            weights = (1 + jnp.arange(gathered.shape[0]) % 3)[:, None]
            return jnp.sum(gathered.astype(jnp.float32) * weights), gathered

        (loss_bf16, gathered_bf16), grad_bf16 = jax.value_and_grad(loss, has_aux=True)(bf16)
        (loss_fp32, gathered_fp32), grad_fp32 = jax.value_and_grad(loss, has_aux=True)(fp32)
        return bf16 + 1, fp32 + 1, gathered_bf16, gathered_fp32, grad_bf16, grad_fp32, loss_bf16, loss_fp32

    rank_values = 16 * np.repeat(np.arange(DEVICE_COUNT, dtype=np.float32), LOCAL_ROWS)[:, None]
    within_rank = np.random.default_rng(0).integers(0, 16, size=(DEVICE_COUNT * LOCAL_ROWS, WIDTH))
    expected = rank_values + within_rank.astype(np.float32)
    weights = np.broadcast_to((1 + np.arange(expected.shape[0]) % 3)[:, None], expected.shape)
    expected_gradient = DEVICE_COUNT * weights
    bf16 = jax.device_put(expected.astype(jnp.bfloat16), sharding)
    fp32 = jax.device_put(expected, sharding)
    outputs = (sharding, sharding, replicated, replicated, sharding, sharding, replicated, replicated)
    with jax.set_mesh(mesh):
        compiled = (
            jax.jit(update, in_shardings=(sharding, sharding), out_shardings=outputs, donate_argnums=(0, 1))
            .lower(bf16, fp32)
            .compile()
        )
        stats = compiled.memory_analysis()
        assert stats is not None
        plan = {
            name: int(getattr(stats, name))
            for name in ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")
        }
        if plan["alias_size_in_bytes"] == 0:
            raise AssertionError("The repeated-execution gate must exercise donated buffers")
        hlo = compiled.as_text()
        if "all-gather" not in hlo:
            raise AssertionError("The compiled gate contains no all-gather")
        collective_lines = [line for line in hlo.splitlines() if re.search(r"\ball-gather(?:-start)?\(", line)]
        collective_dtypes = [
            dtype
            for dtype in ("bf16", "f32")
            if any(f"{dtype}[" in line.partition(" = ")[2].partition(" all-gather")[0] for line in collective_lines)
        ]
        if devices[0].platform == "gpu" and collective_dtypes != ["bf16", "f32"]:
            raise AssertionError(f"GPU gate must retain BF16 and FP32 collectives; found {collective_dtypes}")
        for iteration in range(ITERATIONS):
            bf16, fp32, gathered_bf16, gathered_fp32, grad_bf16, grad_fp32, loss_bf16, loss_fp32 = compiled(bf16, fp32)
            for gathered in (gathered_bf16, gathered_fp32):
                np.testing.assert_array_equal(np.asarray(gathered, dtype=np.float32), expected + iteration)
            for gradient in (grad_bf16, grad_fp32):
                np.testing.assert_array_equal(np.asarray(gradient, dtype=np.float32), expected_gradient)
            for loss in (loss_bf16, loss_fp32):
                np.testing.assert_array_equal(np.asarray(loss), np.sum((expected + iteration) * weights))
            if bf16.sharding != sharding or fp32.sharding != sharding:
                raise AssertionError("Donated state changed sharding")
        np.testing.assert_array_equal(np.asarray(bf16, dtype=np.float32), expected + ITERATIONS)
        np.testing.assert_array_equal(np.asarray(fp32), expected + ITERATIONS)
    return {
        "devices": DEVICE_COUNT,
        "iterations": ITERATIONS,
        "memory_plan": plan,
        "collective_dtypes": collective_dtypes,
        "collective_hlo": collective_lines,
        "hlo": hlo,
    }


@click.command()
@click.option("--scope", type=click.Choice(["raggedalltoall", "raggedalltoall,allgather"]), required=True)
@click.option("--check", type=click.Choice(["collectives", "ragged"]), required=True)
def main(scope: str, check: str) -> None:
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(CHECK_TIMEOUT)
    installed = importlib.metadata.version("jax-cuda13-pjrt")
    assert installed == PJRT_VERSION, installed
    hero_train._apply_hero_ep_runtime_defaults(
        inline_watch_enabled=False,
        moe_implementation=hero_train.RAGGED_MOE_IMPLEMENTATION,
        remat_mode=hero_train.OFFLOAD_CARRY_REMAT_MODE,
        processes_per_task=4,
    )
    # Select the synthetic gate arm before initializing JAX; training uses its source's defaults.
    flags = dict(flag.split("=", 1) for flag in shlex.split(os.environ["XLA_FLAGS"]))
    flags[SCOPE_FLAG] = scope
    assert flags[hero_train.XLA_COLLECTIVE_OVERLAP_FLAG] == "1"
    os.environ["XLA_FLAGS"] = shlex.join(f"{name}={value}" for name, value in flags.items())
    if len(jax.devices()) != DEVICE_COUNT or any(device.platform != "gpu" for device in jax.devices()):
        raise ValueError("The remote gate requires four GPUs")
    output = Path(os.environ["IRIS_OUTPUT_DIR"]) / check
    output.mkdir(parents=True, exist_ok=True)
    if check == "ragged":
        run_benchmark(RaggedEpConfig(output_path=str(output)))
        result = json.loads((output / "results.json").read_text())
    else:
        result = check_all_gathers()
        (output / "compiled-hlo.txt").write_text(result.pop("hlo"))
    result.update(scope=scope, check=check, pjrt_version=installed, xla_flags=os.environ["XLA_FLAGS"])
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    click.echo("HERO_SYMMETRIC_CHECK " + json.dumps(result))
    signal.alarm(0)


if __name__ == "__main__":
    main()
