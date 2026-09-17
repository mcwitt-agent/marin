# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare native SM100 attention backward in restored one-rack hero training."""

import contextlib
import dataclasses
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
from enum import StrEnum
from pathlib import Path

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options

from experiments.benchmarks.fa4.native_sm100 import (
    PINNED_FA4_VERSION,
    native_sm100_backward,
    native_sm100_backward_scope,
)
from experiments.benchmarks.hero_profile import _run_restored_local, build_restored_benchmark, run_restored_benchmark
from experiments.grug.moe_hero_ep.hero_recipe import HeroThroughputResult
from experiments.grug.moe_hero_ep.train import GrugRunConfig, TrainingDataMode


class BackwardImplementation(StrEnum):
    CONTROL = "control"
    NATIVE = "native"


def _run_attention_restored_local(config: GrugRunConfig, *, backward: BackwardImplementation) -> None:
    version = importlib.metadata.version("flash-attn-4")
    if version != PINNED_FA4_VERSION:
        raise RuntimeError(f"Attention comparison requires FA4 {PINNED_FA4_VERSION}, got {version}")
    source_file = inspect.getsourcefile(native_sm100_backward)
    assert source_file is not None
    click.echo(
        "HERO_ATTENTION_RUNTIME "
        + json.dumps(
            {
                "configured_backward": backward,
                "native_source_sha256": hashlib.sha256(Path(source_file).read_bytes()).hexdigest(),
                "flash_attn_version": version,
                "installed_pjrt_version": importlib.metadata.version("jax-cuda13-pjrt"),
                "xla_flags": os.environ.get("XLA_FLAGS", ""),
            }
        )
    )
    scope = native_sm100_backward_scope() if backward == BackwardImplementation.NATIVE else contextlib.nullcontext()
    with scope:
        _run_restored_local(config)


def build_native_attention_benchmark(
    *,
    run_id: str,
    checkpoint: str,
    backward: BackwardImplementation,
    num_steps: int,
    warmup_steps: int,
    profile_steps: int,
    training_data_mode: TrainingDataMode,
    version: str | None = None,
) -> ArtifactStep[HeroThroughputResult]:
    """Build a paired-comparison arm using the repository's locked GPU runtime."""
    step = build_restored_benchmark(
        run_id=run_id,
        checkpoint=checkpoint,
        num_steps=num_steps,
        profile_steps=profile_steps,
        warmup_steps=warmup_steps,
        training_data_mode=training_data_mode,
        version=version,
    )
    return dataclasses.replace(
        step,
        run=functools.partial(
            run_restored_benchmark,
            local_entrypoint=functools.partial(_run_attention_restored_local, backward=backward),
        ),
    )


@click.command()
@click.option("--run-id", required=True)
@click.option("--checkpoint", required=True, help="Checkpoint root; latest resolves after GPU allocation.")
@click.option("--backward", type=click.Choice([item.value for item in BackwardImplementation]), required=True)
@click.option("--num-steps", type=click.IntRange(min=1), default=28)
@click.option("--warmup-steps", type=click.IntRange(min=1), default=5)
@click.option("--profile-steps", type=click.IntRange(min=0), default=2)
@click.option("--training-data", type=click.Choice([item.value for item in TrainingDataMode]), default="mixture")
@build_options
def main(
    run_id: str,
    checkpoint: str,
    backward: str,
    num_steps: int,
    warmup_steps: int,
    profile_steps: int,
    training_data: str,
) -> ArtifactStep[HeroThroughputResult]:
    return build_native_attention_benchmark(
        run_id=run_id,
        checkpoint=checkpoint,
        backward=BackwardImplementation(backward),
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        profile_steps=profile_steps,
        training_data_mode=TrainingDataMode(training_data),
    )


if __name__ == "__main__":
    main()
