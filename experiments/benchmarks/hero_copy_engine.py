# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare copy-engine all-gather in restored one-rack hero training."""

import dataclasses
import functools
import hashlib
import importlib.metadata
import json
import os
import shlex
from enum import StrEnum
from pathlib import Path

import click
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options

from experiments.benchmarks.hero_profile import _run_restored_local, build_restored_benchmark, run_restored_benchmark
from experiments.grug.moe_hero_ep.hero_recipe import HeroThroughputResult
from experiments.grug.moe_hero_ep.train import GrugRunConfig, TrainingDataMode


class CopyEngineArm(StrEnum):
    CONTROL = "control"
    COPYENGINE = "copyengine"


def _run_copy_engine_restored_local(config: GrugRunConfig, *, arm: CopyEngineArm) -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "benchmarks/copy_engine_protocol.json").read_text())
    production = protocol[arm]
    repository = root.parent
    actual = {path: hashlib.sha256((repository / path).read_bytes()).hexdigest() for path in production["source_sha256"]}
    assert actual == production["source_sha256"], (arm, actual)
    flags = dict(flag.split("=", 1) for flag in shlex.split(os.environ["XLA_FLAGS"]))
    scope = flags["--xla_enable_nccl_symmetric_buffers_for_collectives"]
    assert scope == production["registration_scope"], (arm, scope)
    policy = os.environ.get("NCCL_CTA_POLICY", "DEFAULT")
    assert policy == production["nccl_cta_policy"], (arm, policy)
    click.echo(
        "HERO_COPY_ENGINE_RUNTIME "
        + json.dumps(
            {
                "configured_backward": "native",
                "registration_scope": scope,
                "nccl_cta_policy": policy,
                "production_source_sha": production["commit"],
                "source_sha256": actual,
                "flash_attn_version": importlib.metadata.version("flash-attn-4"),
                "installed_pjrt_version": importlib.metadata.version("jax-cuda13-pjrt"),
                "gpu_package_versions": {
                    name: importlib.metadata.version(name) for name in protocol["runtime_packages"]
                },
                "xla_flags": os.environ.get("XLA_FLAGS", ""),
            }
        )
    )
    _run_restored_local(config)


def build_copy_engine_benchmark(
    *,
    run_id: str,
    checkpoint: tuple[str, ...],
    arm: CopyEngineArm,
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
            local_entrypoint=functools.partial(_run_copy_engine_restored_local, arm=arm),
        ),
    )


@click.command()
@click.option("--run-id", required=True)
@click.option(
    "--checkpoint", multiple=True, required=True, help="Checkpoint root; latest resolves after GPU allocation."
)
@click.option("--arm", type=click.Choice([item.value for item in CopyEngineArm]), required=True)
@click.option("--num-steps", type=click.IntRange(min=1), default=28)
@click.option("--warmup-steps", type=click.IntRange(min=1), default=5)
@click.option("--profile-steps", type=click.IntRange(min=0), default=2)
@click.option("--training-data", type=click.Choice([item.value for item in TrainingDataMode]), default="mixture")
@build_options
def main(
    run_id: str,
    checkpoint: tuple[str, ...],
    arm: str,
    num_steps: int,
    warmup_steps: int,
    profile_steps: int,
    training_data: str,
) -> ArtifactStep[HeroThroughputResult]:
    return build_copy_engine_benchmark(
        run_id=run_id,
        checkpoint=checkpoint,
        arm=CopyEngineArm(arm),
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        profile_steps=profile_steps,
        training_data_mode=TrainingDataMode(training_data),
    )


if __name__ == "__main__":
    main()
