# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare routed-expert masks in restored one-rack hero training."""

import dataclasses
import functools
import hashlib
import importlib.metadata
import json
import os
import shlex
from collections import Counter
from enum import StrEnum
from pathlib import Path

import click
import jax
from levanter.utils.jax_utils import multihost_broadcast_sync
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options
from marin.profiling.trace_summary import BreakdownMode, TraceEventTrack
from marin.profiling.xplane import find_xplane_file, parse_xplane_timeline

from experiments.benchmarks.hero_profile import _run_restored_local, build_restored_benchmark, run_restored_benchmark
from experiments.grug.moe_hero_ep.hero_recipe import HeroThroughputResult
from experiments.grug.moe_hero_ep.train import GrugRunConfig, TrainingDataMode


class BenchmarkArm(StrEnum):
    CONTROL = "control"
    CANDIDATE = "candidate"


def _observed_kernels(profile: Path) -> dict[str, int]:
    timeline = parse_xplane_timeline(profile, breakdown_mode=BreakdownMode.EXCLUSIVE_GLOBAL)
    counts = Counter()
    for track in timeline.tracks:
        if not isinstance(track, TraceEventTrack):
            continue
        if track.process_name != "/device:GPU:0" or not (track.thread_name or "").startswith("Stream"):
            continue
        for event in track.events:
            name = event.name.lower()
            for family in (
                "flashattentionforwardsm100",
                "segmented_flash_attention_forward",
                "flashattentionbackwardsm100",
                "_zero_grouped_output_tail",
            ):
                if family in name:
                    counts[family] += 1
    return dict(counts)


def _run_paired_restored_local(config: GrugRunConfig, *, arm: BenchmarkArm) -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "benchmarks/hero_path_training_protocol.json").read_text())
    production = protocol[arm]
    assert config.model.attention_implementation == "gpu_fa4_cute_wide"
    assert config.model.num_layers == 48
    assert config.stop_after_steps == protocol["steps_per_arm"]
    assert config.trainer.trainer.profiler.num_steps == 2
    assert config.trainer.trainer.profiler.start_step == protocol["profile_relative_steps"][0]
    assert config.trainer.trainer.profiler.process_index == 0
    repository = root.parent
    actual = {path: hashlib.sha256((repository / path).read_bytes()).hexdigest() for path in production["source_sha256"]}
    assert actual == production["source_sha256"], (arm, actual)
    flags = dict(flag.split("=", 1) for flag in shlex.split(os.environ["XLA_FLAGS"]))
    scope = flags["--xla_enable_nccl_symmetric_buffers_for_collectives"]
    assert scope == production["registration_scope"], (arm, scope)
    policy = os.environ.get("NCCL_CTA_POLICY", "DEFAULT")
    assert policy == production["nccl_cta_policy"], (arm, policy)
    versions = {name: importlib.metadata.version(name) for name in protocol["packages"]}
    assert versions == protocol["packages"], versions
    click.echo(
        "HERO_MASK_RUNTIME "
        + json.dumps(
            {
                "intended_backward": "native",
                "protocol_sha256": (
                    hashlib.sha256((root / "benchmarks/hero_path_training_protocol.json").read_bytes()).hexdigest()
                ),
                "intended_forward": production["intended_forward"],
                "attention_implementation": config.model.attention_implementation,
                "registration_scope": scope,
                "nccl_cta_policy": policy,
                "production_source_sha": production["commit"],
                "source_sha256": actual,
                "flash_attn_version": importlib.metadata.version("flash-attn-4"),
                "installed_pjrt_version": importlib.metadata.version("jax-cuda13-pjrt"),
                "gpu_package_versions": versions,
                "xla_flags": os.environ.get("XLA_FLAGS", ""),
            }
        )
    )
    _run_restored_local(config)
    # Inspect the captured training profile after timing; never initialize JAX before the distributed entrypoint.
    observed = {}
    if jax.process_index() == 0:
        trainer = config.trainer.trainer
        assert trainer.id is not None
        profile = find_xplane_file(Path(trainer.log_dir) / trainer.id / "profiler")
        observed = _observed_kernels(profile)
        click.echo("HERO_MASK_OBSERVED_KERNELS " + json.dumps({"arm": arm, "counts": observed}))
    observed = multihost_broadcast_sync(observed)
    expected = production["expected_profile_kernel_counts"]
    tail_count = observed.pop("_zero_grouped_output_tail", 0)
    assert observed == expected, (arm, observed, expected)
    if arm == BenchmarkArm.CONTROL:
        assert tail_count == 0, tail_count
    else:
        assert tail_count > 0, "The candidate did not execute the tail-clearing kernel"


def build_paired_benchmark(
    *,
    run_id: str,
    checkpoint: tuple[str, ...],
    arm: BenchmarkArm,
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
            local_entrypoint=functools.partial(_run_paired_restored_local, arm=arm),
        ),
    )


@click.command()
@click.option("--run-id", required=True)
@click.option(
    "--checkpoint", multiple=True, required=True, help="Checkpoint root; latest resolves after GPU allocation."
)
@click.option("--arm", type=click.Choice([item.value for item in BenchmarkArm]), required=True)
@click.option("--num-steps", type=click.IntRange(min=1), default=64)
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
    return build_paired_benchmark(
        run_id=run_id,
        checkpoint=checkpoint,
        arm=BenchmarkArm(arm),
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        profile_steps=profile_steps,
        training_data_mode=TrainingDataMode(training_data),
    )


if __name__ == "__main__":
    main()
