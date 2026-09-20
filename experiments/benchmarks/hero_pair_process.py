# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run two frozen source trees in fresh processes within one GPU task allocation."""

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class ArmSource:
    name: str
    commit: str
    overrides: dict[str, str]
    expected_sha256: dict[str, str]


@dataclasses.dataclass(frozen=True)
class PairSources:
    common_sha256: dict[str, str]
    python_roots: tuple[str, ...]
    arms: tuple[ArmSource, ArmSource]
    timeout: int


def prepare_source(repository: Path, destination: Path, sources: PairSources, arm: ArmSource) -> None:
    """Copy the declared common source and apply the arm's exact source files."""
    for name, expected in sources.common_sha256.items():
        source = repository / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Common source changed: {name}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name, content in arm.overrides.items():
        if name not in sources.common_sha256:
            raise ValueError(f"Override is outside the frozen tree: {name}")
        (destination / name).write_text(content)
    for name, expected in arm.expected_sha256.items():
        if hashlib.sha256((destination / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Arm source changed: {arm.name}: {name}")


def run_pair_sources(
    repository: Path,
    sources: PairSources,
    order: tuple[str, str],
    command: tuple[str, ...],
    arguments: dict[str, dict],
    output: Path,
) -> None:
    """Run each arm once, aborting the pair if either process fails."""
    arms = {arm.name: arm for arm in sources.arms}
    if len(arms) != 2 or len(order) != 2 or set(order) != set(arms):
        raise ValueError("The pair must contain each frozen arm exactly once")
    deadline = time.monotonic() + sources.timeout
    rank = os.environ.get("IRIS_MULTIGPU_PROCESS_INDEX", "0")
    for position, name in enumerate(order):
        arm = arms[name]
        with tempfile.TemporaryDirectory(prefix=f"hero-pair-{name}-rank{rank}-") as temporary:
            source_root = Path(temporary) / "source"
            prepare_source(repository, source_root, sources, arm)
            arm_output = output / name / f"rank-{rank}"
            arm_output.mkdir(parents=True, exist_ok=True)
            argument_path = Path(temporary) / "arguments.json"
            argument_path.write_text(json.dumps(arguments[name]))
            environment = dict(os.environ)
            environment.update(
                PYTHONPATH=os.pathsep.join(str(source_root / root) for root in sources.python_roots),
                IRIS_OUTPUT_DIR=str(arm_output),
                # Each subprocess gets a distinct endpoint name and port. A fast rank must not
                # discover the preceding arm's coordinator while its process is shutting down.
                IRIS_PORT_JAX=str(32980 + position),
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("The paired GPU execution exceeded its aggregate deadline")
            print(
                "HERO_PAIR_PROCESS "
                + json.dumps({"arm": name, "commit": arm.commit, "rank": rank, "phase": "starting"}),
                flush=True,
            )
            subprocess.run(
                [sys.executable, *command, str(argument_path)],
                cwd=source_root,
                env=environment,
                timeout=remaining,
                check=True,
            )
            print(
                "HERO_PAIR_PROCESS "
                + json.dumps({"arm": name, "commit": arm.commit, "rank": rank, "phase": "finished"}),
                flush=True,
            )


def run_training_pair(sources: PairSources, order: tuple[str, str], arguments: dict[str, dict]) -> None:
    repository = Path(__file__).resolve().parents[2]
    run_pair_sources(
        repository,
        sources,
        order,
        ("-m", "experiments.benchmarks.hero_pair_arm"),
        arguments,
        Path(os.environ["IRIS_OUTPUT_DIR"]),
    )
