# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit a paired restored hero comparison in one physical GPU allocation."""

import argparse
import base64
import json
from pathlib import Path

import cloudpickle
import draccus
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from marin.execution.lazy import StepContext
from marin.training.run_environment import extras_for_resources
from marin.training.training import resolve_training_env
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import prefix_join

from experiments.benchmarks.hero_pair_process import ArmSource, PairSources, run_training_pair
from experiments.benchmarks.hero_path_training import BenchmarkArm, build_paired_benchmark
from experiments.grug.dispatch import _forwarded_env_vars
from experiments.grug.moe_hero_ep.train import TrainingDataMode, WatchMode, _apply_hero_ep_runtime_defaults


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--order", choices=["control,candidate", "candidate,control"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if not plan["ready_for_execution"] and not args.dry_run:
        raise ValueError("Freeze and validate the paired protocol before execution")
    first, second = args.order.split(",")
    order = (first, second)
    arm_records = [ArmSource(**arm) for arm in plan["arms"]]
    assert len(arm_records) == 2
    sources = PairSources(
        common_sha256=plan["common_sha256"],
        python_roots=tuple(plan["python_roots"]),
        arms=(arm_records[0], arm_records[1]),
        timeout=plan["protocol"]["training_timeout_seconds"],
    )
    arguments = {}
    configs = []
    for arm in sources.arms:
        run_id = args.run_id + "-" + arm.name
        step = build_paired_benchmark(
            run_id=run_id,
            checkpoint=(args.checkpoint,),
            arm=BenchmarkArm(arm.name),
            num_steps=plan["protocol"]["steps_per_arm"],
            warmup_steps=5,
            profile_steps=2,
            training_data_mode=TrainingDataMode.MIXTURE,
            version="dev",
        )
        prefix = marin_prefix()
        context = StepContext.for_run(
            prefix_join(prefix, "tmp/ttl=30d/" + run_id),
            prefix,
            region="us-east-08a",
            deps=step.deps,
            runtime_args=step.runtime_args,
        )
        config = step.build_config(context)
        configs.append(config)
        arguments[arm.name] = {
            "arm": arm.name,
            "source_commit": arm.commit,
            "config": base64.b64encode(cloudpickle.dumps(config)).decode("ascii"),
            "protocol": plan["protocol"],
        }
    config = configs[0]
    assert all(other.resources == config.resources for other in configs)
    assert all(other.processes_per_task == config.processes_per_task for other in configs)
    trainer = config.trainer.trainer
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=trainer.watch.is_enabled and config.trainer.watch_mode == WatchMode.INLINE,
        processes_per_task=config.processes_per_task,
        moe_implementation=config.model.moe_implementation,
        remat_mode=config.model.remat_mode,
    )
    request = JobRequest(
        name="grug-pair-" + args.run_id,
        entrypoint=Entrypoint.from_callable(run_training_pair, args=[sources, order, arguments]),
        resources=config.resources,
        environment=create_environment(
            env_vars=resolve_training_env(base_env=_forwarded_env_vars(), resources=config.resources),
            extras=extras_for_resources(config.resources),
        ),
        processes_per_task=config.processes_per_task,
        max_retries_failure=0,
        max_retries_preemption=0,
        max_task_failures=0,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "job": request.name,
                    "order": order,
                    "arms": arguments,
                    "configurations": [draccus.encode(config) for config in configs],
                },
                indent=2,
            )
        )
        return
    job = current_client().submit(request, adopt_existing=False)
    job.wait(raise_on_failure=True)


if __name__ == "__main__":
    main()
