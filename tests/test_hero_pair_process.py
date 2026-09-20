# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.benchmarks.hero_pair_process import ArmSource, PairSources, run_pair_sources


def test_pair_runs_fresh_source_processes_and_preserves_the_checkout(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    worker = """import json,os,sys
from pathlib import Path
import kernel
args=json.loads(Path(sys.argv[1]).read_text())
Path(args['result']).write_text(json.dumps({'value':kernel.VALUE,'pid':os.getpid(),'source':kernel.__file__}))
"""
    (repository / "worker.py").write_text(worker)
    (repository / "kernel.py").write_text("VALUE = 2\n")
    common = {name: hashlib.sha256((repository / name).read_bytes()).hexdigest() for name in ("worker.py", "kernel.py")}
    sources = PairSources(
        common,
        (".",),
        (
            ArmSource("control", "control-source", {"kernel.py": "VALUE = 1\n"}, {}),
            ArmSource("candidate", "candidate-source", {}, {}),
        ),
        30,
    )
    results = {arm: tmp_path / f"{arm}.json" for arm in ("control", "candidate")}
    run_pair_sources(
        repository,
        sources,
        ("candidate", "control"),
        ("-m", "worker"),
        {arm: {"result": str(path)} for arm, path in results.items()},
        tmp_path / "outputs",
    )
    control, candidate = (json.loads(results[arm].read_text()) for arm in ("control", "candidate"))
    assert (control["value"], candidate["value"]) == (1, 2)
    assert len({control["pid"], candidate["pid"], os.getpid()}) == 3
    assert control["source"] != candidate["source"]
    assert (repository / "kernel.py").read_text() == "VALUE = 2\n"


def test_failed_first_arm_prevents_second_execution(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    worker = """import json,sys
from pathlib import Path
args=json.loads(Path(sys.argv[1]).read_text())
with Path(args['record']).open('a') as out:
    out.write(args['arm']+'\\n')
raise SystemExit(args['exit'])
"""
    (repository / "worker.py").write_text(worker)
    sources = PairSources(
        {"worker.py": hashlib.sha256(worker.encode()).hexdigest()},
        (".",),
        (ArmSource("control", "c", {}, {}), ArmSource("candidate", "n", {}, {})),
        30,
    )
    record = tmp_path / "executed.txt"
    with pytest.raises(subprocess.CalledProcessError):
        run_pair_sources(
            repository,
            sources,
            ("control", "candidate"),
            ("-m", "worker"),
            {
                "control": {"record": str(record), "arm": "control", "exit": 3},
                "candidate": {"record": str(record), "arm": "candidate", "exit": 0},
            },
            tmp_path / "outputs",
        )
    assert record.read_text() == "control\n"


def test_training_config_preparation_preserves_profiler_and_job_lifecycle(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    protocol = json.loads((repository / "experiments/benchmarks/hero_path_training_protocol.json").read_text())
    plan = {
        "ready_for_execution": False,
        "common_sha256": {},
        "python_roots": ["."],
        "arms": [
            {"name": arm, "commit": protocol[arm]["commit"], "overrides": {}, "expected_sha256": {}}
            for arm in ("control", "candidate")
        ],
        "protocol": protocol,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.benchmarks.hero_pair_training",
            "--plan",
            str(plan_path),
            "--run-id",
            "hero-pair-config-test",
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--order",
            "control,candidate",
            "--dry-run",
        ],
        cwd=repository,
        text=True,
        capture_output=True,
        check=True,
    )
    transmitted = tmp_path / "arguments.json"
    transmitted.write_text(result.stdout)
    receiver = """import json,sys
from pathlib import Path
from types import SimpleNamespace
import draccus
import jax
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import JobName
from iris.cluster.platforms.types import find_free_port
from levanter import distributed
from experiments.benchmarks.hero_pair_arm import prepare_training_config
jax.distributed.initialize(coordinator_address=f"127.0.0.1:{find_free_port()}",num_processes=1,process_id=0,initialization_timeout=15)
# Replace only the Iris job-info and completion RPC boundaries; run actual JAX startup/teardown.
distributed.get_job_info=lambda: JobInfo(task_id=JobName.from_wire("/test/paired-training/0"))
distributed.iris_ctx=lambda: SimpleNamespace(client=SimpleNamespace(complete_job=lambda job: Path(sys.argv[3]).touch()))
data=json.loads(Path(sys.argv[1]).read_text())
configs=[prepare_training_config(arm['config'],Path(sys.argv[2])/name) for name,arm in data['arms'].items()]
for config in configs:
    trainer=config.trainer.trainer
    trainer.distributed.initialize()
    profile_dir=trainer.log_dir/trainer.id/'profiler'
    trainer.profiler.build(str(profile_dir),trainer.id)
    assert profile_dir.is_dir()
print(json.dumps([draccus.encode(config) for config in configs]))
jax.distributed.shutdown()
"""
    output_dir = tmp_path / "outputs"
    completed_job = tmp_path / "completed-job"
    restored = subprocess.run(
        [sys.executable, "-c", receiver, str(transmitted), str(output_dir), str(completed_job)],
        cwd=repository,
        text=True,
        capture_output=True,
        check=True,
    )
    expected = json.loads(result.stdout)["configurations"]
    for arm, config in zip(("control", "candidate"), expected, strict=True):
        config["trainer"]["trainer"]["log_dir"] = str(output_dir / arm / "logs")
        config["trainer"]["trainer"]["distributed"]["initialize_jax_distributed"] = False
    assert json.loads(restored.stdout) == expected
    assert not completed_job.exists(), "A training arm completed the enclosing paired Iris job"
