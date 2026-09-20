# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute one restored training arm from its isolated source snapshot."""

import base64
import dataclasses
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import cloudpickle
import jax
from iris.runtime.jax_init import initialize_jax
from levanter.utils.jax_utils import multihost_broadcast_sync

from experiments.benchmarks.hero_path_training import BenchmarkArm, _run_paired_restored_local
from experiments.grug.moe_hero_ep.train import GrugRunConfig


def prepare_training_config(payload: str, output_dir: Path) -> GrugRunConfig:
    """Restore the transported config and place logs in this arm's output directory."""
    config = cloudpickle.loads(base64.b64decode(payload, validate=True))
    assert isinstance(config, GrugRunConfig)
    return dataclasses.replace(
        config,
        trainer=dataclasses.replace(
            config.trainer,
            trainer=dataclasses.replace(
                config.trainer.trainer,
                log_dir=output_dir / "logs",
                # The paired launcher owns JAX startup and the enclosing Iris job.
                # Levanter's automatic exit hook would complete it after the first arm.
                distributed=dataclasses.replace(config.trainer.trainer.distributed, initialize_jax_distributed=False),
            ),
        ),
    )


def main() -> None:
    arguments = json.loads(Path(sys.argv[1]).read_text())
    repository = Path(__file__).resolve().parents[2]
    protocol_path = repository / "experiments/benchmarks/hero_path_training_protocol.json"
    assert json.loads(protocol_path.read_text()) == arguments["protocol"]
    config = prepare_training_config(arguments["config"], Path(os.environ["IRIS_OUTPUT_DIR"]))
    initialize_jax(endpoint_name=f"hero_pair_{arguments['arm']}")
    devices = [
        {"id": device.id, "process_index": device.process_index, "kind": device.device_kind}
        for device in jax.local_devices()
    ]
    record = {
        "arm": arguments["arm"],
        "run_id": config.trainer.trainer.id,
        "source_commit": arguments["source_commit"],
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "hostname": socket.gethostname(),
        "local_devices": devices,
        "local_device_ids": os.environ["IRIS_MULTIGPU_LOCAL_DEVICE_IDS"],
        "gpu_uuids": (
            subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
            .strip()
            .splitlines()
        ),
    }
    print("HERO_PAIR_ARM_RUNTIME " + json.dumps(record), flush=True)
    _run_paired_restored_local(config, arm=BenchmarkArm(arguments["arm"]))
    multihost_broadcast_sync(True)
    print("HERO_PAIR_ARM_FINISHED " + json.dumps(record), flush=True)
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
