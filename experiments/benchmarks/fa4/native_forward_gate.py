# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run native-forward correctness before its exploratory timing screen."""

import argparse
import hashlib
import importlib.metadata
import json
import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(protocol["gpu_process_timeout"])
    deadline = time.monotonic() + protocol["gpu_process_timeout"]
    root = Path(__file__).resolve().parents[3]
    for name, expected in (protocol["production_source_sha256"] | protocol["harness_source_sha256"]).items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    versions = {name: importlib.metadata.version(name) for name in protocol["packages"]}
    assert versions == protocol["packages"], versions
    print(
        "HERO_NATIVE_FORWARD_GATE_RUNTIME "
        + json.dumps({"protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(), "protocol": protocol}),
        flush=True,
    )
    device_check = (
        "import jax; from levanter.cutlass_kernel_cache import gpu_compute_capability; "
        "import flash_attn.cute.flash_fwd_sm100; import flash_attn.cute.flash_bwd_sm100; "
        "assert jax.default_backend() == 'gpu' and gpu_compute_capability() == 100; "
        "assert jax.device_count() == jax.local_device_count() == 1"
    )
    testfile = "lib/levanter/tests/grug/test_fa4_cute_attention.py"
    commands = [
        [sys.executable, "-c", device_check],
        [sys.executable, "-m", "pytest", testfile, "-k", "real_gpu", "-n", "0", "--session-timeout=2400", "-q", "-ra"],
        [
            sys.executable,
            "-m",
            "pytest",
            testfile + "::test_real_gpu_fa4_cute_zeroes_padding_tiles_before_reusing_query_storage",
            "-m",
            "slow",
            "-n",
            "0",
            "--session-timeout=600",
            "-q",
            "-ra",
        ],
        *[
            [
                sys.executable,
                "-m",
                "experiments.benchmarks.fa4.native_forward_timing",
                "--variant",
                variant,
                "--protocol",
                str(args.protocol),
            ]
            for variant in protocol["timing_order"]
        ],
    ]
    for command in commands:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Native-forward gate exceeded its GPU process deadline")
        print("HERO_NATIVE_FORWARD_GATE_PHASE " + json.dumps(command), flush=True)
        subprocess.run(command, check=True, timeout=remaining)
    print("HERO_NATIVE_FORWARD_GATE_SUCCESS", flush=True)


if __name__ == "__main__":
    main()
