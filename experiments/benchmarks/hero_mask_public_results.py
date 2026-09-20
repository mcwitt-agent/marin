# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export reviewed routed-expert benchmark pairs without raw profile payloads."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_pair(state, number):
    directory = Path(state["directory"])
    submission = state["pairs"][number - 1]
    assert submission["pair"] == number and submission["state"] == "succeeded"
    assert submission["source"] == state["source"]
    assert submission["order"] == state["protocol"]["orders"][number - 1]
    assert number in state["prechecked_pairs"]
    pin = state["checkpoint_pins"][str(number)]
    assert pin["status"] == "released"
    training_path = directory / f"pair-{number}" / "result.json"
    profile_path = directory / f"pair-{number}-validated-result.json"
    review_path = directory / f"pair-{number}-review.json"
    memory_path = directory / f"pair-{number}-allocator-memory" / "summary.json"
    training = read_json(training_path)
    profile = read_json(profile_path)
    review = read_json(review_path)
    memory = read_json(memory_path)
    assert training["study"] == profile["study"] == memory["study"] == state["study"]
    assert training["pair"] == profile["pair"] == review["pair"] == memory["pair"] == number
    assert training["job"] == memory["job"] == submission["child"]
    assert profile["protocol"] == state["protocol"]
    assert profile["training_evidence"] == training
    assert profile["analysis_submission"]["state"] == "succeeded"
    assert review["source"] == state["source"]
    assert review["training_result_sha256"] == digest(training_path)
    assert review["validated_profile_sha256"] == digest(profile_path)
    assert review["decision"] in ("continue_planned_measurements", "complete_planned_measurements")
    arms = {}
    for arm in ("control", "candidate"):
        value = training["arms"][arm]
        runtime = value["runtime"]
        expected = state["protocol"][arm]
        assert runtime["production_source_sha"] == expected["commit"]
        assert runtime["source_sha256"] == expected["source_sha256"]
        assert runtime["gpu_package_versions"] == state["protocol"]["packages"]
        assert runtime["protocol_sha256"] == state["protocol_sha256"]
        assert value["checkpoint"]["benchmark_checkpoint"] == pin["checkpoint"]
        assert value["checkpoint"]["checkpoint_step"] == pin["step"]
        boundaries = value["boundaries"]
        assert len(boundaries) == 64
        assert {row["process_index"] for row in boundaries} == set(range(64))
        assert all(row["source_commit"] == expected["commit"] for row in boundaries)
        assert all(row["protocol_sha256"] == state["protocol_sha256"] for row in boundaries)
        metrics = {int(step): row for step, row in value["metrics"].items()}
        assert set(metrics) == set(range(pin["step"], pin["step"] + 64))
        samples = []
        for step in sorted(metrics):
            row = metrics[step]
            assert math.isfinite(row["loss"]) and math.isfinite(row["duration"]) and row["duration"] > 0
            samples.append(
                {
                    "step": step,
                    "relative_step": step - pin["step"],
                    "duration_seconds": row["duration"],
                    "loss_nats": row["loss"],
                }
            )
        scored = samples[8:64]
        assert len(scored) == 56
        mean = statistics.mean(row["duration_seconds"] for row in scored)
        assert mean == value["mean_scored_seconds"]
        allocation = memory["arms"][arm]
        assert allocation["recorded_steps"] == 64
        derived = profile["profile_evidence"][arm]
        kernel_profile = derived["probes"]["profile"]
        forward_count = sum(row["calls"] for row in kernel_profile["profile_forward_launch_histogram"])
        backward_count = sum(row["calls"] for row in kernel_profile["profile_backward_launch_histogram"])
        tail_count = sum(row["calls"] for row in kernel_profile["groups"] if row["group"] == "routed_output_tail_clear")
        assert (forward_count, backward_count) == (192, 96)
        assert tail_count == 0 if arm == "control" else tail_count > 0
        arms[arm] = {
            "run_id": submission["run_id"] + "-" + arm,
            "runtime": runtime,
            "verified_runtime_ranks": len(boundaries),
            "samples": samples,
            "mean_scored_seconds": mean,
            "memory": {
                key: allocation[key]
                for key in ("max_peak_gib", "max_in_use_gib", "allocator_limits_gib", "recorded_steps")
            },
            "compiler_planned_default_plus_collective_gib": derived["planned_default_plus_collective_gib"],
            "profile_kernel_counts": {
                "segmented_attention_forward": forward_count,
                "native_sm100_attention_backward": backward_count,
                "routed_output_tail_clear": tail_count,
            },
        }
    control_devices = {row["process_index"]: row for row in training["arms"]["control"]["boundaries"]}
    candidate_devices = {row["process_index"]: row for row in training["arms"]["candidate"]["boundaries"]}
    for rank in range(64):
        for key in ("hostname", "local_devices", "local_device_ids", "gpu_uuids"):
            assert control_devices[rank][key] == candidate_devices[rank][key]
    throughput = (arms["control"]["mean_scored_seconds"] / arms["candidate"]["mean_scored_seconds"] - 1) * 100
    assert throughput == training["throughput_percent"]
    differences = [
        candidate["loss_nats"] - control["loss_nats"]
        for control, candidate in zip(arms["control"]["samples"], arms["candidate"]["samples"], strict=True)
    ]
    assert differences == [row["difference"] for row in training["loss"]["trajectory"]]
    assert max(map(abs, differences)) == training["loss"]["max_absolute_difference"]
    return {
        "pair": number,
        "order": submission["order"],
        "job_id": submission["job_id"],
        "child_job_id": submission["child"],
        "submitted_utc": submission["submitted_utc"],
        "finished_utc": submission["finished_utc"],
        "checkpoint_retention": pin,
        "rack": training["rack"],
        "same_physical_gpus_verified": True,
        "arms": arms,
        "throughput_percent": throughput,
        "loss": training["loss"],
        "memory_scope": memory["scope"],
        "candidate_minus_control_peak_gib": memory["candidate_minus_control_peak_gib"],
        "manual_review": review,
        "evidence_sha256": {path.name: digest(path) for path in (training_path, profile_path, review_path, memory_path)},
    }


def build_excluded_attempts(state):
    records = []
    accepted_jobs = {row["job_id"] for row in state["pairs"]}
    for excluded in state.get("excluded_attempts", []):
        submission = excluded["submission"]
        plan_path = Path(excluded["recovery_plan"])
        assert digest(plan_path) == excluded["recovery_plan_sha256"]
        plan = read_json(plan_path)
        assert submission["state"] == "failed" and submission["job_id"] not in accepted_jobs
        assert plan["failed_job"] == submission["job_id"]
        assert excluded["checkpoint_pin"]["status"] == "released"
        diagnostic_path = Path(plan["diagnostic_result"])
        assert digest(diagnostic_path) == plan["diagnostic_result_sha256"]
        diagnostic = read_json(diagnostic_path)
        assert diagnostic["included_in_primary_results"] is False
        assert diagnostic["job"] == submission["job_id"]
        records.append(
            {
                "submission": {
                    key: submission[key]
                    for key in (
                        "pair",
                        "run_id",
                        "job_id",
                        "child",
                        "state",
                        "order",
                        "source",
                        "checkpoint",
                        "submitted_utc",
                        "finished_utc",
                    )
                },
                "checkpoint_retention": excluded["checkpoint_pin"],
                "exclusion_reason": excluded["reason"],
                "incident_url": plan["incident_url"],
                "recovery_policy": plan["policy"],
                "diagnostic_only": diagnostic,
                "recovery_plan_sha256": digest(plan_path),
            }
        )
    return records


def build_export(state, count):
    assert count in (3, 4)
    assert state["protocol"]["steps_per_arm"] == 64
    assert state["protocol"]["score_relative_steps"] == [8, 63]
    assert len(state["protocol"]["orders"]) == 4
    assert state["prechecked_pairs"][:count] == list(range(1, count + 1))
    pairs = [build_pair(state, number) for number in range(1, count + 1)]
    for arm in ("control", "candidate"):
        assert all(pair["arms"][arm]["runtime"] == pairs[0]["arms"][arm]["runtime"] for pair in pairs)
    aggregate = None
    if count == 4:
        assert state["phase"] == "four_pairs_fully_validated"
        aggregate = read_json(Path(state["directory"]) / "validated-study-summary.json")
        assert aggregate["study"] == state["study"] and aggregate["source"] == state["source"]
        assert aggregate["protocol"] == state["protocol"]
        assert [row["pair"] for row in aggregate["validation_evidence"]] == [1, 2, 3, 4]
        for pair, evidence in zip(pairs, aggregate["validation_evidence"], strict=True):
            assert evidence["sha256"] == pair["evidence_sha256"][f"pair-{pair['pair']}-validated-result.json"]
    return {
        "study": state["study"],
        "stage": "three_pair_interim" if count == 3 else "four_pair_final_measurement",
        "comparison": (
            "Remove the routed expert incoming-gradient mask and clear only unused output rows, "
            "on the accepted SM100 backward-attention baseline."
        ),
        "harness_commit": state["source"],
        "protocol": state["protocol"],
        "protocol_sha256": state["protocol_sha256"],
        "source_plan_sha256": state["plan_sha256"],
        "hardware": {"gpu_model": "GB200", "gpu_count": 64, "hosts": 16, "nvlink_racks_per_pair": 1},
        "pairs": pairs,
        "excluded_attempts": build_excluded_attempts(state),
        "aggregate_statistics": aggregate,
        "acceptance_status": "Measurement completion does not establish acceptance; inspect loss and memory reviews.",
        "source_availability": (
            "Production candidate code is local. Source hashes identify the measured "
            "implementation; this data export does not publish that code."
        ),
        "limitations": [
            (
                "Four pairs were planned. The third-pair report is interim, with individual results "
                "and no final confidence interval or early acceptance."
            ),
            (
                "Pairs, not updates, are the statistical replicates. Final paired intervals assume "
                "independent approximately normal run-pair means."
            ),
            (
                "All 56 scored updates in each arm are retained. No timing-based pair replacement or "
                "sample exclusion is allowed in the primary result."
            ),
            (
                "Each pair restores the latest completed hero checkpoint selected at pair setup. "
                "Different pairs may restore different checkpoints."
            ),
            "Training batch size is 1024; optimizer schedule parameters retain production batch size 11264.",
            (
                "64-update loss trajectories do not establish long-term convergence. Pair 1 has "
                "unexplained late positive loss drift despite exact component outputs and gradients."
            ),
            (
                "Pair 1 increases the rank-0 JAX allocator peak by 12.449371814727797 GiB. This "
                "high-water mark includes initialization and compilation and excludes other "
                "allocators."
            ),
            (
                "Compiler-planned allocation is not a measurement of total physical GPU memory. "
                "Profile kernel timings can overlap and are not additive wall-time savings."
            ),
            (
                "Matched source, seed and checkpoint support matched data ordering; input batch byte "
                "checksums were not recorded."
            ),
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--pairs", type=int, choices=(3, 4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = read_json(args.state)
    assert digest(args.protocol) == state["protocol_sha256"]
    assert read_json(args.protocol) == state["protocol"]
    result = build_export(state, args.pairs)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": digest(args.output),
                "pairs": args.pairs,
                "training_samples": args.pairs * 128,
            }
        )
    )


if __name__ == "__main__":
    main()
