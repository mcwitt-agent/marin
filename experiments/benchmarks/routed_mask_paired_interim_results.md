# Routed-expert masking: three-pair interim results

Removing the mixture-of-experts (MoE) incoming-gradient mask and clearing only unused output-buffer tails improved measured training throughput in three completed comparisons: +2.99%, +1.13% and +1.12%. The candidate is not accepted. Four pairs were planned; the fourth remains outstanding, so this interim report gives individual estimates without a final confidence interval.

The baseline is main `1f9c387b6c452895ff58573c2b6117c5500303db` plus the native SM100 attention-backward optimization from [PR #9228](https://github.com/marin-community/marin/pull/9228), retaining the accepted ragged-all-to-all configuration. The frozen control commit is `9562c0779564f130c2b25d3d1bf3f9b49efdff89`; the candidate is `ee204a7569c200e2c0382e54c9d48fea7486b1d2`.

The hero workload is a 48-layer Grug MoE model with hidden width 6,144, 384 routed experts, eight selected experts per token and sequence length 4,096. Each pair restored the latest completed hero checkpoint selected once at pair setup. Control and treatment ran as fresh processes on the same physical 64 GB200 GPUs across 16 hosts in one NVLink rack, with 64 updates each. The global training batch contained 1,024 sequences; optimizer schedule parameters retained the production global batch size of 11,264 sequences. Throughput change is mean control step time divided by mean treatment step time, minus one, using all 56 zero-based relative updates 8–63.

| Pair | Order | Checkpoint step | Control seconds/update | Treatment seconds/update | Throughput change |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Control, treatment | 128449 | 15.002522 | 14.566281 | +2.9949% |
| 2 | Treatment, control | 128861 | 15.021322 | 14.853746 | +1.1282% |
| 3 | Control, treatment | 129063 | 15.012882 | 14.846897 | +1.1180% |

All three pairs passed source, package, device, checkpoint and profile checks. Each arm recorded 64 finite losses. Rank-0 local-GPU JAX allocator lifetime peak was 103.088165 GiB for control and 115.537537 GiB for treatment in every pair, an increase of 12.449372 GiB. This peak includes initialization and compilation and excludes other allocators and devices. The maximum post-update in-use allocation in the same rank-0 GPU allocator was 35.596301 GiB in each arm of every pair.

Loss differences below are treatment minus control across all 64 updates, in nats.

| Pair | Signed mean difference | Signed final difference | Maximum absolute difference |
| --- | ---: | ---: | ---: |
| 1 | −0.0000274 | +0.0017093 | 0.0017093 |
| 2 | −0.0001622 | −0.0000408 | 0.0005901 |
| 3 | +0.0000961 | +0.0003135 | 0.0004752 |

Pair 1's late positive loss gap remains unexplained. Routing metrics differ between arms. Nine matched dense operations choose different kernels between control and treatment in each pair; each arm repeats its kernel choices across all three pairs. These observations do not establish the cause of the loss differences or unchanged long-term convergence. Input-batch byte checksums were not recorded; fixed source, seed and checkpoint support matched ordering but do not directly establish identical input bytes.

The original pair-2 attempt failed Iris's completion gate after a worker pod was deleted. Its diagnostic measurements remain excluded from the primary study and are retained under `excluded_attempts`. One replacement used unchanged code and scoring with a newly selected latest checkpoint. Original checkpoint-retention settings were restored for all three completed pairs and the failed attempt. The [incident record](https://marina.oa.dev/echo/wiki/489) documents the infrastructure failure.

[Derived results](routed_mask_paired_interim_results.json) contain per-update timings and losses, source and protocol hashes, job IDs, checkpoint paths, profile counts, allocator counters, manual reviews and the excluded attempt. The [protocol](hero_path_training_protocol.json) defines the source, placement, profile, loss and memory checks; each pair's `manual_review` and `evidence_sha256` fields identify the reviewed evidence. The final analysis will use the four pairs as replicates and report a paired Fieller 95% interval with three degrees of freedom for the ratio of mean control to mean treatment run-average step times, minus one, weighting each pair equally.

The benchmark harness and [result exporter](hero_mask_public_results.py) accompany the data. The [publication manifest](routed_mask_publication_sources.json) records measured and published source hashes; formatting changes preserve the Python abstract syntax trees. Production candidate code remains local, so these artifacts are not sufficient for a self-contained rerun. Raw profiles, checkpoint contents and credentials are excluded.

Manual review permits the planned fourth pair with unchanged source, scoring and checkpoint policy. It does not accept the optimization or authorize deployment.
