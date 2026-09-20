# Routed-expert masking: four-pair results

Removing the mixture-of-experts (MoE) incoming-gradient mask and clearing only unused output-buffer tails increased measured training throughput by 1.63%, with a paired Fieller 95% interval of +0.22% to +3.09%. Mean step time fell from 15.00310 to 14.76189 seconds. This supports a timing gain in the measured one-rack workload. The candidate remains outside the research baseline because loss differences remain unexplained and its higher observed peak allocation has not been accepted for production.

The baseline is main `1f9c387b6c452895ff58573c2b6117c5500303db` plus the native SM100 attention-backward optimization from [PR #9228](https://github.com/marin-community/marin/pull/9228), retaining ragged-all-to-all-only symmetric registration. The control commit is `9562c0779564f130c2b25d3d1bf3f9b49efdff89`; the candidate is `ee204a7569c200e2c0382e54c9d48fea7486b1d2`. Native forward, copy-engine and shared-MLP candidates are excluded.

The hero workload is a 48-layer Grug MoE model with hidden width 6,144, 384 routed experts, eight selected experts per token and sequence length 4,096. Each pair selected the latest completed hero checkpoint once and restored it in both arms. Both arms ran as fresh processes on the same physical 64 GB200 GPUs across 16 hosts in one NVLink rack. Each arm ran 64 updates. The global training batch contained 1,024 sequences; optimizer schedule parameters retained the production global batch size of 11,264 sequences. Full-production-batch throughput and long-term convergence were not measured.

| Pair | Order | Checkpoint step | Control seconds/update | Treatment seconds/update | Throughput change |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Control, treatment | 128449 | 15.002522 | 14.566281 | +2.9949% |
| 2 | Treatment, control | 128861 | 15.021322 | 14.853746 | +1.1282% |
| 3 | Control, treatment | 129063 | 15.012882 | 14.846897 | +1.1180% |
| 4 | Treatment, control | 129269 | 14.975664 | 14.780644 | +1.3194% |

The primary score uses all 56 zero-based relative updates 8–63 in each arm. The estimator is the ratio of mean control to mean treatment run-average step times, minus one, weighting each pair equally. Pairs are the statistical replicates. The four-pair Fieller and paired-t intervals use three degrees of freedom and assume independent, approximately normal pair means. The paired-t interval for seconds saved per update is 0.03322–0.44919 seconds. The predeclared sensitivity analysis excludes absolute step numbers divisible by ten and gives +1.62% throughput, with a 95% interval of +0.22% to +3.06%. No timing samples or completed pairs were removed based on their results.

All four pairs passed source, package, device, checkpoint and profile checks. The eight training arms recorded 512 finite losses. Rank-0 local-GPU JAX allocator lifetime peak was 103.088165 GiB for control and 115.537537 GiB for treatment in every pair, an increase of 12.449372 GiB. This peak includes initialization and compilation and excludes other allocators and devices. The maximum post-update in-use allocation in the same allocator was 35.596301 GiB in each arm of every pair. No out-of-memory failure occurred in this study.

Loss differences below are treatment minus control across all 64 updates, matched by relative update index, in nats.

| Pair | Signed mean difference | Signed final difference | Maximum absolute difference | Signed last-seven mean |
| --- | ---: | ---: | ---: | ---: |
| 1 | −0.0000274 | +0.0017093 | 0.0017093 | +0.0014444 |
| 2 | −0.0001622 | −0.0000408 | 0.0005901 | −0.0002861 |
| 3 | +0.0000961 | +0.0003135 | 0.0004752 | +0.0002497 |
| 4 | +0.0000942 | +0.0006496 | 0.0006496 | +0.0001505 |

Pair 1's sustained late positive loss gap remains unexplained. Compared expert-MLP outputs, active retained intermediate rows, and input and weight gradients matched the baseline with zero observed differences across three synthetic routing patterns before training, but these short trajectories do not establish numerical equivalence or unchanged convergence. Router metrics differ between arms. Nine matched dense operations choose different kernels between control and treatment, and two operation groups appear only in the treatment's dense-kernel summary. Each arm repeats its operation groups and kernel choices across all four pairs. These observations do not establish the cause of the loss differences. Input-batch byte checksums were not recorded; source, seed and checkpoint checks support matched ordering but do not directly establish identical input bytes.

The original pair-2 attempt failed Iris's completion gate after a worker pod was deleted. Its diagnostic measurements remain excluded from the primary study and are retained under `excluded_attempts`. One replacement used unchanged code and scoring with a newly selected latest checkpoint. Original checkpoint-retention settings were restored for all four completed pairs and the failed attempt. The [incident record](https://marina.oa.dev/echo/wiki/489) documents the infrastructure failure.

[Derived results](routed_mask_paired_results.json) contain all per-update timings and losses, source and protocol hashes, job IDs, checkpoint paths, profile counts, allocator counters, manual reviews, aggregate statistics and the excluded attempt. The [protocol](hero_path_training_protocol.json) defines the checks. The [baseline decision](routed_mask_baseline_decision.json) retains the accepted native SM100 backward implementation for subsequent experiments; the routed-mask candidate is preserved for further investigation. No additional unchanged-candidate pairs or production deployment are planned.

The benchmark harness and [result exporter](hero_mask_public_results.py) accompany the data. The [publication manifest](routed_mask_publication_sources.json) records measured and published harness hashes; formatting changes preserve the Python abstract syntax trees. Production candidate code remains local, so these artifacts are not sufficient for a self-contained training rerun. Raw profiles, checkpoint contents and credentials are excluded. The [three-pair interim report](routed_mask_paired_interim_results.md) remains available as the earlier record.
