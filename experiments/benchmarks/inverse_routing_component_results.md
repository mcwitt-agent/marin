# Inverse token routing: component measurements

Replacing inverse-permutation sorting with an indexed write reduced token-combine and dispatch-gradient component duration by 8.34–8.71% on one NVIDIA GB200. The maximum observed numerical output difference was zero in all six measured path cases, and eleven GPU routing/gradient reference cases passed. These component results do not establish a training-throughput gain or convergence.

Token routing groups assignments by expert and later restores token order. For a permutation `p`, the candidate constructs its inverse by writing `inverse[p[i]] = i`. The permutation contains each position once, even when expert IDs repeat, so destinations are unique. Floating-point reductions are unchanged. The accepted training baseline includes the SM100 backward-attention optimization from [PR #9228](https://github.com/marin-community/marin/pull/9228).

## Token-combine and dispatch-gradient components

The screen uses 65,536 local tokens, top-8 routing, 384 experts and hidden width 3,072. Routed values are BF16 and combine weights are FP32. The combine path restores token order and reduces the eight expert contributions. The dispatch-gradient path restores and sums gradients for each token. Both retain the existing Sonic gather-and-sum implementation. The expert-sorted permutation is constructed before timing; each timed call constructs its inverse. Expert-key sorting, communication, expert matrix multiplies and optimizer work are excluded. Both arms use the same software environment and floating-point implementation.

| Routing distribution | Path | Control (µs/call) | Candidate (µs/call) | Duration reduction |
| --- | --- | ---: | ---: | ---: |
| Uniform expert IDs | Combine | 777.20 | 709.47 | 8.71% |
| Uniform expert IDs | Dispatch gradient | 771.58 | 705.49 | 8.56% |
| All assignments to one expert | Combine | 756.56 | 692.93 | 8.41% |
| All assignments to one expert | Dispatch gradient | 756.35 | 693.30 | 8.34% |
| 80% forced to one expert | Combine | 754.76 | 691.78 | 8.35% |
| 80% forced to one expert | Dispatch gradient | 757.75 | 694.13 | 8.40% |

Table values are arithmetic means of six batch means per arm, with 50 calls per batch. Each batch follows ten warm-up calls. The batch order is control, candidate, candidate, control, repeated three times. Every call blocks for completion; measured time includes host dispatch and synchronization. The distributions and repeated batches are correlated component measurements, not independent training runs. The percentage range across cases is not a confidence interval. All six candidate batch means were below all six control batch means within every measured case; individual samples and ranges are in the data record.

The compiler-reported temporary storage decreased from 6,553,860 to 2,359,568 bytes for combine and from 8,651,012 to 4,456,720 bytes for dispatch gradient. The CUB sort custom call was absent from both candidate components. These allocations do not measure full-training peak memory.

Uniform expert IDs are drawn independently for every assignment, so duplicate experts within a token are permitted. The skewed case independently redirects each assignment to expert zero with probability 0.8; the remainder retain uniformly drawn IDs. This synthetic setup tests permutations and memory access patterns, not a realistic top-k router distribution. Routed values come from seeded standard-normal BF16 samples; combine weights use a softmax of standard-normal FP32 samples. Output comparisons measure the maximum absolute difference from the sorting control after casting both outputs to FP32.

The eleven GPU reference cases cover inverse permutations of lengths 0, 1, 17 and 1,024; dispatch outputs and gradients at top-k 1, 2 and 8 in BF16 and FP32; and agreement between indexed and materialized dispatch.

## Inverse indices alone

The preceding index-only screen used 524,288 int32 assignments with the same three distribution definitions. Six batches of 100 calls per arm measured duration reductions of 39.98% for uniform routing, 38.61% for one-expert routing and 36.46% for skewed routing. All inverse indices matched an independent NumPy reference exactly. Compiler-reported temporary storage fell from 8,480,512 to 524,304 bytes. This screen uses the same warm-up, interleaving, synchronization and mean-aggregation rules. It excludes the gather-and-sum work measured above.

## Evidence and limits

[The data record](inverse_routing_component_results.json) includes all 108 timing-batch records across both screens, exact measurement scripts, frozen protocols, package versions, source hashes, job IDs and derived compiler summaries. The full-path controls were checked against baseline `9562c0779564`: replacing the candidate inverse helper with `jnp.argsort` reproduces the baseline function syntax trees. The candidate revision is `c2b3a4e8db9d`, based on main `1f9c387b6c45` plus the accepted attention changes. Production candidate code is not included in this results branch, so the branch alone is insufficient to rerun the screens.

The records preserve their original protocol metadata. The full-path protocol inherited an “index-only screen” phrase in its compiler-flag provenance note; its executed measurement script and results cover the complete local paths described above.

An exploratory restored-training comparison is in progress. No restored-training result or acceptance decision is included here.
