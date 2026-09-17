# Native SM100 attention backward

Native SM100 attention backward improved restored hero training throughput by **4.08%**, with a **95% paired confidence interval of 2.29%–5.89%**, across four predeclared run pairs using FlashAttention 4.0.0b28. Mean step time fell from 15.5769 to 14.9660 seconds.

The measured source is `ecf3cac2aafdd66403684180da4f595a54a2f813`, based on Marin main `f113d5f0ed47b9cdfa6eb5c5b93a47c5d1ff08f2`. This baseline includes [PR #8741](https://github.com/marin-community/marin/pull/8741), which updates FlashAttention to b28. Both numerical checks and throughput measurements passed on this dependency set. Production attention dispatch remains unchanged; the adapter is enabled only by the benchmark.

## Implementation

The [benchmark adapter](https://github.com/mcwitt-agent/marin/blob/ecf3cac2aafdd66403684180da4f595a54a2f813/experiments/benchmarks/fa4/native_sm100.py) uses the upstream SM100 backward kernel with 128×128 tiles, one CTA per cluster and two-CTA instructions disabled. It supplies packed-document bounds and validity through the custom mask and block-sparse metadata. Preprocessing, backward accumulation and postprocessing are separate JAX FFI calls. The launcher zeros all three FP32 gradient accumulators before each backward invocation, including reused buffers.

This path uses SM100 tensor memory and `tcgen05` matrix instructions. The control uses the existing SM120-oriented warp-MMA path. The adapter supports BF16, head dimension 128 and query-to-KV head ratios four and eight. Select the implementation in a fresh process before JAX tracing: the context manager temporarily replaces the backend's backward entrypoint and does not invalidate previously compiled functions.

## Measured workload and protocol

Each arm ran the hero recipe with expert parallelism across 64 GB200 GPUs in one NVLink rack: 16 nodes with four GPUs each, global batch 1,024 and sequence length 4,096. The model has 48 layers, hidden width 6,144, 48 query heads, 12 local or six global KV heads, head dimension 128, and 384 routed experts with top-eight routing. Local attention uses a 2,048-token window; every fourth layer uses global attention.

The baseline uses XLA `708c3a4ec79c7581ffe28c4bb2be2b6dbf870519`. Ragged all-to-all uses 32 CTAs, 512 threads and rotated peers, without the minimum-block launch hint. QuACK PDL is disabled, optimizer state is offloaded to pinned host memory, and XLA uses an 85% memory-limit budget (`--xla_gpu_memory_limit_slop_factor=85`). Both arms use the same locked runtime and production transport.

Every arm independently resolved the latest completed model-and-optimizer checkpoint after GPU allocation. Exact checkpoint URI and step matched within every pair; all selected step 108195 on the same rack. All eight arms passed source and runtime checks, produced 28 finite losses and completed without failures or preemptions. Each two-step GPU0 profile contained 96 expected backward mainloops, zero from the other implementation, and 1,152 ragged launches at 32×512. The eight training arms ran sequentially, followed by one regional CPU analysis job. The longest 16-node GPU job lifetime, including setup, restore, compilation and teardown, was 843.168 seconds.

| Pair | Run order | Control seconds/step | Native seconds/step | Throughput gain |
| --- | --- | ---: | ---: | ---: |
| 1 | Control, native | 15.772236 | 14.966126 | 5.38623% |
| 2 | Native, control | 15.580979 | 14.886543 | 4.66486% |
| 3 | Control, native | 15.486554 | 15.036319 | 2.99431% |
| 4 | Native, control | 15.467988 | 14.974980 | 3.29221% |

Each arm ran 28 steps. Profiling covered relative steps 5–6; scoring retained all 20 relative steps 8–27, absolute steps 108203–108222. No slow sample was removed, no training pair was replaced and no additional replication was added after observing results. Two coordinator attempts failed before benchmark training began. [Per-step durations and derived statistics](https://github.com/mcwitt-agent/marin/blob/native-sm100-attention/experiments/benchmarks/fa4/native_sm100_results.json) preserve the measured sample.

Throughput gain is the ratio of the overall control and native mean step times minus one. The paired Fieller interval treats the four run pairs as independent replicates, with three degrees of freedom, and retains covariance between control and native means. Mean savings are 0.61095 seconds per step, with a paired Student-t 95% interval of 0.34342–0.87847 seconds. Both positive lower bounds meet the predeclared acceptance condition. A fixed sensitivity excluding absolute steps divisible by ten, when the training loop logs parameter and gradient statistics, gives +3.96% throughput with a 95% interval of +2.04%–5.91%.

## Validation and limits

Both b28 implementations passed the independent FP32 attention reference at `atol=rtol=0.07`. Six cases covered both KV-head counts, lengths 257 and 2,305, global/local attention, packed boundaries, padding and a fully padded batch sequence. Each case reused its compiled executable three times with changed inputs and boundaries. Padded outputs and gradients had to be exactly zero, including with nonzero padded cotangents. The [standalone gate](https://github.com/mcwitt-agent/marin/blob/ecf3cac2aafdd66403684180da4f595a54a2f813/experiments/benchmarks/fa4/native_sm100_check.py) reproduces these checks. The largest absolute control/native loss difference across all four pairs and all 28 steps was 0.001214.

Summed backward mainloop time across the 96 calls in each two-step GPU0 capture fell from 2.25–2.27 seconds to 0.91–0.92 seconds. Switching the backward integration also changed compiler lowering, scheduling and buffer lifetimes. Those changes are part of the treatment, so the whole-step gain cannot be attributed solely to the native mainloop.

The b28 compiler buffer assignments for GPU0 reserve 11.972082 GiB more default device memory than the control. This planned allocation total does not measure a runtime allocator peak. A separate historical b16 diagnostic measured peak default-allocator memory rising from 103.565529 to 115.537611 GiB, excluding the collective allocator. Runtime peak memory has not been remeasured on b28. No additional GPU memory query ran during the b28 acceptance arms.

Four short pairs on one checkpoint and rack give a conditional estimate. The intervals assume independent, approximately normal run-pair means; temporal drift or correlated rack conditions can violate those assumptions. Measure runtime memory at the intended deployment scale before promoting this path.

The [earlier b16 measurements](https://github.com/mcwitt-agent/marin/blob/1057edfe1862931bffd155953b92a4532f40edb2/experiments/benchmarks/fa4/native_sm100_results.json) gave +2.60% throughput with a 95% interval of +1.37%–3.85%. They are a separate study on the older dependency set. The b28 estimate above measures the incremental gain over the baseline that already includes #8741.

## Reproduce on the draft

On a physical SM100 GPU with the repository's locked GPU dependencies, run each reference gate in a fresh process:

```bash
uv run --all-packages --extra gpu python -m experiments.benchmarks.fa4.native_sm100_check --variant control --source-sha "$(git rev-parse HEAD)"
uv run --all-packages --extra gpu python -m experiments.benchmarks.fa4.native_sm100_check --variant native --source-sha "$(git rev-parse HEAD)"
```

Submit each training arm separately from a clean checkout. Set `checkpoint_root` to the live hero checkpoint root in the training region, then use a unique run ID and select `control` or `native`:

```bash
checkpoint_root='<regional hero checkpoint root>'
run_id="native-sm100-control-$(date -u +%Y%m%d-%H%M%S)"
uv run iris --cluster=marin job run --no-wait --enable-extra-resources \
  --target-cluster cw-us-east-08a --priority production \
  --cpu 4 --memory 16GB --disk 32GB --timeout 43200 --max-retries 0 \
  --sync-package marin-root --job-name "$run_id" -- \
  python -m experiments.benchmarks.hero_native_sm100 \
  --run-id "$run_id" --checkpoint "$checkpoint_root" --backward control \
  --num-steps 28 --warmup-steps 5 --profile-steps 2 --version dev --run
```

This submits a CPU coordinator with a 12-hour queue-inclusive deadline. The benchmark requests the 16-node GPU gang through the hero recipe and inherits the coordinator's priority. The GPU process has a separate 55-minute deadline after allocation. The four-pair study used one coordinator with a 48-hour deadline to allow queue waits for all eight sequential arms.

Run one arm or analysis job at a time. Record the observed checkpoint URI and step, runtime manifests and rack for each arm. Only compare matching pairs. Follow the alternating order and scoring window above, and retain failed or mismatched attempts separately.
