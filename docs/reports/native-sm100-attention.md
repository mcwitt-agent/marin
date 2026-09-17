# Native SM100 attention backward

Native SM100 attention backward improved restored hero training throughput by **2.60%**, with a **95% paired confidence interval of 1.37%–3.85%**, in four predeclared run pairs. Mean step time fell from 15.5000 to 15.1072 seconds. A separate diagnostic measured 11.97 GiB more peak default-allocator memory.

These measurements used FlashAttention 4.0.0b16 and Marin research-branch commit `b95ec4a0688e854be416ae2f792ccf986ca39fb6`, which adds the benchmark adapter to cumulative baseline `a428f1d25d53ee825de801f15674a3a439420df4`. That original research commit is retained locally; this draft contains the extracted candidate and derived measurements. The draft implementation is rebased onto main `f113d5f0ed47b9cdfa6eb5c5b93a47c5d1ff08f2`, whose GPU dependency lock selects b28. The adapter now uses its `AuxData` argument and passes the softmax scale to preprocessing. GPU numerical checks and throughput measurements have **not been repeated on this rebased version**. Production attention dispatch remains unchanged.

## Implementation

The [benchmark adapter](../../experiments/benchmarks/fa4/native_sm100.py) uses the upstream SM100 backward kernel with 128×128 tiles, one CTA per cluster and two-CTA instructions disabled. It supplies packed-document bounds and validity through the custom mask and block-sparse metadata. Preprocessing, backward accumulation and postprocessing are separate JAX FFI calls. The launcher zeros all three FP32 gradient accumulators before each backward invocation, including reused buffers.

This path uses SM100 tensor memory and `tcgen05` matrix instructions. The measured control used the existing SM120-oriented warp-MMA path. The adapter supports BF16, head dimension 128 and query-to-KV head ratios four and eight. Select the implementation in a fresh process before JAX tracing: the context manager temporarily replaces the backend's backward entrypoint and does not invalidate previously compiled functions.

## Measured workload and protocol

Each arm ran the EP64 hero recipe on 64 GB200 GPUs in one NVLink rack: 16 nodes with four GPUs each, global batch 1,024 and sequence length 4,096. The model has 48 layers, hidden width 6,144, 48 query heads, 12 local or six global KV heads, head dimension 128, and 384 routed experts with top-eight routing. Local attention uses a 2,048-token window; every fourth layer uses global attention.

The cumulative baseline was Marin main `a428f1d25d53ee825de801f15674a3a439420df4` and XLA `708c3a4ec79c7581ffe28c4bb2be2b6dbf870519`. Ragged all-to-all used 32 CTAs, 512 threads, rotated peers and eight-load copy policy 5, without the minimum-block launch hint. QuACK PDL was disabled, optimizer state was offloaded to pinned host memory, and the memory-limit slop factor was 85.

Every arm independently resolved the latest completed model-and-optimizer checkpoint after GPU allocation. Exact checkpoint URI and step matched within every pair; all selected step 108195 on the same rack. All eight arms passed runtime/source checks, produced 28 finite losses and completed without failures or preemptions. Each two-step GPU0 profile contained 96 expected backward mainloops, zero from the other implementation, and 1,152 ragged launches at 32×512. All training and analysis jobs ran sequentially. The longest GPU-child lifetime, including restore and compilation, was 975.335 seconds.

| Pair | Run order | Control seconds/step | Native seconds/step | Throughput gain |
| --- | --- | ---: | ---: | ---: |
| 1 | Control, native | 15.548272 | 15.091321 | 3.02790% |
| 2 | Native, control | 15.477623 | 15.253441 | 1.46971% |
| 3 | Control, native | 15.530490 | 15.053477 | 3.16879% |
| 4 | Native, control | 15.443490 | 15.030605 | 2.74696% |

Each arm ran 28 steps. Profiling covered relative steps 5–6; scoring retained all 20 relative steps 8–27, absolute steps 108203–108222. No slow sample was removed, no pair was replaced and no additional replication was added after observing results. [Per-step durations and derived statistics](../../experiments/benchmarks/fa4/native_sm100_results.json) preserve the measured sample.

Throughput gain is the ratio of the overall control and native mean step times minus one. The paired Fieller interval treats the four run pairs as independent replicates, with three degrees of freedom, and retains covariance between control and native means. Mean savings are 0.39276 seconds per step, with a paired Student-t 95% interval of 0.20892–0.57659 seconds. Both positive lower bounds meet the predeclared acceptance condition. A fixed sensitivity excluding absolute steps divisible by ten, when the training loop logs parameter and gradient statistics, gives +2.50% throughput with a 95% interval of +1.13%–3.90%.

## Validation and limits

The b16 GPU reference gate checked both implementations against independent FP32 attention at unchanged `atol=rtol=0.07`. Six cases covered both KV-head counts, lengths 257 and 2,305, global/local attention, packed boundaries, padding and a fully padded row. Each case reused its compiled executable three times with changed inputs and boundaries. Padded outputs and gradients had to be exactly zero, including with nonzero padded cotangents. The [standalone gate](../../experiments/benchmarks/fa4/native_sm100_check.py) retains these checks for revalidation.

A separate 12-step diagnostic used the same one-rack recipe, checkpoint step 108195 and b16 runtime. JAX device memory statistics captured allocator peaks at explicit synchronization points after restoration, compilation and execution. It measured peak default-allocator memory rising from 103.565529 to 115.537611 GiB. This excludes the collective allocator. One native-only unsynchronized physical-memory query during pair 1's scored step 108205 found at least 9,129 MiB free per GPU at that instant. It does not bound peak memory and may affect that step's timing; the step is retained. No other arm received this query.

Four short pairs on one checkpoint and rack give a conditional estimate. The intervals assume independent, approximately normal run-pair means; temporal drift or correlated rack conditions can violate those assumptions. Switching the backward integration also changed compiler lowering, scheduling and buffer lifetimes; these were consequences of the treatment. Thus the whole-step gain cannot be attributed solely to the native mainloop. Revalidate correctness, memory and throughput on the rebased dependency set before promoting this path.

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
  --num-steps 28 --warmup-steps 5 --profile-steps 2 --version "$run_id" --run
```

This submits a CPU coordinator with a 12-hour queue-inclusive deadline. The benchmark requests the 16-node GPU gang through the hero recipe and inherits the coordinator's priority. The GPU process has a separate 55-minute deadline after allocation. The runner uses current main's locked runtime and production transport, replacing the research harness's temporary wheel installation and copied transport.

Run one arm or analysis job at a time. Record the observed checkpoint URI and step, runtime manifests and rack for each arm. Only compare matching pairs. Follow the alternating order and scoring window above, and retain failed or mismatched attempts separately. The historical measurements do not establish the performance of these reproduction commands on b28.
