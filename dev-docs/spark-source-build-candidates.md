## Context

The DGX Spark is GB10 (sm_121) with 128GB unified LPDDR5x. Stock PyPI
wheels target a broad arch set and often rely on PTX/JIT fallbacks for
newer SMs, which can add startup overhead and binary bloat. The user has
already built `sageattention` from source (with Triton disabled due to
reported stability problems on SM121) and wants a scoping list of what
else might be worth building from source.

This note is a scoping document, not a build guide. Each candidate has a
one-line purpose, a cost/benefit sketch, and a recommendation.

## Candidates

### 1. Flash Attention 3

- **Purpose.** Newer attention kernel library competitive with sageattention
  on some shapes; supposedly better on long-sequence forward passes.
- **Cost/benefit on Spark.** SM121 support is uncertain — FA3 is primarily
  targeted at Hopper/SM90+, and it is not clear whether the Blackwell-class
  SM121 path is compiled or falls back. Build cost is moderate (CUDA
  compile, minutes to tens of minutes). Benefit is unknown until we know
  whether sageattention is leaving performance on the table.
- **Recommendation. Defer** until the T6 attention counters show whether
  sageattention is silently falling back to the math kernel on any workload
  or whether a specific kernel is a bottleneck. No point optimizing a path
  that never triggers.

### 2. xformers

- **Purpose.** Older alternative attention/fused-op library.
- **Cost/benefit on Spark.** Largely superseded by PyTorch native SDPA and
  by sageattention for the kernels we care about. Building is non-trivial.
  No known win for the current model set.
- **Recommendation. Skip** unless a specific model family is found whose
  cross-attention path only xformers covers.

### 3. PyTorch itself (`TORCH_CUDA_ARCH_LIST=12.1`)

- **Purpose.** Stock torch wheels target a broad arch list and include PTX
  fallbacks that JIT at load for unsupported SMs. A Spark-targeted build
  could marginally reduce binary size and avoid any load-time PTX compile.
- **Cost/benefit on Spark.** Build cost is very high: building PyTorch from
  source is multi-hour, and the build has to be repeated on every torch
  upgrade (which happens often). Benefit is speculative — we do not yet
  know whether any measurable fraction of current startup time is PTX JIT.
- **Recommendation. Defer.** Revisit only if profiling shows a noticeable
  JIT compile phase at model load.

### 4. bitsandbytes

- **Purpose.** INT8 / INT4 quantization kernels used by many HF checkpoints.
- **Cost/benefit on Spark.** Only relevant if we actually plan to load
  bitsandbytes-quantized weights. The Hunyuan INT8 build the user
  mentioned uses its own loader path and is orthogonal to bnb. Building
  from source is moderate effort but only pays off if bnb is on the
  critical path.
- **Recommendation. Defer.** Pursue only once a bnb-quantized checkpoint
  is on the roadmap.

### 5. Triton

- **Purpose.** Kernel compilation framework backing many custom attention
  and elementwise ops.
- **Cost/benefit on Spark.** The user explicitly disabled Triton for the
  sageattention build due to reported SM121 stability issues. Until
  upstream Triton stability improves, a source build changes nothing we
  care about.
- **Recommendation. Skip** until upstream reports confirm SM121 stability.

### 6. cuDNN / cuBLAS

- **Purpose.** NVIDIA's closed-source kernel libraries; bundled with torch.
- **Cost/benefit on Spark.** These are not rebuildable from source in any
  practical sense. Nothing to do here.
- **Recommendation. Skip.**

### 7. NCCL

- **Purpose.** Multi-GPU/multi-node collective communication.
- **Cost/benefit on Spark.** Spark is single-GPU GB10. No collectives to
  optimize.
- **Recommendation. Skip.**

### 8. safetensors

- **Purpose.** Safetensors file format reader, Rust-backed.
- **Cost/benefit on Spark.** Already fast and already central to the
  loader. No known path where a custom build would help — the unified
  memory loading work in this repo lives above the safetensors layer, not
  inside it.
- **Recommendation. Skip.**

## What I'm NOT including and why

- **xformers-flash / flash-attn-v2** — strictly older than FA3; if FA3 is useful, v2 is subsumed.
- **DeepSpeed** — distributed training/inference runtime; single-node Spark does not need it.
- **Megatron-LM** — distributed training framework; irrelevant to single-node inference.
- **Apex** — legacy NVIDIA mixed-precision helpers, absorbed into upstream PyTorch.
- **TensorRT / TensorRT-LLM** — high effort to integrate into ComfyUI; a different project.

## Prioritization

| Candidate | Priority | Trigger to revisit |
|---|---|---|
| Flash Attention 3 | Low | T6 counters show sage fallbacks > 0 or a specific kernel bottleneck |
| PyTorch source build | Low | Measurable JIT compile overhead observed at model load |
| bitsandbytes | Low | A bitsandbytes-quantized checkpoint enters the roadmap |
| Triton | Skip | Upstream stability improves on SM121 |
| xformers | Skip | A model family needs an xformers-only kernel |
| cuDNN / cuBLAS | Skip | N/A (not source-buildable) |
| NCCL | Skip | Spark becomes multi-GPU (it will not) |
| safetensors | Skip | Evidence that the Rust loader itself is a bottleneck |

The default posture is "don't build from source unless there is a
measured reason to." Sageattention was worth it because the workload
visibly needed it; everything above should wait for a comparable signal.
