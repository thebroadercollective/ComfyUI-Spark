## Background

Upgrading `transformers` from 4.5.7 to 5.x caused explosive memory growth and
OOM kills when loading HunyuanImage3 on the DGX Spark. Rolling back to 4.5.7
restored working behavior with the same ComfyUI flags and the same checkpoint.
We cannot upgrade `transformers` until we understand what changed.

This is not blocking any current workflow: 4.5.7 is stable, all models
currently on the roadmap load and run on it. The note exists so the
investigation state survives into future sessions.

## Reproduction plan

1. Create a clean venv alongside the working one, install the latest 5.x
   release pinned exactly (record the version).
2. Launch ComfyUI-Spark with the standard Spark flags:

   ```
   uv run python main.py --listen 0.0.0.0 \
     --disable-dynamic-vram --reserve-vram 1 --disable-pinned-memory \
     --disable-mmap --dont-upcast-attention \
     --bf16-unet --bf16-vae --bf16-text-enc
   ```

3. Load HunyuanImage3. Capture a memory timeline using the T1/T3
   instrumentation (see "Cross-reference" below).
4. Compare against a 4.5.7 run with an identical workflow and identical
   flags. Save both `memory_report` snapshots.

The key diagnostic is the `POST_MODEL_INIT` snapshot added by T3 at
`comfy/sd.py:1668`. If the regression is visible there, the extra memory is
being allocated by HF `from_pretrained`. If `POST_MODEL_INIT` looks the same
between 4.5.7 and 5.x but a later snapshot diverges, the regression is in
ComfyUI's `load_clip` consumption path reacting differently to what 5.x
returns (e.g., dtype, device map, or parameter layout differences).

## Candidate causes (hypotheses to bisect)

All items below are hypotheses based on general themes in HF transformers
development; none are confirmed against the 5.x changelog yet. Verify each
against release notes before acting on it.

- **`low_cpu_mem_usage` default flip.** Hypothesis: 5.x may have changed the
  default of `low_cpu_mem_usage` on `from_pretrained` (either toggled it on
  where it was off, or vice versa). A wrong default on a large multi-shard
  checkpoint can easily double peak memory during load. Verify against 5.x
  release notes.
- **`torch_dtype="auto"` or implicit dtype behavior.** Hypothesis: 5.x may
  load weights in a different precision than 4.5.7 when the caller does not
  explicitly pass `torch_dtype`. On a checkpoint whose on-disk dtype differs
  from the intended runtime dtype, this can stage a full fp32 copy. Verify.
- **Safetensors sharded-loader changes.** Hypothesis: 5.x may have refactored
  the shard reader (e.g., to stage shards in a different order, or to keep
  shard metadata alive longer). Any change that holds a shard buffer past the
  tensor materialization point doubles its footprint temporarily. Verify by
  diffing `modeling_utils.py` and `modeling_utils_shard_loading.py` between
  the last working 4.x and the first regressing 5.x.
- **`accelerate` integration / `device_map` default.** Hypothesis: 5.x may
  have changed how it interacts with `accelerate` (version pin, default
  `device_map`, or meta-device init). If a module that used to init on CPU
  now inits on CUDA, or vice versa, peak memory changes. Verify.
- **Eager init vs meta-device init.** Hypothesis: 5.x may have flipped
  whether `from_pretrained` constructs parameters on the real device or on
  the meta device before streaming weights in. Eager init of a large
  model on a real device is roughly a full extra allocation on top of the
  checkpoint. Verify.
- **Safetensors minimum version bump.** Hypothesis: 5.x may have pulled in a
  newer `safetensors` that changed its own mmap or staging behavior. Worth
  checking transitive dep pins as part of the bisection. Verify.

## Bisection steps

1. Record exact versions: working `transformers==4.5.7` and the regressing
   `transformers==5.x.y`. Also record `safetensors`, `accelerate`, `huggingface-hub`,
   `tokenizers`, and `torch` versions in both environments.
2. Walk minor versions with `pip install transformers==4.N.M` (and also the
   first 5.0.0 if it exists) to find the first version that shows the
   regression. Run the reproduction above at each step.
3. Inside the first regressing version, diff `modeling_utils.py` against the
   last working version — specifically `from_pretrained`, the sharded
   loader, and the accelerate bridge.
4. For each hypothesis above, try a targeted toggle to see whether it
   restores 4.5.7-like memory behavior. Examples:
   - Explicitly pass `low_cpu_mem_usage=False` (and separately `=True`).
   - Explicitly pass `torch_dtype=torch.bfloat16`.
   - Explicitly pass `device_map=None`.
   - Pin `accelerate` back to the version that shipped alongside 4.5.7.
5. Record each attempt and its memory outcome in this file.

## Outcome action

- As findings emerge, append them to this file under a "Findings" section.
- If the cause is upstream, open an issue on
  https://github.com/huggingface/transformers with a minimal reproduction
  (ideally reduced to a single `from_pretrained` call on HunyuanImage3
  without ComfyUI in the loop).
- If the cause is in how ComfyUI calls `from_pretrained`, file a follow-up
  task in this repo and link it here.

## Cross-reference to T1/T3 instrumentation

The instrumentation that will be most useful for this investigation:

- **T1 `memory_report` snapshots** — the before/after pairing around the
  loader call will show whether the growth is on the `from_pretrained` step
  or later in `load_clip`.
- **T3 load-path INFO logs** — the `POST_MODEL_INIT` snapshot at
  `comfy/sd.py:1668` brackets `from_pretrained` specifically. This is the
  single most useful line for this investigation: if it diverges between
  4.5.7 and 5.x, the regression is inside HF; if it matches but a later
  snapshot diverges, the regression is on the ComfyUI side reacting to a
  different return shape.
- **T6 attention counters** — not directly relevant to loading, but worth
  capturing anyway since the same reproduction can exercise them.

Capture both a 4.5.7 baseline timeline and a 5.x reproduction timeline and
commit them next to this file (as `transformers-5x-timeline-457.txt` and
`transformers-5x-timeline-5xy.txt`) when the bisection begins.
