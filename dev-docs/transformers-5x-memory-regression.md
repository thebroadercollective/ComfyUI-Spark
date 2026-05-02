## Background

Upgrading `transformers` from 4.5.7 (as reported by user; verify against
`pip show` in the working venv — the transformers release scheme is
4.NN.M, so "4.5.7" may be a transcription of "4.57.x" or similar) to 5.x
caused explosive memory growth and OOM kills when loading HunyuanImage3
on the DGX Spark, with the same ComfyUI flags and checkpoint. Rolling
back to the working 4.x version restored working behavior. We cannot
upgrade until we understand what changed. Not blocking: the working 4.x
version is stable and all roadmap models load on it — this note exists
so the investigation state survives into future sessions.

### Observed signature (from user report)

During HunyuanImage3 load with Transformers 5.x, memory usage grew
explosively (significantly beyond the 4.x baseline) and OOM-killed the
process. User rolled back to 4.x and the problem went away. Exact peak
memory, exact OOM location in the load pipeline, and exact Transformers
5.x version are **not** captured — record these from a live reproduction
before bisection.

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
   instrumentation (T1 and T3 are the memory-instrumentation tasks from
   the same branch; see Cross-reference section for details).
4. Compare against a working-4.x run with an identical workflow and
   identical flags. Save both `memory_report` snapshots.

The key diagnostic is the `POST_MODEL_INIT` snapshot T3 adds near
`get_model()` / `load_model_weights()` in
`comfy/sd.py::load_state_dict_guess_config` (approximate location,
verify when T3 lands). If the regression is visible there, the extra
memory is being allocated by HF `from_pretrained`. If `POST_MODEL_INIT`
looks the same between the working 4.x and 5.x but a later snapshot
diverges, the regression is in ComfyUI's `load_clip` consumption path
reacting differently to what 5.x returns (e.g., dtype, device map, or
parameter layout differences).

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
  load weights in a different precision than the working 4.x version when
  the caller does not explicitly pass `torch_dtype`. On a checkpoint whose
  on-disk dtype differs from the intended runtime dtype, this can stage a
  full fp32 copy. Verify.
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

0. Confirm the actually-working transformers version by running
   `pip show transformers` in a venv where HunyuanImage3 loads
   successfully, and record the full output in a sidecar file committed
   next to this note. Do the same in the regressing venv. The "4.5.7"
   string in the user's report may be a transcription of "4.57.x" or a
   similar real release — the bisection must start from a verified pin.
1. Record exact versions of both environments: the working 4.x release
   and the regressing 5.x release. Also record `safetensors`,
   `accelerate`, `huggingface-hub`, `tokenizers`, and `torch` versions in
   both environments.
2. Walk minor versions with `pip install transformers==4.N.M` (and also
   the first 5.0.0 if it exists) to find the first version that shows
   the regression. Run the reproduction above at each step.
3. Inside the first regressing version, diff `modeling_utils.py` against
   the last working version — specifically `from_pretrained`, the
   sharded loader, and the accelerate bridge.
4. For each hypothesis above, try a targeted toggle to see whether it
   restores working-4.x memory behavior. Examples:
   - Explicitly pass `low_cpu_mem_usage=False` (and separately `=True`).
   - Explicitly pass `torch_dtype=torch.bfloat16`.
   - Explicitly pass `device_map=None`.
   - Pin `accelerate` back to the version that shipped alongside the
     working 4.x release.
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

T3 is scoped but may not be landed yet. Verify with
`grep POST_MODEL_INIT comfy/sd.py` before relying on the anchors below;
if missing, land T3 first. T1 (`memory_report`) has landed on this
branch and can be relied on today.

The instrumentation that will be most useful for this investigation:

- **T1 `memory_report` snapshots** — the before/after pairing around the
  loader call will show whether the growth is on the `from_pretrained`
  step or later in `load_clip`. Available today.
- **T3 load-path INFO logs** — the `POST_MODEL_INIT` snapshot T3 adds
  near `get_model()` / `load_model_weights()` in
  `comfy/sd.py::load_state_dict_guess_config` brackets `from_pretrained`
  specifically (approximate location, verify when T3 lands). This is
  the single most useful line for this investigation: if it diverges
  between the working 4.x and 5.x, the regression is inside HF; if it
  matches but a later snapshot diverges, the regression is on the
  ComfyUI side reacting to a different return shape.
- **T6 attention counters** — not directly relevant to loading, but
  worth capturing anyway since the same reproduction can exercise them.

When the bisection begins, capture both a working-4.x and 5.x timeline
and commit them next to this file as
`transformers-<version>-timeline.txt` (where `<version>` is the actual
string from `pip show`).
