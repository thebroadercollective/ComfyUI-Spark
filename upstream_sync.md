# Upstream sync

Snapshot HEAD first so a bad rebase is recoverable:

```bash
git branch backup/pre-rebase-$(date +%Y%m%d) HEAD
git tag pre-rebase-$(date +%Y%m%d) HEAD
```

Then sync:

```bash
git fetch upstream
git rebase upstream/master
```

## Worktree must be clean

If `models/` is a symlink to a local model collection (e.g. `/home/ai/ComfyUI/models`), the tracked placeholder files (`models/*/put_*_here`) show as deletions in `git status`, and `git rebase --autostash` will fail with `'models/audio_encoders/...' is beyond a symbolic link`. Materialize the tracked placeholders before rebase, restore the symlink after:

```bash
# Save the target so you can restore it
SYMLINK_TARGET=$(readlink models)

# Replace symlink with real directories from git
rm models
git checkout -- models/

# ... run the rebase ...

# Restore symlink
rm -rf models
ln -s "$SYMLINK_TARGET" models
```

Any *other* uncommitted tracked changes (e.g. edits to this file) also block the rebase. Stash them by pathspec first and restore after — independent of the `models` dance:

```bash
git stash push -m wip <path>   # e.g. upstream_sync.md
# ... rebase, restore symlink ...
git stash pop
```

## Expected conflict

Conflicts have shown up in two files. `model_patcher.py` did **not** conflict on the 2026-05-28 sync (it auto-merged clean), and the 2026-06-06 sync (v0.22.0 → v0.24.1, 60 upstream commits) replayed all 23 fork commits with **zero** conflicts — so don't assume conflicts, but verify the invariants below regardless. On 2026-06-06 upstream's changes to the sensitive files were all aimdo/dynamic-pin internals plus new model support (TripoSplat, Ideogram4); it also removed the `is_dynamic()` gate in `free_memory`'s smart-memory branch (see the updated CLAUDE.md note — behavior is strictly milder, no fork action needed). The 2026-06-11 sync (38 upstream commits, version stayed at 0.24.0) likewise replayed all 24 fork commits with **zero** conflicts; upstream's touches to the sensitive files were again aimdo/dynamic internals (`ops: tolerate already force casted dynamic weight`, `mm: dont reset cast buffers in cleanup_models_gc()`, `main: force cudnn.benchmark to false`) plus new model support (SCAIL-2, Depth Anything 3, Bernini-R Wan video, SeedVR2) — none touched the fork's unified-memory invariants. Verified post-rebase: `restore_loaded_backups` still branches on `bk.invertible`, the streaming loader + `should_assign_weights()` are intact, ruff is clean, and the streaming-loader + ram-pressure regression tests are green. The 2026-06-18 sync (v0.24.0 → v0.25.0, 41 upstream commits) again replayed all 26 fork commits with **zero** conflicts. Sensitive-file touches were all additive new-model support (Boogu-Image, SCAIL-2 multireference, Qwen3-VL text generation + Qwen3-VL-as-flux2-klein-TE — all in `model_base.py`/`sd.py`) plus one new flag, `--high-ram` (`d7a55272`): it only short-circuits the dynamic-pin path (`ensure_pin_budget`/`pinned_hostbuf_size`, dead on the fork with `--disable-dynamic-vram` → `is_dynamic()` False) and forces `--cache-classic` — so like the other vram-mode flags it is **inert/wrong on the Spark; do NOT add it to the launch command**. Verified post-rebase: `bk.invertible` branches (incl. `restore_loaded_backups`), streaming loader, and `should_assign_weights()` all intact; ruff clean; 13/13 streaming-loader + ram-pressure regression tests green. The 2026-06-27 sync (v0.25.0 → v0.26.0, 40 upstream commits) again replayed all 27 fork commits with **zero** conflicts. Sensitive-file touches were all additive feature/model support (int8 model support + int8-LoRA requant fix + faster/Turing int8, Krea2, LTX2 Context-Windows/IC-LoRa — `1a510f04`, `470ac36a`, `2a610155`, `cd77c551`); none touched the fork's invertible fast-path or unified-memory invariants. Two things to watch: (1) **`--disable-dynamic-vram` is now on upstream's deprecation path** — `833bfb57` rewrote its warning to say the arg "will be removed soon" and to recommend native fp8 formats over disabling. The flag and `enables_dynamic_vram()` gating still work (and a *new* `--enable-dynamic-vram` flag defaults off, so the fork's disable still wins via `main.py:239`), but this flag is load-bearing for the fork's loader — re-verify the branch selection in `load_torch_file()` if a future sync removes it. (2) The int8 commits surface a soft warning at import — `comfy_kitchen` 0.2.9 (what upstream's own `uv.lock` pins) lacks `TensorWiseINT8Layout`, so "fp8 and fp4 support will not be available". Not fork-specific (upstream's locked env has the same gap; their vestigial `requirements.txt` lists 0.2.14) and unused on the Spark — left as-is, no dep bump during the sync. Verified post-rebase: all `bk.invertible` branches (incl. `restore_loaded_backups`) intact, streaming loader + `should_assign_weights()` intact, ruff clean, 13/13 regression tests green. The 2026-07-01 sync (v0.26.0 → v0.27.0, 16 upstream commits) again replayed all 30 fork commits with **zero** conflicts. The *only* upstream touch to any sensitive file was one line in `comfy/ops.py` (`79c555ce`, "Fix int8 mm being skipped on offloaded lora weights": `want_requant=want_requant` → `want_requant=True`), which auto-merged cleanly with the fork's `.cpu().numpy()` edits — the hunks are in different regions of the file (upstream ~line 1216, fork ~1062/1446). `model_patcher.py`, `utils.py`, `sd.py`, `model_management.py`, `model_base.py`, and `cache_policy.py` were **untouched** by upstream this cycle. Everything else was additive/non-sensitive: a `ConditioningMultiply` node (`nodes.py`), a Qwen3-VL custom-embeddings tokenizer-crash fix (`text_encoders/qwen3vl.py`), Google partner nodes (Gemini Video Omni, Nano Banana 2 Lite), `--enable-asset-hashing` (opt-in, off by default), AGENTS.md, frontend 1.45.20, embedded docs v0.5.6, workflow templates, and a team-gated Cursor-review CI workflow. Two upstream commits billed as int8 fixes — `Fix memory leak related to int8.` (`d395813b`) and `Better and faster int8 lora applying.` (`78514105`) — are **`requirements.txt`-only** bumps; the real change lives in the `comfy_kitchen` dependency, and the fork doesn't drive the vestigial `requirements.txt`, so no action. The `comfy_kitchen` 0.2.9 / `TensorWiseINT8Layout` soft import warning persists unchanged (still upstream-lock-vs-code lag, unused on the Spark — no dep bump). Verified post-rebase: all `bk.invertible` branches (incl. `restore_loaded_backups`) intact, streaming loader + `should_assign_weights()` intact, `ops.py` carries both the upstream and fork edits, ruff clean, 13/13 streaming-loader + ram-pressure regression tests green. The 2026-07-05 sync (version stayed at 0.27.0, 15 upstream commits) replayed all 40 fork commits with **one** conflict — a *new* shape not seen before: upstream added `CLAUDE.md` as a **symlink to `AGENTS.md`** (`3fe9f5fe`, #14757), which collides with the fork's regular-file `CLAUDE.md` (the Spark instructions) at the first fork commit that writes it (`chore(spark): personal config`). Git can't 3-way-merge distinct types, so it reports `CONFLICT (distinct types)` and splits the path (`added by us` = upstream's symlink, `added by them` = the fork's file renamed to `CLAUDE.md~<hash> (<subject>)`). Resolution: keep the fork's regular file — see the new `### CLAUDE.md — file-vs-symlink type conflict` subsection below. **Zero** sensitive-file touches this cycle (`model_patcher.py`/`utils.py`/`sd.py`/`model_management.py`/`model_base.py`/`cache_policy.py`/`ops.py` all untouched by upstream across `2c935de1..985fb9d6`); the rest was security fixes (GHSA-779p-m5rp-r4h4, partner-node auth-header masking), several `AGENTS.md` iterations, partner-node cleanup (StabilityAI + IdeogramV1/V2 nodes removed, ByteDance Seed Audio 1.0 added), a color-picker transparency fix, and an embedding-with-llama_template fix. Verified post-rebase: all `bk.invertible` branches (incl. `restore_loaded_backups`) intact, streaming loader + `should_assign_weights()` intact, ruff clean, and 31/31 streaming-loader + ram-pressure + mixed-dtype-assign regression tests green. The 2026-07-10 sync (v0.27.0 → v0.27.1, 32 upstream commits) replayed all 52 fork commits with **three** conflicts, two of them *new* loci: **(1) `comfy/ldm/modules/attention.py`** — upstream reworked `attention_sage`: it added `SAGE_ATTENTION_SUPPORTS_MASK` (probed via `inspect.signature`) and a `mask is not None and not SAGE_ATTENTION_SUPPORTS_MASK` clause on the low-precision fallback, and replaced the positional `sageattn(q,k,v,attn_mask=…,is_causal=…,tensor_layout=…)` call with a `sage_kwargs` dict that also carries `sm_scale`/`smooth_k`. This collided with the fork's `feat(attention)` runtime-verification commit (the `global _sage_first_call_logged/_sage_call_count/_sage_fallback_count` decl + first-call `logging.info` + `_sage_call_count += 1`). Resolution: take **upstream's** condition and `out = sageattn(q, k, v, **sage_kwargs)` call wholesale, re-insert only the fork's `global` line (right after the `def`) and the logging/counter block (right after the call). Same add-logging-onto-upstream-logic pattern as the `sd.py` conflicts. **(2) `comfy_execution/caching.py`** — upstream added `RAM_CACHE_LARGE_INTERMEDIATE` and an `all_outputs_dynamic()` helper plus a new eviction guard (`if all_outputs_dynamic(cache_entry.outputs) and used_generation[key] == generation: continue`, protecting current-gen entries whose outputs are all `is_dynamic()`), colliding with the fork's `_outputs_contain_model_patcher()` helper + unified-memory `protect_resident_models` guard. Both helpers and **both** guards must coexist (they're complementary: upstream's protects *dynamic* outputs; the fork's protects any current-gen `ModelPatcher` on unified memory — needed precisely because `--disable-dynamic-vram` makes real patchers non-dynamic, so `all_outputs_dynamic` returns False and never fires on the fork). Keeping both required a one-line **test** fix too: `test_ram_pressure_model_protection.py`'s `MagicMock(spec=ModelPatcher)` returns a *truthy* `is_dynamic()`, which trips the new upstream guard and wrongly protected the model on the non-unified test — pinned `m.is_dynamic.return_value = False` in the fixture to match fork reality (commit `test(spark): pin mock is_dynamic()=False …`). **(3) `CLAUDE.md`** — this cycle it was a **modify/delete** (not the 2026-07-05 distinct-types symlink): upstream *removed* its `CLAUDE.md` symlink entirely, so the base has no `CLAUDE.md` (only `AGENTS.md`), and the fork's `chore(spark): personal config` commit modifies it → `CONFLICT (modify/delete): CLAUDE.md deleted in HEAD and modified in <fork-commit>`. Git leaves the fork's regular file on disk (`DU` status); resolve by `git add CLAUDE.md` (keep the fork file — same intent as the type-conflict subsection below, simpler mechanics). **Zero** touches to `model_patcher.py`/`sd.py`/`utils.py`/`model_management.py`/`model_base.py`/`cache_policy.py`/`ops.py` this cycle; the rest was additive node/model support (Save 3D Advanced, Save Text, Create Bounding Boxes bboxes input, etc.). Verified post-rebase: all `bk.invertible` branches (incl. `restore_loaded_backups`) intact, streaming loader + `should_assign_weights()` intact, GB10 gates (`spark_defaults.enabled()` in `cli_args.py`/`cache_policy.py`) intact, ruff clean, and 97/97 streaming-loader + ram-pressure + mixed-dtype-assign + spark-defaults + dynamic-vram-gate + spark-consumer + cache-policy tests green. The 2026-08-16 sync (v0.27.1 → **v0.33.0**, **207** upstream commits — the largest cycle so far) replayed all 54 fork commits with **five** conflicts, four of them in already-known loci plus one new file. **(1) `comfy/text_encoders/flux.py`** — upstream reworked `load_mistral_tokenizer` to return `{"tokenizer_object": from_tekken_json(data)}`; keep upstream's body and re-apply only the fork's `.cpu()` before `.numpy()` (a rebase-breadcrumb comment now marks it). **(2) `comfy/model_management.py` (`load_models_gpu` tail)** — upstream added a `detail("Model loaded: patcher=… ram_mb=… vram_mb=…")` line where the fork emits `MODEL_GPU_READY`; keep **both**, upstream's first. **(3) `comfy/model_patcher.py`** — pure add/add at the top of the module: upstream added `is_model_patcher_output()` + `PromptModelTracker` exactly where the fork defines `WeightBackup` + `_patches_are_invertible`; keep both blocks. Note upstream **no longer has any module-level `WeightBackup`** (it still builds the backup namedtuple inline in `patch_weight_to_device`), so the fork's hoisted definition is now unopposed. **(4) `comfy/text_encoders/lt.py`** — upstream dedented the `text_embedding_projection` load out of the `if "model.layers.0…" in sd:` branch and added an `sdo` prefix-replace; take upstream's structure, then hoist the fork's `assign = getattr(self, "can_assign_sd", False)` + `normalize_component_if_needed()` closure to that same dedented level so the audio/video-connector loads below can still use them. **(5) `main.py`** — upstream split logger setup into `get_console_log_level`/`get_file_log_outputs`; keep upstream's three lines and re-append `comfy.spark_defaults.log_summary()`. `comfy/pinned_memory.py` conflicted too (upstream reworked `_pin`→`_pins[subset]` dicts): keep upstream's structure, swap `args.disable_pinned_memory` → `comfy.model_management.pinned_memory_disabled()` and drop the now-unused `from comfy.cli_args import args`. **Two post-rebase fixups were required** (both committed on top): (a) upstream removed `caching.py`'s `from comfy.model_patcher import ModelPatcher` in favour of `is_model_patcher_output()`, leaving the fork's `_outputs_contain_model_patcher()` referencing an undefined name (ruff F821 caught it) — switched to the upstream helper, which is a strict superset; (b) **dependency floors**: `comfy/ldm/modules/attention.py` now calls `comfy_kitchen.int8_attention_is_available()` at import time (needs ≥0.2.31; the stale lock had 0.2.18 → hard `AttributeError` on `import comfy.model_patcher`) and `comfy_extras/nodes_glsl.py` imports the new `comfy_angle` package — both added/raised in `pyproject.toml` (`uv.lock` is untracked here, so a stale lock is the failure mode to watch after a big sync: run `uv lock --upgrade-package <pkg>` / `uv sync`, then re-install the ad-hoc `pytest*` and editable `./SageAttention`, which `uv sync` prunes). Upstream's memory/pinning machinery moved a lot this cycle (`c0117553` "Load weights to process RAM with MRU policy using pinning infrastructure", `e377e263` don't over-pin on swapless Linux, `de5625a6` delayed dynamic-pin cleanup, `611f2a4e` pin-registration priority, `6b30dc20` dynamic VRAM no longer disabled on WSL, `a7365071` dynamic VRAM on by default on ROCm ≥7.14, `cbbc9dab` `cast_bias_weight` context manager, `bf4c9a08` comfy-kitchen attention, `b53e247c` `comfy/logging.py`→`comfy/internal_logging.py`) — none of it conflicted, but it is the right place to look first if unified-memory behavior regresses. Verified post-rebase: all `bk.invertible` branches (incl. `restore_loaded_backups`) intact, streaming loader + `should_assign_weights()` intact, GB10 gates (`spark_defaults.enabled()` in `cli_args.py`/`cache_policy.py`/`model_management.py` + the `main.py` aimdo self-check) intact, ruff clean, **1367/1367** `tests-unit/` tests green, and a live `--quick-test-for-ci` startup on the Spark logs the auto-defaults line, `Unified memory system detected`, `SPARK: reserve-vram auto-set to 1GB`, `SPARK: pinned memory auto-disabled`, and **no** aimdo-regression warning. The 2026-08-28 sync (v0.33.0 → **v0.34.0** +13, 56 upstream commits) replayed all 62 fork commits with **two** conflicts, both in `comfy/utils.py` and both trivial **add/add** at the same spot — the module-level block just above `load_safetensors`. (1) Upstream `a3e5cce5` ("friendly errors to the dynamic vram safetensors loader") added `_SAFETENSORS_MAX_HEADER_SIZE` + `_invalid_safetensors_error()` / `_incomplete_safetensors_error()` exactly where the fork defines `STREAMING_LOAD_THRESHOLD` / `STREAMING_DROP_INTERVAL` / `_SAFETENSORS_DTYPES`; (2) the next fork commit adds `_sys_mem_str()` in the same region against an empty HEAD side. Resolution both times: **keep both blocks, upstream's first**, and mind the two-blank-line spacing (ruff E302). Upstream's commit is otherwise a pure refactor — it only swaps the two inline `raise ValueError(...)` bodies in `load_torch_file`'s exception handler for the new helpers, and the rest of it lives in `load_safetensors` (the aimdo-only path, dead on GB10) — so the fork's branch-selection block and streaming/`safe_open` fallback merged untouched.

Sensitive-file touches were unusually few this cycle. The one that matters: **`77739723` "respect the container cgroup memory limit instead of host RAM"** added `comfy/system_memory.py` and made it the single source of truth for RAM accounting, replacing direct `psutil.virtual_memory()` reads in `get_total_memory`/`get_free_memory`/`ensure_pin_budget`/`should_free_pins_for_ram_pressure` (`model_management.py`), `RAMPressureCache.ram_release` (`comfy_execution/caching.py`) and the `ram_shortfall` computation (`execution.py`). Two fork consequences: (a) the fork's RAM-pressure regression test monkeypatched `caching.psutil`, which no longer exists — three tests died with `AttributeError`; fixed by patching `caching.virtual_memory_available` instead (commit `test(spark): patch virtual_memory_available …`). The fork's `_outputs_contain_model_patcher` guard itself was untouched and still coexists with upstream's `all_outputs_dynamic()` guard. (b) `cache_policy._watermark_triggered()` was still reading `psutil` directly, so `--cache-drop-threshold-gb` would have disagreed with every other memory gate under a cgroup cap — routed through `comfy.system_memory.virtual_memory_available()` (commit `fix(cache-policy): …`). **All of this is a no-op on the Spark**: bare metal, `cgroup_memory_limit()` returns `None`, `virtual_memory_available()` is a straight passthrough to psutil (verified live). `memory_report()` / `_sys_mem_str()` deliberately keep their direct psutil reads — they report the host free/avail/used triple and `system_memory` exposes only total/available.

The only other sensitive touches: **`2504e68d` (MiniMax-H3 PDD LoRA)** added `or comfy.lora.calculate_shape(...) != m.weight.shape` to the four `force_patch_weights` branches in `ModelPatcher.load()`/`partially_load()`, i.e. shape-changing LoRAs now force a real `patch_weight_to_device()` instead of a `LowVramPatch`. Harmless twice over on the fork: those branches sit under `lowvram_possible`, which never fires given the unconditional unified full-load, and `_patches_are_invertible()` already rejects shape mismatches (`diff.shape != weight_shape`, plus the `LoRAAdapter` slot-5 reshape-spec check), so a PDD LoRA can never reach the in-place fast path — it takes the legacy copy path, which is correct. `sd.py` (`8e869efc` taeh3 VAE branch, `0e65cb90` Pixal3d/TRELLIS2) and `model_base.py` (`ff6c8a8a` MiniMax-H3 noise masks) were additive new-model support. `main.py`/`cli_args.py` saw only Windows multi-GPU work (`5ab2f7a2`, `3422dfdf` — both under `os.name == "nt"`, inert on Linux), a Python-3.10-EOL warning (`2618d32d`), and a database-path fix (`5f0c4e18`). `ops.py`, `pinned_memory.py`, and `ldm/modules/attention.py` were **untouched** by upstream. Dependency reconciliation: frontend 1.49.6 → **1.51.9**, workflow-templates 0.11.41 → **0.11.50**, `av` floor → **>=17.0.0** (`b78cec87`); `comfy-aimdo` deliberately left at the installed 0.4.10 despite upstream's 0.4.15 bump — aimdo is off on GB10 so only its import-time surface matters, and startup now prints upstream's "lower than the recommended version" nag, which is expected and inert here. Verified post-rebase: all `bk.invertible` branches (incl. `restore_loaded_backups`) intact, streaming loader + `should_assign_weights()` intact, all GB10 gates (`spark_defaults.enabled()` in `cli_args.py`/`cache_policy.py`/`model_management.py` + the `main.py` aimdo self-check) intact, ruff clean, **1574/1574** `tests-unit/` tests green, SageAttention rebuilt from source, and a live `--quick-test-for-ci` startup logs the auto-defaults line, `Unified memory system detected`, `SPARK: reserve-vram auto-set to 1GB`, `SPARK: pinned memory auto-disabled`, no cgroup-limit line, and **no** aimdo-regression warning.

### CLAUDE.md — file-vs-symlink type conflict (seen 2026-07-05)

Upstream (`3fe9f5fe`, #14757) ships `CLAUDE.md` as a **symlink → `AGENTS.md`** (mode `120000`) so agent tools pick up its engineering-guidelines file. The fork keeps `CLAUDE.md` as its own **regular file** (the Spark instructions). At the first replayed fork commit that writes `CLAUDE.md` (`chore(spark): personal config …`), git sees a regular file on one side and a symlink on the other, cannot 3-way-merge distinct types, and emits:

```
CONFLICT (distinct types): CLAUDE.md had different types on each side; renamed one of them so each can be recorded somewhere.
```

It records both under different index paths (in a rebase, "us" = the upstream base you're replaying onto, "them" = the fork commit):

- `added by us:   CLAUDE.md`  ← upstream's symlink (on disk, `CLAUDE.md -> AGENTS.md`)
- `added by them: CLAUDE.md~<hash> (<commit subject>)`  ← the fork's regular file, renamed aside

Resolve by keeping the fork's regular file — pull it from the commit git names in the conflict, then drop the rename artifact:

```bash
git checkout <fork-commit> -- CLAUDE.md               # overwrites the symlink with the regular file AND stages it
git rm --cached --force -- "CLAUDE.md~<hash> (…)"     # unstage the rename artifact
rm -f -- "CLAUDE.md~<hash> (…)"                       # remove its worktree copy
git rebase --continue
```

The replayed commit records `mode change 120000 => 100644 CLAUDE.md` — that's the intended flip back to a regular file. **Only this one commit conflicts;** once `CLAUDE.md` is a regular file again the rest of the fork's docs commits patch it normally. Two things NOT to do: (1) don't merge `AGENTS.md`'s content into `CLAUDE.md` — `AGENTS.md` is upstream's separate, non-conflicting file and survives on its own; the fork simply declines to symlink `CLAUDE.md` at it. (2) Ignore any harness "CLAUDE.md was modified" notice that fires mid-conflict — during the conflict the on-disk `CLAUDE.md` is transiently the symlink, so tools read `AGENTS.md`'s content; the resolution restores the Spark file.

### `comfy/model_patcher.py` — invertible LoRA fast-path

Two shapes have shown up so far:

1. **`patch_weight_to_device` signature churn** — keep the fork's invertible branching in `force_load_param` and thread any new upstream parameters (e.g. `force_cast=True`) through the `self.patch_weight_to_device(...)` call.
2. **Upstream refactoring the backup-restore loop** (seen 2026-05-25) — upstream extracted `ModelPatcherDynamic.partially_unload`'s restore loop into a new `restore_loaded_backups()` helper. The auto-merge kept upstream's `freed += self.restore_loaded_backups()` call site but dropped the fork's `bk.invertible` branch. Fix: thread the invertible branch *into* `restore_loaded_backups()` (so both its `load()` and `partially_unload()` callers get it) and keep the conflict site as the plain helper call — do **not** re-inline the loop. The same commit also had an import-block collision: `comfy_aimdo.host_buffer` (upstream) and `comfy.weight_adapter` (fork) landed on the same line — keep both.

**Invariant for any shape:** every site that restores `self.backup` must branch on `bk.invertible` and call `_invert_fast_path_weight(key, bk.patches)` instead of `set_attr_param(self.model, key, bk.weight)`. An invertible backup carries `weight=None`, so a plain restore writes `None` into the parameter and corrupts the model.

See the `perf(model-patcher): in-place LoRA fast-path with invertible unpatch` commit for the fast-path structure. Upstream changes to other functions in `model_patcher.py` (e.g. `save_lora_for_models`) generally won't conflict.

### `comfy/sd.py` — observability logging vs. upstream's `load_state_dict_guess_config` evolution (seen 2026-05-28)

The fork's `feat(memory)` commits inject memory-accounting logging (`CHECKPOINT done`, `MODEL_INIT`, `UNET_LOADED`, `VAE_LOADED`, `memory_report()`/`memory_delta()`) into `load_checkpoint_guess_config` / `load_state_dict_guess_config`, and change `assign=` to `model_patcher.should_assign_weights()`. Upstream keeps reworking the same lines, producing three concurrent conflict regions:

1. **Configurable `offload_device`** — upstream added `offload_device = model_options.get("offload_device", model_management.unet_offload_device())`. Keep upstream's config line **and** the fork's `unet_load_before` snapshot + `should_assign_weights()` (a superset of upstream's `is_dynamic()`: `return self.is_dynamic() or comfy.model_management.UNIFIED_MEMORY` — always prefer `should_assign_weights()`).
2. **Configurable VAE `load_device`** — upstream added `vae_device = model_options.get("load_device", None)` / `VAE(sd=vae_sd, metadata=metadata, device=vae_device)`. Keep upstream's config line **and** the fork's `VAE_LOADED` logging.
3. **`cached_patcher_init` registration refactor** — upstream replaced the old `load_checkpoint_guess_config_model_only` / `_clip_only` registration with `load_checkpoint_clip_patcher` / `load_checkpoint_vae_patcher` (and added VAE registration). Take upstream's new registration block wholesale, then append the fork's `CHECKPOINT done` logging before `return out`. (The `_model_only` / `_clip_only` functions still exist as upstream public API — uncalled in-repo, leave them.)

**Pattern for all three:** the fork only ever *adds logging lines and the `should_assign_weights()` call*; everything else in these hunks is upstream code the fork happened to carry. Take upstream's version of the surrounding logic, re-insert the fork's logging/`should_assign_weights()`.

### GB10 auto-defaults (`comfy/spark_defaults.py` + consumer gates) — new rebase-risk locus

Added by the GB10-auto-defaults feature (`feat/gb10-auto-defaults`, base `3f4185fd`; see
`dev-docs/gb10-auto-defaults.md` for the full design). Sits in the same rebase-risk loci as
the invertible-LoRA (`model_patcher.py`) and observability-logging (`sd.py`) conflicts above
— re-verify all three of the following after every sync, not just on conflict:

1. **`comfy.cli_args.enables_dynamic_vram()`** still short-circuits to `False` when
   `comfy.spark_defaults.enabled()` is `True`, with `args.enable_dynamic_vram` winning
   first (checked before the `spark_defaults` clause, unconditionally). If a rebase
   reworks this function's body (e.g. upstream adds a new VRAM-mode flag to the legacy
   fallback expression), the `spark_defaults.enabled()` short-circuit must stay intact and
   still precede that expression — it must never read `args.disable_dynamic_vram` on the
   GB10 path, or the deprecation-proofing is defeated.
2. **The `main.py` aimdo self-check WARNING** (placed right after the aimdo-activation `if
   args.enable_dynamic_vram or (enables_dynamic_vram() and ...):` block) survived the
   rebase, and its condition still includes all four of
   `comfy.model_management.UNIFIED_MEMORY`, `comfy.memory_management.aimdo_enabled`,
   `comfy.spark_defaults.enabled()`, and `not args.enable_dynamic_vram`. Losing the
   `spark_defaults.enabled()` guard specifically reintroduces a false positive on
   `--spark-defaults off` (the bug fixed in `4ccf603b`).
3. **The consumer-side gates are intact**: the `elif UNIFIED_MEMORY and
   comfy.spark_defaults.enabled():` arm on `EXTRA_RESERVED_VRAM` in
   `model_management.py`; `pinned_memory_disabled()` (`model_management.py`) still OR'd
   into all three read sites (`model_management.py`'s `MAX_PINNED_MEMORY` guard,
   `comfy/pinned_memory.py`'s `get_pin`/`pin_memory`); the `text_encoder_dtype()` bf16
   clause in `model_management.py` still sits after the five explicit dtype-flag `elif`s;
   and `comfy/cache_policy.py::_resolved_preset()`'s auto-`high` branch still requires
   `comfy.spark_defaults.enabled()` in addition to `mm.UNIFIED_MEMORY and not
   comfy.memory_management.aimdo_enabled` (losing just this `enabled()` clause silently
   breaks `--spark-defaults off`'s uniformity — the bug fixed in `42c2d483`).

Quick post-rebase spot-check:

```bash
uv run python -m pytest tests-unit/comfy_test/test_spark_defaults.py \
  tests-unit/comfy_test/test_dynamic_vram_gate.py \
  tests-unit/comfy_test/test_spark_consumer_defaults.py \
  tests-unit/comfy_test/test_cache_policy.py -q
```

## After the rebase

Restore the `models` symlink and pop any stash (above), then verify before trusting the result:

```bash
grep -rn '^<<<<<<<\|^>>>>>>>\|^=======' comfy/ || echo "no conflict markers"
uv run python -c "import comfy.model_patcher"   # a broken merge can still parse; importing catches more
uvx ruff check comfy/model_patcher.py
git rev-list --count upstream/master..HEAD       # = number of fork commits (23 on the 2026-06-06 sync)
git rev-list --count HEAD..upstream/master       # = 0
```

The first `uv run` after a rebase re-syncs the venv to the rebased `pyproject.toml`/`uv.lock` — expected, not a problem. `ruff` is **not** a pinned dependency, so use `uvx ruff` (or `uv run --with ruff ruff`), never `uv run ruff` (it fails with `Failed to spawn: ruff` once a resync drops the ad-hoc install).

## Publishing to origin

A rebase rewrites the fork's commit hashes, so `origin/master` diverges and a plain `git push` is rejected. Push with lease (safe — refuses if origin moved unexpectedly):

```bash
git push --force-with-lease origin master
```

## Recovery

If a rebase goes badly: `git rebase --abort`, then `git reset --hard backup/pre-rebase-<date>` (whichever backup tag is most recent) to restore.
