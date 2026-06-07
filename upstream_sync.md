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

Conflicts have shown up in two files. `model_patcher.py` did **not** conflict on the 2026-05-28 sync (it auto-merged clean), and the 2026-06-06 sync (v0.22.0 → v0.24.1, 60 upstream commits) replayed all 23 fork commits with **zero** conflicts — so don't assume conflicts, but verify the invariants below regardless. On 2026-06-06 upstream's changes to the sensitive files were all aimdo/dynamic-pin internals plus new model support (TripoSplat, Ideogram4); it also removed the `is_dynamic()` gate in `free_memory`'s smart-memory branch (see the updated CLAUDE.md note — behavior is strictly milder, no fork action needed).

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
