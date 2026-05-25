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

`comfy/model_patcher.py`, always around the fork's invertible LoRA fast-path. Two shapes have shown up so far:

1. **`patch_weight_to_device` signature churn** — keep the fork's invertible branching in `force_load_param` and thread any new upstream parameters (e.g. `force_cast=True`) through the `self.patch_weight_to_device(...)` call.
2. **Upstream refactoring the backup-restore loop** (seen 2026-05-25) — upstream extracted `ModelPatcherDynamic.partially_unload`'s restore loop into a new `restore_loaded_backups()` helper. The auto-merge kept upstream's `freed += self.restore_loaded_backups()` call site but dropped the fork's `bk.invertible` branch. Fix: thread the invertible branch *into* `restore_loaded_backups()` (so both its `load()` and `partially_unload()` callers get it) and keep the conflict site as the plain helper call — do **not** re-inline the loop. The same commit also had an import-block collision: `comfy_aimdo.host_buffer` (upstream) and `comfy.weight_adapter` (fork) landed on the same line — keep both.

**Invariant for any shape:** every site that restores `self.backup` must branch on `bk.invertible` and call `_invert_fast_path_weight(key, bk.patches)` instead of `set_attr_param(self.model, key, bk.weight)`. An invertible backup carries `weight=None`, so a plain restore writes `None` into the parameter and corrupts the model.

See the `perf(model-patcher): in-place LoRA fast-path with invertible unpatch` commit for the fast-path structure. Upstream changes to other functions in `model_patcher.py` (e.g. `save_lora_for_models`) generally won't conflict.

## After the rebase

Restore the `models` symlink and pop any stash (above), then verify before trusting the result:

```bash
grep -rn '^<<<<<<<\|^>>>>>>>\|^=======' comfy/ || echo "no conflict markers"
uv run python -c "import comfy.model_patcher"   # a broken merge can still parse; importing catches more
uvx ruff check comfy/model_patcher.py
git rev-list --count upstream/master..HEAD       # = number of fork commits (currently 9)
git rev-list --count HEAD..upstream/master       # = 0
```

The first `uv run` after a rebase re-syncs the venv to the rebased `pyproject.toml`/`uv.lock` — expected, not a problem. `ruff` is **not** a pinned dependency, so use `uvx ruff` (or `uv run --with ruff ruff`), never `uv run ruff` (it fails with `Failed to spawn: ruff` once a resync drops the ad-hoc install).

## Recovery

If a rebase goes badly: `git rebase --abort`, then `git reset --hard backup/pre-rebase-<date>` (whichever backup tag is most recent) to restore.
