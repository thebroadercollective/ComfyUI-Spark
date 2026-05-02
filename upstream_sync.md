# Upstream sync

```bash
git fetch upstream
git rebase upstream/master
```

Expected conflict during rebase: `comfy/model_patcher.py` if upstream churned `patch_weight_to_device` again. Resolve by keeping the fork's invertible LoRA fast-path branching and threading any new upstream parameters through. See the `perf(model-patcher): in-place LoRA fast-path with invertible unpatch` commit for the structure.

If a rebase goes badly: `git rebase --abort`, then `git reset --hard backup/pre-rebase-20260502` (or whichever backup tag is most recent) to restore.
