# Pending Spark port for Comfy_HunyuanImage3

## What's here

- `hunyuan_instruct_nodes.py.spark-snapshot` — full file with Spark optimizations applied (from commit `3ae4cb9` of `EricRollei/Comfy_HunyuanImage3` fork, snapshot 2026-05-02).
- `3ae4cb9-spark-instruct-loader.patch` — the fork commit as a patch file.

## Why dropped

On 2026-05-02 the fork commit was dropped from `custom_nodes/Comfy_HunyuanImage3/main` so the directory could fast-forward to upstream `EricRollei/main`. The commit could not be auto-rebased: upstream commits `eeaa92e`, `bf738db`, `c12f69b`, `c8e761c` heavily restructured `HunyuanInstructLoader` (added `moe_drop_tokens`, `vae_dtype`, bucket bypass, seqlen auto-bump). All 10 of the inner hunks rejected; only the import block + helper definitions would apply cleanly.

## What the patch does

Adds DGX-Spark-specific page-cache management around `from_pretrained()` calls inside `HunyuanInstructLoader`:
- `_is_unified_memory()` helper → checks `comfy.model_management.UNIFIED_MEMORY`
- `_page_cache_dropper()` background thread → drops OS page cache for safetensors shards every 30s during model load, prevents ~83GB mmap'd page cache buildup on unified memory.
- Replaces `torch.cuda.empty_cache()` with `soft_empty_cache_unified()` at several sites.
- Replaces a manual `time.sleep(15)` hack with programmatic page-cache management.
- Adds `memory_report()` / `memory_delta()` observability around the load path.

## Before re-implementing

Confirm whether the issue this addressed (~83GB page-cache buildup during HunyuanImage3 load on unified memory) still reproduces against the new upstream version. Upstream may have fixed equivalent symptoms via `vae_dtype` selection, the new `moe_drop_tokens`, or unrelated load-path changes. If the symptom is gone, the port is unnecessary.

## Recovery

The full pre-drop commit history lives at `backup/pre-rebase-20260502` (local branch) inside `custom_nodes/Comfy_HunyuanImage3/.git`. To revive:

```bash
cd custom_nodes/Comfy_HunyuanImage3
git switch backup/pre-rebase-20260502
```
