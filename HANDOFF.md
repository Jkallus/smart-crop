# Handoff (2026-07-24)

Status snapshot for resuming in a fresh session. Read `CLAUDE.md` for repo orientation and
`agent_spec.md` for design details first -- this is just "where we left off."

## What's working

Full pipeline validated end-to-end against real photos (a 45MP bridge shot, a 45MP drone sunset,
and a 100-image batch pulled from the real library):

- `smart_crop.agent` -- single image/folder, one backend.
- `smart_crop.compare` -- multi-backend batch with concurrent dispatch, JSONL decision log, and
  `flags.py` review-flagging so large batches don't require eyeballing every crop.
- Both `qwen` (Qwen3.6-35B-A3B-MLX-8bit) and `gemma` (gemma-4-31B-it-MLX-8bit) backends work
  against the same oMLX server and can stay resident simultaneously (confirmed after the user
  freed up memory -- no more model-swap thrashing).
- `enable_thinking: False` on the qwen backend: ~4-5x speedup (70-80s -> ~15s/call), no observed
  quality loss. Root cause of an earlier "is it stuck?" scare -- it wasn't, it was just burning
  ~20k tokens of chain-of-thought per image by default.
- Concurrent dispatch in `compare.py` (`ThreadPoolExecutor`, 8 workers): ~2-3x throughput,
  confirmed via benchmark, not just theoretical.
- Multi-candidate `iphone` crops: confirmed triggering correctly and selectively (5/100 images in
  the full batch), including the original motivating case (Sydney Harbour Bridge vs. Opera House
  as two independent portrait crops from the same source image).
- Resolution-floor safety net: confirmed catching real cases, including a systematic one --
  several older/lower-res drone photos in the library (~12MP, e.g. `DJI_20260601033100_0258_D.jpg`
  at 4057x3043) structurally cannot produce a native-resolution `ultrawide` export no matter what
  crop is chosen. This is a fact about the source material, not a bug.

## Last full batch run

100 images (`test_batch/`, mixed drone + DSC/portrait), both backends, concurrent: 990 decisions
logged, 279 flagged (28%). Breakdown is in the conversation history, short version:
- `disagreement:cy` (80) -- mostly one systematic Qwen-vs-Gemma behavioral difference
  (Gemma defaults to centered, Qwen makes an actual positional call), inflated 3x by repeating
  across tv/macbook/ultrawide for the same image since they share an axis.
- `resolution_rejected` (45, 37 of which are `ultrawide` at `scale=1.0`) -- the lower-res drone
  photo issue above.
- `disagreement:worthwhile` (62), `skipped` (48) -- expected, concentrated on `iphone` (hardest
  target).
- `multiple_candidates` (10 entries / 5 images) -- working as designed.

Output artifacts (`test_batch_output/`, `test_batch_log.jsonl`) are gitignored and were not kept
around after review -- rerun `smart_crop.compare` on `test_batch/` to reproduce if needed.

## Open decision (not yet made)

Whether to tune `flags.py`'s `DISAGREEMENT_CX_CY` threshold (currently 0.15) or de-duplicate the
correlated tv/macbook/ultrawide disagreement flags for the same image -- asked the user, no answer
yet as of this handoff. Pick this up first if resuming immediately.

## Deferred (explicit decisions, not forgotten)

- **Self-review pass** (feed the cropped result back to the model, ask if it's good): doesn't need
  an agent framework, just one more API call. Deliberately not built -- revisit only if prompt
  guidance + flags.py stop catching enough on larger batches, since it roughly doubles/triples cost
  per image.
- **Model choice**: user's early impression is Gemma > Qwen, but that was partly confounded by
  Qwen's thinking-mode overhead (now fixed) skewing perception of it as slow/heavyweight. Worth
  revisiting decision quality now that timing is comparable. User floated possibly using multiple
  models in the final solution rather than picking one.
- **Lightroom integration** (import from "Done WIP", export to per-ratio folders): not started,
  still just the original ask in `plan.md`. Current tooling assumes a flat folder of exported
  JPEGs as input.

## Not yet done, no decision needed, just hasn't come up

- `smart_crop:main` in `__init__.py` and the `[project.scripts]` entry in `pyproject.toml` are
  unused uv-init boilerplate (prints "Hello from smart-crop!"). Harmless, could be removed or wired
  to something real.
