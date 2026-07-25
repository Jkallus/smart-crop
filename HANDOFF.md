# Handoff (2026-07-24)

Status snapshot for resuming in a fresh session. Read `CLAUDE.md` for repo orientation and
`agent_spec.md` for design details first -- this is just "where we left off."

## What's working

Full pipeline validated end-to-end against real photos (a 45MP bridge shot, a 45MP drone sunset,
and repeated 100-image batches pulled from the real library) across three full-batch runs today:

- `smart_crop.agent` -- single image/folder, one backend.
- `smart_crop.compare` -- multi-backend batch, now pipelined per-image (crops + logs an image as
  soon as every requested backend has returned for it, instead of waiting for the whole batch's
  analysis phase to finish before cropping anything), image-major job ordering (see CLAUDE.md for
  why), JSONL decision log with `flags.py` review-flagging, and per-call `duration_s` timing with a
  per-backend avg/min/max summary printed at the end.
- Both `qwen` (Qwen3.6-35B-A3B-MLX-8bit) and `gemma` (gemma-4-31B-it-MLX-8bit) backends work
  against the same oMLX server and stay resident simultaneously.
- `enable_thinking: False` on the qwen backend: ~4-5x speedup, no observed quality loss.
- Multi-candidate `iphone` crops: confirmed triggering correctly and selectively.
- Resolution-floor safety net: confirmed catching real cases, including a systematic one -- several
  older/lower-res drone photos (~12MP) structurally cannot produce a native-resolution `ultrawide`
  export no matter the crop. A fact about the source material, not a bug.
- Output quality: `apply_crop` now saves at `quality=100`/`subsampling=0`/`optimize=True` with
  ICC profile + EXIF preserved (previously silently dropped), and does a byte-for-byte copy instead
  of a lossy re-encode when the crop box is the full source frame (e.g. 4:3 drone photo -> `ipad`).

## Fixed today: systematic ultrawide subject-clipping

User flagged 8 real images (boats, bridges, rock formations, Sydney Opera House, sheep, hills)
across `ultrawide` crops from both backends, all cutting off part of a discrete subject near a
frame edge while leaving in excess empty sky/water. Root cause: the prompt's positioning guidance
for `tv`/`macbook`/`ultrawide`/`ipad` was "use sky/foreground/horizon balance," with nothing
telling the model to check whether a subject's full extent actually fits before committing to a
position. Fixed by tightening `agent.py`'s `SYSTEM_PROMPT` (and the matching `agent_spec.md`
section) to require identifying each discrete subject's extent first, and treating a clipped
subject as strictly worse than an unevenly balanced crop of empty space.

Verified two ways:
1. Re-ran the original 8 flagged images -- all clean afterward (visually confirmed).
2. Re-ran the full 100-image batch twice more (995 and 1001 decisions logged) -- `disagreement:cy`
   dropped slightly (80 -> ~52-76 across runs) rather than rising, confirming the fix generalized
   rather than just papering over the 8 known cases. Overall flag rate ticked up (28% -> ~30-32%),
   but that's fully explained by `disagreement:worthwhile`/`skipped` rising on `iphone` (the model
   now more willing to say "doesn't fit" instead of forcing a mediocre crop -- arguably correct
   behavior, already covered by existing "a mediocre crop is worse than no crop" guidance) plus two
   small pre-existing issues unrelated to the fix (see below). No regressions found.

## Batch output location

Recent test batches were written to `/Volumes/home/Pictures/Background Photo Exports/smart-crop-tests`
(a mounted network volume) instead of a local path, at the user's request, to avoid excess writes to
the local SSD. This is just a CLI argument (`output_dir`/`log_path` in `smart_crop.compare`), not a
code default -- pass that path explicitly when running real batches if you want output to land
there again.

## Open decisions (not yet made)

- Whether to tune `flags.py`'s `DISAGREEMENT_CX_CY` threshold (currently 0.15) or de-duplicate the
  correlated tv/macbook/ultrawide disagreement flags for the same image. Asked the user previously,
  no answer yet.
- `MAX_WORKERS = 8` in `compare.py` was only ever benchmarked against sequential (1 worker), never
  swept against other worker counts -- unconfirmed whether 8 is actually optimal or just "better
  than nothing." User asked about this; agreed a sweep (4/8/16/32 on a fixed image subset,
  wall-clock comparison) would be worth doing but it hasn't been run yet.

## Known issues found today, not yet fixed

- **Gemma portrait-source bug**: on a portrait-orientation (2:3) source, Gemma sometimes rejects
  `tv`/`macbook`/`ultrawide`/`ipad` with reasoning like "would require extending the frame" --
  factually wrong, since cropping a portrait source to a landscape ratio just trims height hard,
  it never needs to extend anything. Qwen doesn't have this bug on the same images
  (`_DSC0562.jpg` in `test_batch/` is the example that surfaced it). Low volume (1 image out of
  100 in the last full batch) but worth a prompt fix if it recurs.
- **`reason` text drift** (pre-existing, documented in CLAUDE.md): confirmed again on
  `_DSC6257.jpg` -- qwen's `macbook` decision has `worthwhile: false` but a `reason` string that
  describes a positive, well-justified crop. Numeric fields are still the only ground truth.

## Deferred (explicit decisions, not forgotten)

- **Self-review pass** (feed the cropped result back to the model, ask if it's good): deliberately
  not built -- revisit only if prompt guidance + flags.py stop catching enough on larger batches.
- **Model choice**: no final decision; user has floated possibly using multiple models in the final
  solution rather than picking one.
- **Lightroom integration** (import from "Done WIP", export to per-ratio folders): not started,
  still just the original ask in `plan.md`. Current tooling assumes a flat folder of exported
  JPEGs as input.
- **Lossless crop via MCU-aligned JPEG transforms** (jpegtran-style, avoiding the decode/re-encode
  cycle entirely for non-full-frame crops): explicitly not pursued -- would require constraining
  `cx`/`cy`/`scale` to an 8x8 (or 16x16) pixel grid, not worth the complexity unless quality=100
  re-encoding is later found to be visibly insufficient.

## Not yet done, no decision needed, just hasn't come up

- `smart_crop:main` in `__init__.py` and the `[project.scripts]` entry in `pyproject.toml` are
  unused uv-init boilerplate (prints "Hello from smart-crop!"). Harmless, could be removed or wired
  to something real.
