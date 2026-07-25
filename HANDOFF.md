# Handoff (2026-07-25)

Status snapshot for resuming in a fresh session. Read `CLAUDE.md` for repo orientation and
`agent_spec.md` for design details first -- this is just "where we left off."

## What's working

Full pipeline validated end-to-end against real photos across several full 100-image batch runs
(`test_batch/` and `test-batch-2/`, both real-library pulls) over two sessions:

- `smart_crop.agent` -- single image/folder, one backend.
- `smart_crop.compare` -- multi-backend batch, pipelined per-image (crops + logs an image as soon
  as every requested backend has returned for it, instead of waiting for the whole batch's analysis
  phase to finish before cropping anything), image-major job ordering (see CLAUDE.md for why),
  JSONL decision log with `flags.py` review-flagging, per-call `duration_s`/token timing with a
  per-backend avg/min/max summary at the end, and a self-describing `run_meta` header line (git
  commit, backend/model names, `max_workers`, image count).
- Both `qwen` (Qwen3.6-35B-A3B-MLX-8bit) and `gemma` (gemma-4-31B-it-MLX-8bit) backends work
  against the same oMLX server and stay resident simultaneously.
- `enable_thinking: False` on the qwen backend: ~4-5x speedup, no observed quality loss.
- Multi-candidate `iphone` crops: confirmed triggering correctly and selectively.
- Resolution-floor safety net: confirmed catching real cases, including a systematic one -- several
  older/lower-res drone photos (~12MP) structurally cannot produce a native-resolution `ultrawide`
  export no matter the crop. A fact about the source material, not a bug.
- Output quality: `apply_crop` saves at `quality=100`/`subsampling=0`/`optimize=True` with ICC
  profile + EXIF preserved, and does a byte-for-byte copy instead of a lossy re-encode when the
  crop box is the full source frame (e.g. 4:3 drone photo -> `ipad`).
- **Second-pass review gate** (new, see agent_spec.md for full design): `iphone` decisions that
  pass the first-pass model now get a second, separate model call shown *only the rendered crop*
  (not the source image), asked whether it's actually a strong standalone portrait. Built because
  two specific `iphone` judgment failures (take-rate too high on small/distant subjects in mostly
  empty space; multi-candidate crops blending two unrelated scene elements) kept surviving direct
  prompt tightening -- grounding the check in the actual output pixels instead of the model's own
  description of a hypothetical crop is what finally caught them. Spot-verified 4-for-4 correct
  gate decisions against the specific images that exposed the original failures (including one case
  where the gate was right and an earlier *manual* review of the same crop had been wrong -- see
  agent_spec.md). Not yet run across a full batch to see its effect at scale or its own cost/false-
  positive rate.

## Fixed this session (2026-07-24 to 2026-07-25)

Roughly chronological; see `agent_spec.md` for full rationale on each:

1. **Ultrawide subject-clipping**: `tv`/`macbook`/`ultrawide`/`ipad` positioning was balance-based
   only, with nothing checking whether a subject's full extent actually fit -- clipped boats,
   bridges, rock formations, animal heads. Fixed by prompt tightening; verified clean on the 8
   originally-flagged images plus two full batch reruns (disagreement:cy dropped, not rose).
2. **Output quality**: `quality=100`/`subsampling=0`/ICC+EXIF preservation, plus a full-frame
   passthrough copy (no lossy re-encode) when source and target ratios already match.
3. **Batch pipelining**: per-image crop+log as soon as ready, image-major job ordering, so analysis
   and cropping overlap instead of running as two sequential phases.
4. **Debugging metadata**: `duration_s`, `prompt_tokens`/`completion_tokens`, `source_w`/`source_h`
   on every decision; `run_meta` header; `call_failed`/`malformed_decision` log entries (previously
   silently discarded or console-only). This is what surfaced the qwen-vs-gemma speed gap (below)
   and a real crash bug (a non-dict entry in a model's `decisions` array) with real data instead of
   guesswork.
5. **Parsing robustness**: a qwen response with a bare string in `decisions` crashed the whole call
   (132s wasted). Now treated as malformed and skipped, same as an unrecognized target.
6. **`disagreement:cx`/`disagreement:cy` false positives**: these flagged even when the axis had
   zero slack (both backends' boxes spanning the full source dimension regardless of the
   coordinate), which happened on ~2/3 of real occurrences in `test-batch-2`. Fixed by only
   flagging when at least one backend's box actually has slack on that axis. Confirmed on real
   data: `disagreement:cx` 72->24, `disagreement:cy` 94->47 on the same log.
7. **Iphone zoom threshold**: user strongly prefers full-resolution/full-frame exports over tighter
   "nicer-looking" zoomed crops. Prompt tightened to treat `scale < 1.0` as a rare exception only
   when no full-height slice works at all, not a taste-based choice. Verified: qwen's decision on a
   specific flagged image switched from `scale=0.7` to `scale=1.0`, matching gemma.
8. **Iphone natural-breakpoint cuts**: crops were cutting through an animal's body at arbitrary
   points instead of a natural break (below the shoulders, full body, etc.). Fixed by prompt
   tightening; this one generalized well on retest.
9. **Second-pass review gate**: see above -- built for the two `iphone` judgment failures that (7)
   and (8)'s prompt tuning fixed positioning/zoom for, but that resisted fixing via more prompt
   wording alone (take-rate on small/distant subjects; multi-candidate blending).

## Real qwen vs. gemma data (new this session)

First actual measurements, from `test-batch-2` (101 images, both backends): gemma averaged
**37.1s/call** (356 completion tokens avg), qwen averaged **91.5s/call** (689 completion tokens
avg) -- gemma is ~2.5x faster, directly explained by generating about half the completion tokens,
not a config difference (both have `enable_thinking` off where applicable). Combined with a
confirmed Gemma-only correctness bug (below) being low-volume and no quality edge found for qwen in
several rounds of spot-checking, this is now real evidence leaning toward gemma as the single-model
pick if forced to choose -- previously this was pure impression/vibes.

## Batch output location

Recent test batches were written to `/Volumes/home/Pictures/Background Photo Exports/smart-crop-tests`
(a mounted network volume) instead of a local path, at the user's request, to avoid excess writes to
the local SSD. This is just a CLI argument (`output_dir`/`log_path` in `smart_crop.compare`), not a
code default. Use distinctly-named `output`/`log.jsonl` paths per batch (e.g. `output-2`/`log-2.jsonl`)
so reruns don't clobber previous results in that shared folder.

## Open decisions (not yet made)

- `MAX_WORKERS = 8` in `compare.py` was only ever benchmarked against sequential (1 worker), never
  swept against other worker counts -- unconfirmed whether 8 is actually optimal. Still not run.
- Whether/how to validate the new review gate at full-batch scale -- watch for over-rejection
  (false positives costing good crops) or under-rejection, and its aggregate cost impact.

## Known issues, not yet fixed

- **Gemma portrait-source bug**: on a portrait-orientation (2:3) source, Gemma sometimes rejects
  `tv`/`macbook`/`ultrawide`/`ipad` with reasoning like "would require extending the frame" --
  factually wrong, since cropping a portrait source to a landscape ratio just trims height hard, it
  never needs to extend anything. Qwen doesn't have this bug on the same images. Confirmed twice now
  (`_DSC0562.jpg` in `test_batch/`, `_DSC3559.jpg` in `test-batch-2/`) -- worth a prompt fix if it
  recurs a third time.
- **`reason` text drift** (pre-existing): a decision's free-text `reason` can describe a positive,
  well-justified crop while the numeric `worthwhile` field says false (or vice versa). Numeric
  fields are still the only ground truth; don't trust `reason` for anything except human-readable
  context in the log.

## Deferred (explicit decisions, not forgotten)

- **Model choice**: no final decision; user has floated possibly using multiple models in the final
  solution rather than picking one. New timing/quality data (above) leans gemma if forced to choose.
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
