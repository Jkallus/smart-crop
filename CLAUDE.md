# CLAUDE.md

Orientation for working in this repo. Read `agent_spec.md` first -- it has the actual design
rationale (crop-box math, tool schema, prompt guidance, gotchas found via real-image testing).
This file is about how the code is organized and how to work in it, not why it's designed this way.

## What this is

Local vision LLM decides which of 5 export aspect ratios suit a landscape photo and picks a crop
for each; deterministic Python does the pixel math. Replaces a manual Lightroom workflow. See
`plan.md` for the original ask, `README.md` for usage.

## Module map (`src/smart_crop/`)

- `ratios.py` -- the 5 `Target`s (ratio, output folder, min resolution floor).
- `crop.py` -- `CropDecision` dataclass, `crop_box()` (pure math, well-tested), `apply_crop()`
  (does the actual crop + save, raises `ResolutionFloorError` as a safety net independent of the
  model's own judgment). Saves at `quality=100`/`subsampling=0`/`optimize=True` and preserves
  ICC profile + EXIF, to minimize loss from the unavoidable decode/crop/re-encode cycle. When the
  computed crop box is the full source frame (source and target ratios already match at
  `scale=1.0`, e.g. a 4:3 drone photo against `ipad`), it skips the decode/encode entirely and
  `shutil.copy2`s the original bytes instead.
- `preview.py` -- downsizes + base64-encodes an image for the vision model; the full-res original
  is only ever touched by `apply_crop`, never sent to the model. `preview_data_url()` previews the
  whole source; `crop_preview_data_url()` previews just a pixel box, used by the review gate below.
- `backends.py` -- named model configs (`BACKENDS` dict). Add a new model by adding an entry here.
- `agent.py` -- the system prompt, tool schema, and `get_crop_plan()` (one model, one image ->
  `CropPlanResult`, wrapping `plan: dict[str, list[CropDecision]]` plus `malformed` decisions and
  token `usage`). Also a single-backend CLI. All 5 targets (and any `iphone` multi-candidate
  entries) come back from a single tool call in a single model request -- the model reasons about
  all targets holistically in one context, never one call per target. Also `gate_iphone_crop()` /
  `review_iphone_crop()`: a second-pass review call, `iphone` only, shown just the rendered crop
  (not the source) and asked whether it's actually a strong standalone portrait -- see
  agent_spec.md's "Second-pass review gate" section for why and how this differs from the
  first-pass prompt.
- `compare.py` -- runs N backends over a batch, pipelined per-image (crops + logs an image as soon
  as every requested backend has returned for it, rather than waiting for the whole batch to finish
  analysis before cropping anything), writes a JSONL decision log, CLI entry point for real batches.
  Every line has a `"type"` field: `run_meta` (one header line: git commit, backend/model names,
  `max_workers`, image count, start time -- so a log file is self-describing without cross-
  referencing `HANDOFF.md`), `decision` (the normal per-target rows, now also carrying `source_w`/
  `source_h`, `duration_s`, `prompt_tokens`/`completion_tokens`, and -- for `iphone` -- 
  `gate_worthwhile`/`gate_reason` from the review gate, alongside `flags.py`-derived review flags
  including `gated_out` when the gate overrides a first-pass `worthwhile: true`), `call_failed` (a
  backend call that raised, instead of being lost to console scrollback), or `malformed_decision`
  (a raw decision dict the model returned with an unrecognized `target`, previously only printed
  and discarded). Also prints a per-backend avg/min/max duration and total token summary at the end
  of a run.
- `flags.py` -- cheap, geometry-only heuristics (no image content inspection) for flagging
  decisions worth a human look: unused coordinates, low scale, edge anchors, cross-backend
  disagreement. This is what makes batches of hundreds of images tractable to review.
- `batch.py` -- runs hand-written JSON decisions through `apply_crop`, no model involved. Useful
  for testing crop math changes without burning model calls.

## Conventions

- `uv run pytest` before considering any change to `crop.py`/`flags.py` done -- `tests/test_crop.py`
  covers the crop-box math (wide/narrow/matching ratios, scale/anchor positioning, bounds
  clamping). Extend it when the math changes; don't skip it because a change "looks obviously
  right" -- several real bugs here were exactly that kind of case.
- `CropDecision`/`Target` are the shared vocabulary across every module. `get_crop_plan` returns
  `dict[str, list[CropDecision]]` (not a single `CropDecision`) -- every target has exactly one
  entry except `iphone`, which may have more than one for the rare distinct-multi-subject case
  (see agent_spec.md). Don't assume `len(decisions) == 1`.
- All model calls go through `backends.py`'s `Backend`/`client_for` -- never hardcode a base_url or
  API key. `OMLX_API_KEY` must be set in the environment; never write it to a file.
- `compare.py` dispatches concurrently (`ThreadPoolExecutor`, `MAX_WORKERS = 8`, benchmarked only
  against sequential, not swept against other worker counts -- true optimum unconfirmed). Both
  models currently stay resident in oMLX simultaneously -- if that stops being true (memory
  pressure, a third model added), model-swap thrashing becomes a real risk again; see
  `agent_spec.md`'s history section before assuming concurrency is free. Jobs are queued
  image-major (all backends for image N before image N+1) -- the thread pool's work queue is FIFO,
  so backend-major ordering would starve the per-image pipelining (nothing becomes "ready" to crop
  until nearly a full backend pass completes).
- Sample/test images (loose `.jpg` files, `test_batch/`) are gitignored -- don't commit real photos
  to this repo. One deliberate exception: `docs/example/` (README illustration -- one source image
  and its 5 crops, downsized to ~1200px long edge for repo size), carved out via a `.gitignore`
  negation. Don't add other images there without the same downsizing treatment.

## Known rough edges

- The model's free-text `reason` field can drift from its actual numeric decision (stated intent
  vs. what the geometry actually does) -- don't trust `reason` as ground truth, only the numeric
  fields drive `crop_box`.
- Qwen3 models default to an extended "thinking" mode that's pure overhead for this structured-
  output task (~20k tokens/call, no measured quality benefit) -- disabled via
  `Backend.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}`. If you add another
  Qwen3-family backend, it likely needs the same treatment.
- `flags.py`'s `disagreement:cx`/`disagreement:cy` used to fire even when the axis had zero slack
  (both backends' boxes spanning the full source dimension regardless of the coordinate value) --
  fixed by suppressing the flag unless at least one backend's box actually has slack on that axis
  (`max_crop_box()`, shared with `crop_box()`). The same underlying behavioral difference between
  two models can still repeat identically across `tv`/`macbook`/`ultrawide` for one image when the
  axis *does* matter, since they share the same crop axis -- that part is not de-duplicated.
- Gemma has a reasoning bug on portrait-source-to-landscape-target crops: it sometimes rejects
  `tv`/`macbook`/`ultrawide`/`ipad` on a portrait-orientation source with reasoning like "would
  require extending the frame," which is factually wrong -- that crop just trims height hard, it
  never needs to extend anything. Qwen doesn't have this bug on the same images. Not yet fixed;
  found via real-batch review, see `HANDOFF.md`.
