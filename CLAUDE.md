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
  model's own judgment).
- `preview.py` -- downsizes + base64-encodes an image for the vision model; the full-res original
  is only ever touched by `apply_crop`, never sent to the model.
- `backends.py` -- named model configs (`BACKENDS` dict). Add a new model by adding an entry here.
- `agent.py` -- the system prompt, tool schema, and `get_crop_plan()` (one model, one image ->
  `dict[str, list[CropDecision]]`). Also a single-backend CLI.
- `compare.py` -- runs N backends over a batch concurrently, writes a JSONL decision log with
  `flags.py`-derived review flags, CLI entry point for real batches.
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
- `compare.py` dispatches concurrently (`ThreadPoolExecutor`, `MAX_WORKERS = 8`, benchmarked). Both
  models currently stay resident in oMLX simultaneously -- if that stops being true (memory
  pressure, a third model added), model-swap thrashing becomes a real risk again; see
  `agent_spec.md`'s history section before assuming concurrency is free.
- Sample/test images (loose `.jpg` files, `test_batch/`) are gitignored -- don't commit real photos
  to this repo.

## Known rough edges

- The model's free-text `reason` field can drift from its actual numeric decision (stated intent
  vs. what the geometry actually does) -- don't trust `reason` as ground truth, only the numeric
  fields drive `crop_box`.
- Qwen3 models default to an extended "thinking" mode that's pure overhead for this structured-
  output task (~20k tokens/call, no measured quality benefit) -- disabled via
  `Backend.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}`. If you add another
  Qwen3-family backend, it likely needs the same treatment.
- `flags.py`'s `disagreement:cx`/`disagreement:cy` threshold (0.15) can over-count: the same
  underlying behavioral difference between two models often repeats identically across
  `tv`/`macbook`/`ultrawide` for one image, since they share the same crop axis. Not yet
  de-duplicated -- see `HANDOFF.md` for current status.
