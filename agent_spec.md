# Crop Agent Spec (draft)

## Target ratios

| Name | Ratio (W/H) | Notes |
|---|---|---|
| tv | 16:9 = 1.778 | TVs, monitors |
| macbook | 16:10 = 1.6 | MacBook Pro screen |
| ultrawide | 5120:2160 = 2.3704 | ultrawide monitor |
| ipad | 4:3 = 1.333 | iPad screen |
| iphone | 9:19.5 = 0.4615 | iPhone portrait |

Source ratios: 3:2 = 1.5 (majority), 2:3 = 0.667 (portrait originals), 4:3 = 1.333 (drone).

## Crop model

For a source ratio `S` and target ratio `T` (both W/H):

- `T == S`: no crop needed, straight export.
- `T > S`: crop height, keep full width (vertical anchor).
- `T < S`: crop width, keep full height (horizontal anchor).

`max_w, max_h` = the largest box at ratio `T` that fits inside the source. One of `max_w == source_w` or `max_h == source_h` always holds.

## Tool call

The model calls **one** tool, once per image, submitting a decision for all five targets at once (keeps reasoning holistic instead of five independent, possibly-inconsistent calls):

```json
{
  "name": "submit_crop_plan",
  "parameters": {
    "decisions": [
      {
        "target": "tv | macbook | ultrawide | ipad | iphone",
        "worthwhile": true,
        "scale": 1.0,
        "cx": 0.5,
        "cy": 0.62,
        "reason": "short justification"
      }
      // one entry per target, always 5 entries
    ]
  }
}
```

- `scale`: fraction (0-1] of `max_w × max_h` to actually use. `1.0` = keep as much resolution as possible (the default/common case). Lower values zoom in for a deliberately tighter, more isolated composition.
- `cx`, `cy`: normalized center of the crop box in the *source* image, 0-1 across width/height. On the axis that isn't being cropped (i.e. at `scale == 1.0`, the "full" axis), this coordinate is ignored.
- `worthwhile: false`: omit `scale`/`cx`/`cy`, just give `reason`. Means: don't produce this export.

### Python side (deterministic, no model math)

```
box_w, box_h = max_w * scale, max_h * scale
left = clamp(cx * source_w - box_w/2, 0, source_w - box_w)
top  = clamp(cy * source_h - box_h/2, 0, source_h - box_h)
crop = source.crop(left, top, left+box_w, top+box_h)
```

Plus a resolution floor check as a safety net (reject/log, don't silently export): compare final crop's pixel dimensions against a minimum per target. In practice this should only ever be able to fire on `iphone`, since the other four targets never crop aggressively enough at these source megapixel counts to approach a floor.

## System prompt guidance (draft)

> You are evaluating a single landscape photograph against five export targets: tv (16:9), macbook (16:10), ultrawide (5120:2160), ipad (4:3), and iphone (9:19.5 portrait). These are landscape photographs — do not reason about how people are framed; reason about composition, horizon placement, sky/foreground balance, and whether a coherent subject exists within a narrower crop.
>
> For **tv, macbook, ultrawide, ipad** (the four ratios close to or wider than typical source ratios): default to `scale: 1.0` — keep maximum resolution. Your only real decision is where to position the crop on the axis being trimmed (`cy` for tv/macbook/ultrawide since they crop height; `cx` for ipad since it crops width, unless the source is already 4:3). Before choosing that position, identify the full extent of every discrete subject you want to keep (a boat's mast, a bridge deck and towers, a rock formation's peak, a skyline, an animal's head, a ridgeline) and pick the position that keeps each one entirely inside the box, even at the cost of an unevenly balanced crop of empty sky/water/ground. Clipping through part of a subject is a worse failure than uneven empty space. Only fall back to pure sky/foreground/horizon balance when no discrete subject sits near either edge. These should almost always be `worthwhile: true` — mark `false` only if literally no crop position preserves a coherent image (rare).
>
> For **iphone** (portrait, the most aggressive crop): strongly prefer keeping full height (`scale: 1.0`, `cy` irrelevant) and choose `cx` to find a vertically coherent slice of the frame — a tree, rock formation, waterfall, path, or other subject/line that reads well in a narrow vertical crop. Preserving resolution and the appearance of the original image matters more than a tighter composition. Only drop `scale` below 1.0 if literally no full-height slice reads as coherent, and even then use the least aggressive zoom that fixes the problem, not the tightest one that looks nicest — never zoom purely to improve a composition that already works at `scale: 1.0`. If no full-height slice works and zooming doesn't rescue it either, set `worthwhile: false` — a mediocre crop is worse than no crop.
>
> Respond only by calling `submit_crop_plan` with exactly five decisions, one per target.

## Gotcha: source ratio exactly matches target

If `source_ratio == target.ratio`, `max_w == source_w` and `max_h == source_h` simultaneously --
there's zero slack on *either* axis at `scale: 1.0`, so `cx`/`cy` have no effect regardless of
value (e.g. a 4:3 drone photo against the `ipad` target). To reposition/reframe in that case, the
model must lower `scale`, same as any other artistic crop. The system prompt calls this out
explicitly, since it's easy for a model to describe a shift in `reason` while leaving `scale: 1.0`,
producing a tool call whose stated intent doesn't match its actual (full-frame) output.

## Multi-candidate crops (iphone only)

`tv`/`macbook`/`ultrawide`/`ipad` must appear exactly once each in `decisions`. `iphone` normally
appears once too, but the model may submit more than one `iphone` entry when a wide image contains
two or more distinct, independently strong subjects far enough apart that no single vertical slice
can include both (e.g. two separate landmarks in the same frame) -- each candidate must stand as a
complete crop on its own, not a hedge between two mediocre options. `get_crop_plan` returns
`dict[str, list[CropDecision]]` (keyed by target) to accommodate this; every other target's list
has exactly one entry. When a target has more than one candidate, each gets exported with a
`_alt1`, `_alt2`, ... filename suffix. This is deliberately rare in practice -- most images still
produce a single iphone decision.

## Prompt guidance for subject-cutting and context loss

Two failure modes found via real-image testing, now covered in the system prompt:

- **Bisecting a moving/discrete subject** (e.g. a running dog) when flipping portrait/landscape:
  the model is told to prefer `worthwhile: false` over a crop that would slice through a subject's
  body to force it to fit.
- **Cropping out an entire compositional element** (e.g. all of the sky) when zooming to isolate a
  subject at `scale < 1.0`: the model is told to leave a visible margin of context unless the
  subject truly fills the frame edge-to-edge.

Neither required a schema change, just tighter prompt wording -- verified against the images that
originally exposed the problems.

A third failure mode, found in a 2026-07-24 review of a `test_batch` run: both backends'
**ultrawide** decisions repeatedly clipped a discrete subject near the top or bottom edge of the
frame (a boat's mast, a bridge's deck/tower, a rock formation's peak, the Sydney Opera
House/skyline, an animal's head, a hillside ridge) while leaving in a disproportionate amount of
empty sky or water on the other side -- i.e. "balance" framing was winning out over "don't cut the
subject." Same failure on both `tv`/`macbook`/`ipad` positioning logic (they share the `cy`/`cx`
math with ultrawide), just observed concretely on ultrawide crops first. Fixed by tightening the
tv/macbook/ultrawide/ipad prompt paragraph to require identifying each discrete subject's full
extent *before* choosing a position, and treating a clipped subject as strictly worse than an
unevenly balanced crop of empty space. Re-validated against two fresh 100-image batch runs after
the fix -- `disagreement:cy` dropped rather than rose, confirming the fix generalized rather than
just covering the originally-flagged images.

A fourth round of feedback, from user review of a `test-batch-2` run (2026-07-25), covered three
more `iphone`-specific issues, all fixed by prompt tightening only (no code changes):

- **Take rate too high**: some accepted crops had a geometrically coherent slice but a subject too
  small/distant within mostly empty space (e.g. two cows barely visible on a hillside) to actually
  be a strong standalone portrait. The `iphone` paragraph now explicitly requires the subject be a
  "clear, meaningful visual focus that fills a real portion of the frame," not just technically
  coherent.
- **Arbitrary mid-body cuts**: several accepted crops were fine to take but cut through an animal's
  torso/legs at an arbitrary point instead of a natural breakpoint (below the shoulders, full body,
  etc.), or clipped through signage/windows on a building rather than respecting an architectural
  line. The existing "don't bisect a moving/discrete subject" guidance only covered the binary
  take/skip decision, not *where* to cut when taking it -- generalized to require a natural
  breakpoint when one exists, falling back to `worthwhile: false` only when no position offers one
  and the subject's most essential recognizable part (e.g. an animal's head) can't be preserved
  either.
- **Multi-candidate blending unrelated elements**: one flagged `iphone_alt2` combined a building
  (church tower) with unrelated street activity below it into a single candidate that didn't read
  as one coherent subject. Multi-candidate guidance tightened to require each candidate focus on a
  single coherent subject, not blend two loosely related scene elements.

Verified by re-running the seven specific images the user flagged (across both backends) after the
prompt change -- see git log for the before/after comparison at the time of the fix.

## Second-pass review gate (iphone only)

Implemented 2026-07-25, after prompt tuning alone stopped closing the gap on two `iphone`-specific
judgment failures the user kept finding in real batches: take rate too high (a technically-coherent
slice with a small/distant subject in mostly empty space) and multi-candidate crops blending two
unrelated scene elements. This was the "self-review pass" idea from the original open-items list,
scoped down to just `iphone` (the one target with these problems) to keep the cost increase small.

Design: `agent.gate_iphone_crop()` runs after a first-pass `worthwhile: true` iphone decision, but
before `apply_crop` actually writes it. It renders *only the cropped region* (via
`preview.crop_preview_data_url`, not the whole source) and sends that to a second, separate model
call (`agent.review_iphone_crop`, its own system prompt + `submit_review` tool schema) that judges
the actual output pixels, not the model's own description of a hypothetical crop. This matters:
the failures above were cases where the first-pass model's reasoning sounded right but didn't match
what the rendered crop actually looked like -- grounding the second call in the real output instead
of abstract coordinates is what let it catch what prompt wording alone couldn't.

If rejected, the decision is logged with a `gated_out` flag and `gate_worthwhile`/`gate_reason`
fields, and `apply_crop` is skipped for it (nothing written to disk). Fails open (approves) if the
review call itself errors, so a network hiccup can't silently discard a good export. Runs as a
plain blocking call in the batch runner's main thread rather than through the thread pool -- this
doesn't serialize the rest of the batch, since Python releases the GIL during network I/O and the
pool's worker threads keep making progress on other jobs in parallel; the only cost is that the
main loop's bookkeeping for *this* image's finalization is delayed by one extra network round trip.

Verified against the seven images that exposed the two failures: the gate correctly rejected the
empty-hillside cow (`_DSC3394.jpg`) and the tower+street blend (`_DSC2968_alt2.jpg`) that survived
prompt tuning, and correctly rejected a `_DSC3424.jpg` candidate that (on manual re-inspection of
the raw pixels) genuinely did cut off the subject's face -- a crop that had been mis-judged as good
in an earlier manual review pass, underscoring that grounding the check in actual rendered pixels
catches mistakes a text-only judgment (mine included) can miss.

Cost impact: only fires for `worthwhile: true` iphone decisions (not all 5 targets), so it's a
fraction of total calls, not a doubling -- not yet measured in aggregate token/time terms across a
full batch.

## Open items

- Exact wording/examples for the system prompt will likely need further iteration.
- Whether the review gate itself needs prompt tuning over time (it's a fresh, unvalidated-at-scale
  prompt) -- watch for it over- or under-rejecting once run across a full batch.
- Folder/output naming convention beyond `_altN` suffixes not otherwise revisited.
