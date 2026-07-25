"""Calls the local Qwen vision model to produce a crop plan for one image.

See agent_spec.md for the schema and reasoning this is built on.
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from smart_crop.backends import BACKENDS, Backend, client_for
from smart_crop.crop import CropDecision, ResolutionFloorError, apply_crop, crop_box
from smart_crop.preview import crop_preview_data_url, preview_data_url
from smart_crop.ratios import TARGETS, Target

DEFAULT_BACKEND = BACKENDS["qwen"]

SYSTEM_PROMPT = """\
You are evaluating a single landscape photograph against five export targets: \
tv (16:9), macbook (16:10), ultrawide (5120:2160), ipad (4:3), and iphone (9:19.5, portrait). \
These are landscape photographs -- do not reason about how people are framed; reason about \
composition, horizon placement, sky/foreground balance, and whether a coherent subject exists \
within a narrower crop.

For tv, macbook, ultrawide, and ipad: default to scale 1.0 (keep maximum resolution). Your only \
real decision is where to position the crop on the axis being trimmed (cy for tv/macbook/ultrawide, \
which crop height; cx for ipad, which crops width). Before choosing that position, first identify \
the full extent (topmost/bottommost, or leftmost/rightmost for ipad) of every discrete subject you \
want to keep -- a boat's mast, a bridge deck and towers, a rock formation's peak, a building or \
skyline, an animal's head, a ridgeline. Then choose the position that keeps each such subject \
entirely inside the crop box, even if that means the crop looks less "centered" or leaves more \
empty sky/water/foreground on one side than the other. Clipping through part of a subject (a boat \
missing its mast, a bridge missing a tower, a rock formation missing its top, an animal missing its \
head) is a worse failure than an unevenly balanced crop of empty space -- sky, water, and flat \
ground can be trimmed freely, discrete subjects cannot. Only fall back to a purely balance-based \
position when there is no single discrete subject near either edge to protect. These should almost \
always be worthwhile; mark worthwhile false only if literally no crop position preserves a coherent \
image (e.g. every position clips through some subject no matter where you place it).

Important: if the source's aspect ratio already exactly matches a target (e.g. a 4:3 drone photo \
for the ipad target), then at scale 1.0 there is zero room to move on *either* axis -- the crop is \
necessarily the full, unmodified frame no matter what cx/cy you set. If you want to reframe or \
emphasize a subject in that situation, you must lower scale below 1.0 to create room to reposition \
(exactly as you would for an artistic iphone crop); otherwise leave cx=cy=0.5 and accept the \
full-frame export rather than setting a cx/cy that will silently have no effect.

For iphone (the most aggressive crop): strongly prefer keeping full height (scale 1.0) and choose \
cx to find a vertically coherent slice of the frame -- a tree, rock formation, tower, path, or \
other subject/line that reads well in a narrow vertical crop. Preserving resolution and the \
appearance of the original image matters more than a tighter composition. Only drop scale below \
1.0 if literally no full-height slice reads as coherent -- the interesting content is spread \
horizontally with nothing that isolates well vertically at full height -- and even then use the \
least aggressive zoom that fixes the problem, not the tightest one that looks nicest. A looser, \
full-height crop is strongly preferred over a tighter zoomed one; never zoom purely to improve the \
composition of a slice that already works at scale 1.0. If no full-height slice works and zooming \
doesn't rescue it either, set worthwhile false -- a mediocre crop is worse than no crop.

Be a strict, human-like curator about worthwhile for iphone: a slice being geometrically coherent \
is not enough on its own. The subject needs to be a clear, meaningful visual focus that fills a \
real portion of the frame -- if the interesting content (an animal, a structure) is small and \
distant within mostly empty grass, sky, water, or pavement, that is not a strong portrait even \
though the slice is technically valid. Set worthwhile false in that case rather than exporting a \
crop that's mostly empty space around a minor detail.

If the frame contains a moving or discrete subject near where the crop would need to fall -- a \
person or animal in motion, a vehicle -- do not choose a crop that would slice through the \
subject's body to force it to fit. Prefer worthwhile=false over a crop that visibly bisects a \
subject; a missing crop is better than one that looks like a mistake. More generally, when a crop \
boundary falls across a subject's body or a structure, prefer a natural breakpoint over an \
arbitrary cut: for an animal, cut cleanly below the shoulders/neck or include the full body and \
legs, not an arbitrary point through the torso or mid-leg; for a building, prefer a cut that \
respects an architectural line (roofline, floor line) over one that clips through signage or \
windows without reason. If every available crop position cuts awkwardly through the subject with \
no natural breakpoint available, that's a legitimate case where nothing can be done well -- prefer \
worthwhile=false over an awkward cut, unless the subject's single most essential, recognizable part \
(e.g. an animal's head and face) is still fully included by some position.

Occasionally a wide image contains two or more distinct, independently strong subjects far enough \
apart that no single vertical slice can include both (e.g. two separate landmarks in the same \
frame). In that case only, you may submit more than one iphone decision, each isolating one \
subject, by including multiple entries with target="iphone" in decisions. Do this only when each \
candidate would independently stand as a strong, complete crop on its own, focused on a single \
coherent subject -- never as a way to hedge between two mediocre options, and never by blending two \
loosely related scene elements (e.g. a building and unrelated street activity below it) into one \
candidate that doesn't clearly belong together. Every other target (tv, macbook, ultrawide, ipad) \
must appear exactly once.

When zooming in (scale below 1.0) to isolate a subject, don't crop out an entire compositional \
element that was present in the original frame -- e.g. all of the sky, or all of the foreground -- \
unless the subject truly fills the frame edge-to-edge. A crop so tight it eliminates all \
surrounding context reads as a mistake, not a deliberate choice; leave a visible margin around the \
subject.

Call submit_crop_plan exactly once, with your decisions. Every target other than iphone must \
appear exactly once; iphone may appear more than once only in the rare multi-subject case above. \
For a decision with worthwhile=false, omit scale/cx/cy and just give a reason.\
"""

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_crop_plan",
        "description": (
            "Submit the crop decision for all five export targets for this image. "
            "tv/macbook/ultrawide/ipad must appear exactly once each; iphone normally appears "
            "once but may appear more than once for the rare distinct-multi-subject case."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "minItems": 5,
                    "items": {
                        "type": "object",
                        "properties": {
                            "target": {
                                "type": "string",
                                "enum": ["tv", "macbook", "ultrawide", "ipad", "iphone"],
                            },
                            "worthwhile": {"type": "boolean"},
                            "scale": {
                                "type": "number",
                                "description": "Fraction (0-1] of the max possible crop to use. 1.0 = max resolution.",
                            },
                            "cx": {"type": "number", "description": "Normalized crop-box center, 0-1 across width."},
                            "cy": {"type": "number", "description": "Normalized crop-box center, 0-1 across height."},
                            "reason": {"type": "string"},
                        },
                        "required": ["target", "worthwhile", "reason"],
                    },
                }
            },
            "required": ["decisions"],
        },
    },
}


@dataclass
class CropPlanResult:
    plan: dict[str, list[CropDecision]]
    # Raw decision dicts whose "target" wasn't one of the five known targets -- kept instead of
    # only printed, so a batch run can log them for post-hoc debugging instead of losing them to
    # console scrollback.
    malformed: list[dict] = field(default_factory=list)
    usage: dict | None = None  # {"prompt_tokens", "completion_tokens", "total_tokens"}, if reported


def get_crop_plan(image_path: Path, backend: Backend = DEFAULT_BACKEND) -> CropPlanResult:
    client = client_for(backend)
    data_url = preview_data_url(image_path)

    response = client.chat.completions.create(
        model=backend.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "Produce the crop plan for this image."},
                ],
            },
        ],
        tools=[TOOL_SCHEMA],
        tool_choice={"type": "function", "function": {"name": "submit_crop_plan"}},
        extra_body=backend.extra_body,
    )

    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    usage = None
    if response.usage is not None:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    plan: dict[str, list[CropDecision]] = {}
    malformed: list[dict] = []
    for d in args["decisions"]:
        # The model occasionally returns a decisions array with a non-dict element (e.g. a bare
        # string) instead of every entry being a proper object -- treat that as malformed too,
        # rather than crashing the whole call on d.get().
        target = d.get("target") if isinstance(d, dict) else None
        if target not in TARGETS:
            print(f"  malformed decision, skipping: {d}")
            malformed.append(d)
            continue
        plan.setdefault(target, []).append(
            CropDecision(
                target=target,
                # A model omitting "worthwhile" is malformed output; default to not exporting
                # rather than crashing the whole batch on one bad decision.
                worthwhile=d.get("worthwhile", False),
                scale=d.get("scale", 1.0),
                cx=d.get("cx", 0.5),
                cy=d.get("cy", 0.5),
                reason=d.get("reason", ""),
            )
        )
    return CropPlanResult(plan=plan, malformed=malformed, usage=usage)


REVIEW_SYSTEM_PROMPT = """\
You are reviewing a single candidate crop already produced for an iPhone portrait export \
(9:19.5 aspect ratio). You are shown only the finished crop, not the original wider photo -- \
judge it purely as a standalone photo, the way someone scrolling through exported photos would.

Reject (worthwhile: false) if any of the following apply:
- The subject is small, distant, or incidental within a frame that's mostly empty space (sky, \
grass, water, pavement) -- it doesn't read as a photo *of* anything in particular, even if some \
detail is technically visible somewhere in the frame.
- The crop combines two visually disconnected elements (e.g. an architectural closeup on top and \
unrelated street-level activity at the bottom) that don't cohere as one photographic subject.
- The crop boundary cuts across a subject's body or a structure at an arbitrary, awkward point \
rather than a natural breakpoint (e.g. mid-torso on an animal instead of below the shoulders, or \
through signage/windows on a building instead of along a roofline).

Otherwise, if this reads as a strong, coherent standalone portrait photo, approve it \
(worthwhile: true). Call submit_review exactly once with your verdict and a one-sentence reason.\
"""

REVIEW_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_review",
        "description": "Submit your verdict on whether this crop is a strong standalone portrait export.",
        "parameters": {
            "type": "object",
            "properties": {
                "worthwhile": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["worthwhile", "reason"],
        },
    },
}


def review_iphone_crop(
    image_path: Path, box: tuple[int, int, int, int], backend: Backend
) -> tuple[bool, str, dict | None]:
    """Second-pass review of an already-rendered iphone crop, shown only the cropped pixels.

    Grounds the worthwhile judgment in the actual output instead of the model's own description of
    a hypothetical crop -- catches cases (small/distant subject in mostly empty space, two
    disconnected elements blended together) that resisted first-pass prompt tuning alone.
    """
    client = client_for(backend)
    data_url = crop_preview_data_url(image_path, box)

    response = client.chat.completions.create(
        model=backend.model,
        messages=[
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "Review this candidate iPhone portrait crop."},
                ],
            },
        ],
        tools=[REVIEW_TOOL_SCHEMA],
        tool_choice={"type": "function", "function": {"name": "submit_review"}},
        extra_body=backend.extra_body,
    )

    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    usage = None
    if response.usage is not None:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return bool(args.get("worthwhile", True)), args.get("reason", ""), usage


def gate_iphone_crop(
    image_path: Path, source_w: int, source_h: int, target: Target, decision: CropDecision, backend: Backend
) -> tuple[bool, str, dict | None]:
    """Run review_iphone_crop for one worthwhile iphone decision. Fails open (approves) if the
    review call itself errors, so a network hiccup can't silently discard an otherwise-good export.
    """
    box = crop_box(source_w, source_h, target, decision)
    try:
        return review_iphone_crop(image_path, box, backend)
    except Exception as e:
        return True, f"review call failed, approved by default: {e}", None


IMAGE_EXTENSIONS = {".jpg", ".jpeg"}


def process_image(image_path: Path, output_dir: Path, backend: Backend = DEFAULT_BACKEND) -> None:
    plan = get_crop_plan(image_path, backend=backend).plan

    with Image.open(image_path) as img:
        source_w, source_h = img.width, img.height

    for target_name, decisions in plan.items():
        target = TARGETS[target_name]
        multi = len(decisions) > 1
        for i, decision in enumerate(decisions, 1):
            suffix = f"_alt{i}" if multi else ""

            if target_name == "iphone" and decision.worthwhile:
                approved, gate_reason, _ = gate_iphone_crop(image_path, source_w, source_h, target, decision, backend)
                if not approved:
                    print(f"  {target_name}{suffix}: gated out -- {gate_reason}  [{decision.reason}]")
                    continue

            try:
                result = apply_crop(image_path, target, decision, output_dir, suffix=suffix)
            except ResolutionFloorError as e:
                print(f"  {target_name}{suffix}: rejected -- {e}")
                continue

            status = result if result else f"skipped ({decision.reason})"
            print(f"  {target_name}{suffix}: {status}  [{decision.reason}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", type=Path, help="A single image, or a directory of images.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--backend", default="qwen", choices=sorted(BACKENDS))
    args = parser.parse_args()
    backend = BACKENDS[args.backend]

    if args.image_path.is_dir():
        images = sorted(p for p in args.image_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        for image_path in images:
            print(f"{image_path.name}:")
            try:
                process_image(image_path, args.output_dir, backend=backend)
            except Exception as e:
                print(f"  FAILED: {e}")
    else:
        print(f"{args.image_path.name}:")
        process_image(args.image_path, args.output_dir, backend=backend)


if __name__ == "__main__":
    main()
