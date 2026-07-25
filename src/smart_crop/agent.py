"""Calls the local Qwen vision model to produce a crop plan for one image.

See agent_spec.md for the schema and reasoning this is built on.
"""

import argparse
import json
from pathlib import Path

from smart_crop.backends import BACKENDS, Backend, client_for
from smart_crop.crop import CropDecision, ResolutionFloorError, apply_crop
from smart_crop.preview import preview_data_url
from smart_crop.ratios import TARGETS

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

For iphone (the most aggressive crop): default to keeping full height (scale 1.0) and choose cx to \
find a vertically coherent slice of the frame -- a tree, rock formation, tower, path, or other \
subject/line that reads well in a narrow vertical crop. If no such slice exists, set worthwhile \
false. Only drop scale below 1.0 for a deliberate, clearly-improved zoom on an isolated vertical \
subject; do not zoom just to make something work -- a mediocre crop is worse than no crop.

If the frame contains a moving or discrete subject near where the crop would need to fall -- a \
person or animal in motion, a vehicle -- do not choose a crop that would slice through the \
subject's body to force it to fit. Prefer worthwhile=false over a crop that visibly bisects a \
subject; a missing crop is better than one that looks like a mistake.

Occasionally a wide image contains two or more distinct, independently strong subjects far enough \
apart that no single vertical slice can include both (e.g. two separate landmarks in the same \
frame). In that case only, you may submit more than one iphone decision, each isolating one \
subject, by including multiple entries with target="iphone" in decisions. Do this only when each \
candidate would independently stand as a strong, complete crop on its own -- never as a way to \
hedge between two mediocre options. Every other target (tv, macbook, ultrawide, ipad) must appear \
exactly once.

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


def get_crop_plan(image_path: Path, backend: Backend = DEFAULT_BACKEND) -> dict[str, list[CropDecision]]:
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

    plan: dict[str, list[CropDecision]] = {}
    for d in args["decisions"]:
        target = d.get("target")
        if target not in TARGETS:
            print(f"  malformed decision, skipping: {d}")
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
    return plan


IMAGE_EXTENSIONS = {".jpg", ".jpeg"}


def process_image(image_path: Path, output_dir: Path, backend: Backend = DEFAULT_BACKEND) -> None:
    plan = get_crop_plan(image_path, backend=backend)

    for target_name, decisions in plan.items():
        target = TARGETS[target_name]
        multi = len(decisions) > 1
        for i, decision in enumerate(decisions, 1):
            suffix = f"_alt{i}" if multi else ""
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
