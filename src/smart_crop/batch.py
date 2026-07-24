"""Batch-run crop.py over an image folder using a hand-written JSON decisions file.

Lets us visually check crop quality before the LLM agent is wired up.

Decisions file format:
{
  "IMG_0001.jpg": {
    "tv": {"worthwhile": true, "scale": 1.0, "cx": 0.5, "cy": 0.6, "reason": "..."},
    "iphone": {"worthwhile": false, "reason": "no vertical subject"}
  }
}
"""

import argparse
import json
from pathlib import Path

from smart_crop.crop import CropDecision, apply_crop
from smart_crop.ratios import TARGETS


def run(image_dir: Path, decisions_path: Path, output_dir: Path) -> None:
    decisions = json.loads(decisions_path.read_text())

    for filename, per_target in decisions.items():
        source_path = image_dir / filename
        if not source_path.exists():
            print(f"skip: {filename} not found in {image_dir}")
            continue

        for target_name, fields in per_target.items():
            target = TARGETS[target_name]
            decision = CropDecision(target=target_name, **fields)
            result = apply_crop(source_path, target, decision, output_dir)
            status = result if result else f"skipped ({decision.reason})"
            print(f"{filename} -> {target_name}: {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("decisions_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    run(args.image_dir, args.decisions_json, args.output_dir)


if __name__ == "__main__":
    main()
