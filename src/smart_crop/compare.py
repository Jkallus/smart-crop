"""Run one or more backends over a batch of images and log per-decision metadata + flags.

Meant for batches too large to eyeball -- flags.py derives cheap, geometry-only signals
(no image content inspection) so a human only needs to look at what's actually flagged.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from smart_crop.agent import IMAGE_EXTENSIONS, get_crop_plan
from smart_crop.backends import BACKENDS, Backend
from smart_crop.crop import CropDecision, ResolutionFloorError, apply_crop, crop_box
from smart_crop.flags import decision_flags, disagreement_flags
from smart_crop.ratios import TARGETS

MAX_WORKERS = 8  # benchmarked: ~2-3x throughput vs sequential with both models resident in oMLX


def run_batch(
    image_paths: list[Path],
    output_dir: Path,
    backends: list[Backend],
    log_path: Path,
    max_workers: int = MAX_WORKERS,
) -> None:
    flagged = []
    total = 0

    # All (image, backend) jobs dispatched concurrently -- oMLX can hold multiple models resident
    # and genuinely batches concurrent requests (benchmarked ~2-3x throughput), so there's no more
    # reason to serialize by backend the way model-swap thrashing used to force.
    jobs = [(image_path, backend) for backend in backends for image_path in image_paths]
    all_plans: dict[str, dict[str, dict[str, list[CropDecision]]]] = {}  # image_name -> backend -> plan

    print(f"Dispatching {len(jobs)} jobs across {max_workers} workers...", flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(get_crop_plan, image_path, backend): (image_path, backend) for image_path, backend in jobs}
        for n, future in enumerate(as_completed(futures), 1):
            image_path, backend = futures[future]
            try:
                plan = future.result()
            except Exception as e:
                print(f"  [{n}/{len(jobs)}] {backend.name}/{image_path.name}: FAILED -- {e}", flush=True)
                continue
            all_plans.setdefault(image_path.name, {})[backend.name] = plan
            print(f"  [{n}/{len(jobs)}] {backend.name}/{image_path.name}: ok", flush=True)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        for image_path in image_paths:
            plans = all_plans.get(image_path.name, {})
            if not plans:
                continue

            with Image.open(image_path) as img:
                source_w, source_h = img.width, img.height

            for target_name, target in TARGETS.items():
                by_backend = {
                    name: plan[target_name] for name, plan in plans.items() if target_name in plan
                }
                cross_flags = disagreement_flags(by_backend)

                for backend_name, decisions in by_backend.items():
                    multi = len(decisions) > 1
                    for i, decision in enumerate(decisions, 1):
                        suffix = f"_alt{i}" if multi else ""
                        total += 1
                        flags = decision_flags(decision, target, source_w, source_h) + cross_flags
                        if multi:
                            flags = flags + ["multiple_candidates"]

                        entry = {
                            "image": image_path.name,
                            "backend": backend_name,
                            "target": target_name + suffix,
                            "worthwhile": decision.worthwhile,
                            "scale": decision.scale,
                            "cx": decision.cx,
                            "cy": decision.cy,
                            "reason": decision.reason,
                            "flags": flags,
                            "output_path": None,
                        }

                        if decision.worthwhile:
                            entry["box"] = list(crop_box(source_w, source_h, target, decision))
                            try:
                                result = apply_crop(
                                    image_path, target, decision, output_dir / backend_name, suffix=suffix
                                )
                                entry["output_path"] = str(result) if result else None
                            except ResolutionFloorError as e:
                                flags.append("resolution_rejected")
                                entry["flags"] = flags
                                entry["error"] = str(e)

                        log.write(json.dumps(entry) + "\n")

                        if flags:
                            flagged.append(entry)
                            print(f"  {image_path.name} {backend_name}/{target_name}{suffix}: {flags}", flush=True)

    print(f"\n{total} decisions logged to {log_path}, {len(flagged)} flagged for review.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", type=Path, help="A single image, or a directory of images.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--backends", nargs="+", default=["qwen"], choices=sorted(BACKENDS))
    args = parser.parse_args()

    if args.image_path.is_dir():
        images = sorted(p for p in args.image_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        images = [args.image_path]

    backends = [BACKENDS[name] for name in args.backends]
    run_batch(images, args.output_dir, backends, args.log_path)


if __name__ == "__main__":
    main()
