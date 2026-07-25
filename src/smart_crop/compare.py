"""Run one or more backends over a batch of images and log per-decision metadata + flags.

Meant for batches too large to eyeball -- flags.py derives cheap, geometry-only signals
(no image content inspection) so a human only needs to look at what's actually flagged.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from smart_crop.agent import IMAGE_EXTENSIONS, get_crop_plan
from smart_crop.backends import BACKENDS, Backend
from smart_crop.crop import CropDecision, ResolutionFloorError, apply_crop, crop_box
from smart_crop.flags import decision_flags, disagreement_flags
from smart_crop.ratios import TARGETS

MAX_WORKERS = 8  # benchmarked: ~2-3x throughput vs sequential with both models resident in oMLX


def _timed_get_crop_plan(image_path: Path, backend: Backend) -> tuple[dict[str, list[CropDecision]], float]:
    start = time.monotonic()
    plan = get_crop_plan(image_path, backend)
    return plan, time.monotonic() - start


def _process_image(
    image_path: Path,
    plans: dict[str, dict[str, list[CropDecision]]],
    durations: dict[str, float],
    output_dir: Path,
    log,
) -> tuple[int, list[dict]]:
    """Crop + log every decision for one image, across whichever backends finished for it.

    Called as soon as an image's last pending backend completes, not after the whole batch --
    lets cropping/logging for early-finishing images overlap with still-running model calls for
    the rest, and lets the log file be tailed/read mid-run instead of only after everything
    finishes.
    """
    total = 0
    flagged = []
    if not plans:
        return total, flagged

    with Image.open(image_path) as img:
        source_w, source_h = img.width, img.height

    for target_name, target in TARGETS.items():
        by_backend = {name: plan[target_name] for name, plan in plans.items() if target_name in plan}
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
                    # seconds for the single get_crop_plan call that produced all of this backend's
                    # decisions for this image -- same value repeated across its target rows.
                    "duration_s": round(durations[backend_name], 2) if backend_name in durations else None,
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

    log.flush()
    return total, flagged


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
    #
    # Ordered image-major (all backends for image N before any for image N+1), not backend-major:
    # the thread pool's work queue is FIFO, so with backend-major ordering every backend's jobs for
    # early images would sit behind the entire previous backend's pass before even starting,
    # defeating the point of processing an image as soon as all its backends land -- nothing would
    # be "ready" until nearly a full backend pass had completed.
    jobs = [(image_path, backend) for image_path in image_paths for backend in backends]
    backend_names = {b.name for b in backends}
    path_by_name = {p.name: p for p in image_paths}
    plans_by_image: dict[str, dict[str, dict[str, list[CropDecision]]]] = {}
    durations_by_image: dict[str, dict[str, float]] = {}
    all_durations: dict[str, list[float]] = {name: [] for name in backend_names}
    pending_backends: dict[str, set[str]] = {p.name: set(backend_names) for p in image_paths}

    print(f"Dispatching {len(jobs)} jobs across {max_workers} workers, pipelined per-image...", flush=True)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log, ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_timed_get_crop_plan, image_path, backend): (image_path, backend) for image_path, backend in jobs
        }
        for n, future in enumerate(as_completed(futures), 1):
            image_path, backend = futures[future]
            name = image_path.name
            try:
                plan, duration = future.result()
                plans_by_image.setdefault(name, {})[backend.name] = plan
                durations_by_image.setdefault(name, {})[backend.name] = duration
                all_durations[backend.name].append(duration)
                print(f"  [{n}/{len(jobs)}] {backend.name}/{name}: ok ({duration:.1f}s)", flush=True)
            except Exception as e:
                print(f"  [{n}/{len(jobs)}] {backend.name}/{name}: FAILED -- {e}", flush=True)

            pending_backends[name].discard(backend.name)
            if pending_backends[name]:
                continue

            # Every backend for this image has either succeeded or failed -- crop and log it now,
            # in parallel with the executor still chewing through the rest of the batch.
            del pending_backends[name]
            image_total, image_flagged = _process_image(
                path_by_name[name],
                plans_by_image.pop(name, {}),
                durations_by_image.pop(name, {}),
                output_dir,
                log,
            )
            total += image_total
            flagged.extend(image_flagged)

    print(f"\n{total} decisions logged to {log_path}, {len(flagged)} flagged for review.", flush=True)
    for name, durs in sorted(all_durations.items()):
        if durs:
            print(f"  {name}: avg {sum(durs) / len(durs):.1f}s/call over {len(durs)} calls "
                  f"(min {min(durs):.1f}s, max {max(durs):.1f}s)", flush=True)


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
