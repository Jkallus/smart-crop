"""Run one or more backends over a batch of images and log per-decision metadata + flags.

Meant for batches too large to eyeball -- flags.py derives cheap, geometry-only signals
(no image content inspection) so a human only needs to look at what's actually flagged.
"""

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from smart_crop.agent import IMAGE_EXTENSIONS, get_crop_plan
from smart_crop.backends import BACKENDS, Backend
from smart_crop.crop import CropDecision, ResolutionFloorError, apply_crop, crop_box
from smart_crop.flags import decision_flags, disagreement_flags
from smart_crop.ratios import TARGETS

MAX_WORKERS = 8  # benchmarked: ~2-3x throughput vs sequential with both models resident in oMLX


@dataclass
class _CallResult:
    plan: dict[str, list[CropDecision]] | None
    malformed: list[dict] = field(default_factory=list)
    usage: dict | None = None
    duration_s: float = 0.0
    error: str | None = None


def _timed_get_crop_plan(image_path: Path, backend: Backend) -> _CallResult:
    start = time.monotonic()
    try:
        result = get_crop_plan(image_path, backend)
        return _CallResult(
            plan=result.plan, malformed=result.malformed, usage=result.usage, duration_s=time.monotonic() - start
        )
    except Exception as e:
        # Caught here (not left to propagate through future.result()) so a single bad call still
        # yields a duration and an error string we can write to the log, instead of a batch run
        # losing the failure to console scrollback with no durable record.
        return _CallResult(plan=None, duration_s=time.monotonic() - start, error=str(e))


def _git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _process_image(
    image_path: Path,
    plans: dict[str, dict[str, list[CropDecision]]],
    durations: dict[str, float],
    usage: dict[str, dict],
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
        cross_flags = disagreement_flags(by_backend, target, source_w, source_h)

        for backend_name, decisions in by_backend.items():
            multi = len(decisions) > 1
            for i, decision in enumerate(decisions, 1):
                suffix = f"_alt{i}" if multi else ""
                total += 1
                flags = decision_flags(decision, target, source_w, source_h) + cross_flags
                if multi:
                    flags = flags + ["multiple_candidates"]

                backend_usage = usage.get(backend_name) or {}
                entry = {
                    "type": "decision",
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
                    "source_w": source_w,
                    "source_h": source_h,
                    # seconds/tokens for the single get_crop_plan call that produced all of this
                    # backend's decisions for this image -- same values repeated across its target rows.
                    "duration_s": round(durations[backend_name], 2) if backend_name in durations else None,
                    "prompt_tokens": backend_usage.get("prompt_tokens"),
                    "completion_tokens": backend_usage.get("completion_tokens"),
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
    usage_by_image: dict[str, dict[str, dict]] = {}
    all_durations: dict[str, list[float]] = {name: [] for name in backend_names}
    all_usage: dict[str, list[dict]] = {name: [] for name in backend_names}
    pending_backends: dict[str, set[str]] = {p.name: set(backend_names) for p in image_paths}

    print(f"Dispatching {len(jobs)} jobs across {max_workers} workers, pipelined per-image...", flush=True)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log, ThreadPoolExecutor(max_workers=max_workers) as ex:
        # Self-describing run header -- without this, comparing two log files months apart means
        # cross-referencing HANDOFF.md notes to guess which prompt/code version produced which.
        log.write(
            json.dumps(
                {
                    "type": "run_meta",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "git_commit": _git_commit_hash(),
                    "backends": [{"name": b.name, "model": b.model} for b in backends],
                    "max_workers": max_workers,
                    "image_count": len(image_paths),
                }
            )
            + "\n"
        )
        log.flush()

        futures = {
            ex.submit(_timed_get_crop_plan, image_path, backend): (image_path, backend) for image_path, backend in jobs
        }
        for n, future in enumerate(as_completed(futures), 1):
            image_path, backend = futures[future]
            name = image_path.name
            call = future.result()

            if call.error is not None:
                print(f"  [{n}/{len(jobs)}] {backend.name}/{name}: FAILED ({call.duration_s:.1f}s) -- {call.error}", flush=True)
                log.write(
                    json.dumps(
                        {
                            "type": "call_failed",
                            "image": name,
                            "backend": backend.name,
                            "duration_s": round(call.duration_s, 2),
                            "error": call.error,
                        }
                    )
                    + "\n"
                )
                log.flush()
            else:
                plans_by_image.setdefault(name, {})[backend.name] = call.plan
                durations_by_image.setdefault(name, {})[backend.name] = call.duration_s
                all_durations[backend.name].append(call.duration_s)
                if call.usage:
                    usage_by_image.setdefault(name, {})[backend.name] = call.usage
                    all_usage[backend.name].append(call.usage)
                for raw in call.malformed:
                    log.write(
                        json.dumps({"type": "malformed_decision", "image": name, "backend": backend.name, "raw": raw})
                        + "\n"
                    )
                if call.malformed:
                    log.flush()
                print(f"  [{n}/{len(jobs)}] {backend.name}/{name}: ok ({call.duration_s:.1f}s)", flush=True)

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
                usage_by_image.pop(name, {}),
                output_dir,
                log,
            )
            total += image_total
            flagged.extend(image_flagged)

    print(f"\n{total} decisions logged to {log_path}, {len(flagged)} flagged for review.", flush=True)
    for name in sorted(backend_names):
        durs = all_durations[name]
        if not durs:
            continue
        print(f"  {name}: avg {sum(durs) / len(durs):.1f}s/call over {len(durs)} calls "
              f"(min {min(durs):.1f}s, max {max(durs):.1f}s)", flush=True)
        usages = all_usage[name]
        if usages:
            total_prompt = sum(u["prompt_tokens"] for u in usages)
            total_completion = sum(u["completion_tokens"] for u in usages)
            print(f"    tokens: {total_prompt} prompt + {total_completion} completion "
                  f"({total_completion / len(usages):.0f} completion/call avg)", flush=True)


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
