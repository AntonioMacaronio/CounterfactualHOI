"""Batch video-to-video augmentation with a self-hosted MiniMax H3 Ref2VA server.

  python -m videogen.generate --video-dir ../videos --clips IMG_5690 --variants tall_can \
      --modes identity object object_hand --seeds 0 1

Outputs (default $VIDEOGEN_OUT/gen):
  sources/<clip>.mp4                                normalized reference clip actually sent to H3
  <clip>/<mode>/<variant>/seed<k>.mp4               generated clip (mode=identity uses variant "identity")
  <clip>/<mode>/<variant>/seed<k>.prompt.txt        exact prompt
  <clip>/<mode>/<variant>/seed<k>_compare.mp4       source | generated, for visual QA
  manifest.jsonl                                    one record per attempt, including failures
                                                    (needed for "GPU time per valid demonstration")
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from . import prep
from .h3_client import DEFAULT_URL, H3Client, H3Error
from .prompts import MODES, build_prompt

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/videogen/sources.json")
    p.add_argument("--video-dir", type=Path, default=os.environ.get("REF_VIDEO_DIR"),
                   help="directory holding the raw reference clips (or set REF_VIDEO_DIR)")
    p.add_argument("--out", type=Path, default=Path(os.environ.get("VIDEOGEN_OUT", "outputs/videogen")) / "gen")
    p.add_argument("--clips", nargs="*", help="clip ids from the config (default: all)")
    p.add_argument("--variants", nargs="*", help="variant ids (default: all of each clip's variants)")
    p.add_argument("--modes", nargs="*", default=["object", "object_hand"], choices=MODES)
    p.add_argument("--seeds", nargs="*", type=int, default=[0])
    p.add_argument("--steps", type=int, default=None, help="denoising steps (server default: 50)")
    p.add_argument("--aspect-ratio", default="auto")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--num-gpus", type=int, default=int(os.environ.get("NUM_GPUS", 4)),
                   help="GPUs the server runs on; used to report GPU-seconds")
    p.add_argument("--no-compare", action="store_true", help="skip side-by-side QA videos")
    p.add_argument("--dry-run", action="store_true", help="print jobs and prompts without calling the server")
    return p.parse_args(argv)


def plan_jobs(cfg: dict, args) -> list[dict]:
    clip_ids = args.clips or list(cfg["clips"])
    unknown = set(clip_ids) - set(cfg["clips"])
    if unknown:
        sys.exit(f"unknown clips: {sorted(unknown)}")
    jobs = []
    for clip_id in clip_ids:
        clip = cfg["clips"][clip_id]
        variants = [v for v in clip["variants"] if not args.variants or v["id"] in args.variants]
        for mode in args.modes:
            for variant in ([None] if mode == "identity" else variants):
                for seed in args.seeds:
                    jobs.append({"clip_id": clip_id, "clip": clip, "mode": mode, "variant": variant, "seed": seed})
    return jobs


def main(argv=None):
    args = parse_args(argv)
    cfg = json.loads(args.config.read_text())
    jobs = plan_jobs(cfg, args)
    if not jobs:
        sys.exit("no jobs selected (check --clips/--variants/--modes)")
    print(f"{len(jobs)} job(s)")

    if args.dry_run:
        for j in jobs:
            vid = j["variant"]["id"] if j["variant"] else "identity"
            print(f"\n=== {j['clip_id']} / {j['mode']} / {vid} / seed {j['seed']}\n")
            print(build_prompt(j["clip"], j["mode"], j["variant"]))
        return

    if args.video_dir is None:
        sys.exit("--video-dir (or REF_VIDEO_DIR) is required")
    client = H3Client(args.url)
    if not client.healthy():
        sys.exit(f"H3 server at {args.url} is not healthy; start it with scripts/serve_h3_ref2va.sh")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.jsonl"
    sources = {}
    n_ok = n_fail = n_skip = 0

    for i, j in enumerate(jobs, 1):
        clip_id, clip, mode, variant, seed = j["clip_id"], j["clip"], j["mode"], j["variant"], j["seed"]
        vid = variant["id"] if variant else "identity"
        out_dir = args.out / clip_id / mode / vid
        out_mp4 = out_dir / f"seed{seed}.mp4"
        tag = f"[{i}/{len(jobs)}] {clip_id}/{mode}/{vid}/seed{seed}"
        if out_mp4.exists():
            print(f"{tag}: exists, skipping")
            n_skip += 1
            continue

        if clip_id not in sources:
            src = args.out / "sources" / f"{clip_id}.mp4"
            sources[clip_id] = (src, prep.normalize(args.video_dir / clip["file"], src))
        src, src_info = sources[clip_id]

        prompt = build_prompt(clip, mode, variant)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"seed{seed}.prompt.txt").write_text(prompt)
        record = {
            "clip": clip_id, "task": clip.get("task"), "mode": mode, "variant": vid,
            "category": variant["category"] if variant else None,
            "dims_cm": variant.get("dims_cm") if variant else None,
            "seed": seed, "steps": args.steps,
            "prompt_sha1": hashlib.sha1(prompt.encode()).hexdigest(),
            "source": str(src), "source_info": src_info, "output": str(out_mp4),
            "num_gpus": args.num_gpus, "h3_revision": os.environ.get("H3_REVISION"),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        t0 = time.time()
        try:
            video_id = client.submit_ref2va(
                prompt, [str(src)], duration_seconds=min(src_info["duration"], prep.MAX_REF_SECONDS),
                seed=seed, aspect_ratio=args.aspect_ratio, num_inference_steps=args.steps,
            )
            print(f"{tag}: submitted {video_id}")
            job = client.wait(video_id)
            client.download(video_id, out_mp4)
            wall = time.time() - t0
            infer = job.get("inference_time_s") or wall
            record.update(status="ok", video_id=video_id, wall_s=round(wall, 1),
                          inference_time_s=round(infer, 1), gpu_seconds=round(infer * args.num_gpus, 1),
                          peak_memory_mb=job.get("peak_memory_mb"), output_info=prep.probe(out_mp4))
            if not args.no_compare:
                record["compare"] = str(prep.side_by_side(src, out_mp4, out_dir / f"seed{seed}_compare.mp4"))
            n_ok += 1
            print(f"{tag}: ok in {wall:.0f}s -> {out_mp4}")
        except (H3Error, OSError, ValueError) as e:
            wall = time.time() - t0
            record.update(status="failed", error=str(e)[:2000], wall_s=round(wall, 1),
                          gpu_seconds=round(wall * args.num_gpus, 1))
            n_fail += 1
            print(f"{tag}: FAILED: {e}", file=sys.stderr)

        with open(manifest, "a") as f:
            f.write(json.dumps(record) + "\n")

    print(f"done: {n_ok} ok, {n_fail} failed, {n_skip} skipped; manifest: {manifest}")


if __name__ == "__main__":
    main()
