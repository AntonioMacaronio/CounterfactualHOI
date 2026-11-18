"""Normalize source clips for H3 Ref2VA and build side-by-side QA videos (needs ffmpeg/ffprobe on PATH)."""

import json
import subprocess
from pathlib import Path

# Ref2VA reference-video limits (MiniMax H3 docs): 2-15 s, 23.976-60 fps, MP4/MOV, H.264/H.265.
MAX_REF_SECONDS = 15.0
# iPhone "60 fps" clips are often 60.05 fps (just over the limit); H3 outputs 24 fps anyway.
MAX_REF_FPS = 30


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_frames:format=duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    d = json.loads(out)
    s = d["streams"][0]
    num, den = s["avg_frame_rate"].split("/")
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": float(num) / float(den),
        "frames": int(s.get("nb_frames") or 0),
        "duration": float(d["format"]["duration"]),
    }


def normalize(src: Path, dst: Path) -> dict:
    """iPhone .MOV -> upright H.264 MP4, no audio, fps capped at MAX_REF_FPS, trimmed to the Ref2VA max length.

    ffmpeg applies the rotation flag while decoding, so the output is upright with no rotation metadata.
    Re-encodes if the source is newer than the cached copy (e.g. a clip re-recorded under the same name).
    """
    if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.stem + ".part.mp4")
        fps_filter = ["-vf", f"fps={MAX_REF_FPS}"] if probe(src)["fps"] > MAX_REF_FPS else []
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-t", str(MAX_REF_SECONDS), "-an",
             *fps_filter, "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(tmp)],
            check=True,
        )
        tmp.replace(dst)
    return probe(dst)


def side_by_side(source: Path, generated: Path, out: Path, height: int = 480) -> Path:
    """Source (left) vs generated (right) at a common height and the generated clip's fps, for visual QA."""
    gen_fps = probe(generated)["fps"]
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-i", str(generated),
         "-filter_complex",
         f"[0:v]fps={gen_fps},scale=-2:{height},setsar=1[a];[1:v]scale=-2:{height},setsar=1[b];"
         "[a][b]hstack=inputs=2:shortest=1[v]",
         "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )
    return out
