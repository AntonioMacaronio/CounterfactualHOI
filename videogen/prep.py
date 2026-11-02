"""Normalize source clips for H3 Ref2VA and build side-by-side QA videos (needs ffmpeg/ffprobe on PATH)."""

import json
import subprocess
from pathlib import Path

# Ref2VA reference-video limits (MiniMax H3 docs): 2-15 s, 23.976-60 fps, MP4/MOV, H.264/H.265.
MAX_REF_SECONDS = 15.0


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
    """iPhone HEVC .MOV -> H.264 MP4, no audio, same resolution/fps, trimmed to the Ref2VA max length."""
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.stem + ".part.mp4")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-t", str(MAX_REF_SECONDS), "-an",
             "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p",
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
