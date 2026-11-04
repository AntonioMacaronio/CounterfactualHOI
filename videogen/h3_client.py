"""Minimal stdlib client for a self-hosted MiniMax H3 Ref2VA server (SGLang /v1/videos)."""

import json
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = os.environ.get("H3_URL", "http://127.0.0.1:30011")


class H3Error(RuntimeError):
    pass


class H3Client:
    def __init__(self, url: str = DEFAULT_URL, timeout_s: float = 60.0):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def _request(self, method: str, path: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.url + path, data=data, method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        try:
            return urllib.request.urlopen(req, timeout=self.timeout_s)
        except urllib.error.HTTPError as e:
            raise H3Error(f"{method} {path} -> HTTP {e.code}: {e.read().decode(errors='replace')[:2000]}") from e

    def healthy(self) -> bool:
        try:
            with self._request("GET", "/health") as r:
                return json.load(r).get("status") == "ok"
        except (H3Error, urllib.error.URLError, OSError):
            return False

    def wait_healthy(self, timeout_s: float = 1800.0, poll_s: float = 10.0) -> None:
        deadline = time.time() + timeout_s
        while not self.healthy():
            if time.time() > deadline:
                raise H3Error(f"server at {self.url} not healthy after {timeout_s}s")
            time.sleep(poll_s)

    def submit_ref2va(
        self,
        prompt: str,
        reference_videos: list[str],
        duration_seconds: float,
        seed: int = 0,
        aspect_ratio: str = "auto",
        num_inference_steps: int | None = None,
        reference_images: list[str] = (),
    ) -> str:
        """Submit a video-to-video (Ref2VA) job. Reference paths must be readable by the server.

        In the prompt, videos are <Video 1..n> and images are <Picture 1..n>, numbered per type.
        """
        body = {
            "task": "ref2va",
            "prompt": prompt,
            "conditions": [
                {"type": "video", "uri": str(Path(p).resolve()), "role": "reference"}
                for p in reference_videos
            ] + [
                {"type": "image", "uri": str(Path(p).resolve()), "role": "reference"}
                for p in reference_images
            ],
            "target": {"short_edge": 768, "aspect_ratio": aspect_ratio, "duration_seconds": duration_seconds},
            "seed": seed,
        }
        if num_inference_steps is not None:
            body["num_inference_steps"] = num_inference_steps
        with self._request("POST", "/v1/videos", body) as r:
            return json.load(r)["id"]

    def status(self, video_id: str) -> dict:
        with self._request("GET", f"/v1/videos/{video_id}") as r:
            return json.load(r)

    def wait(self, video_id: str, timeout_s: float = 3600.0, poll_s: float = 5.0) -> dict:
        deadline = time.time() + timeout_s
        while True:
            job = self.status(video_id)
            if job["status"] == "completed":
                return job
            if job["status"] in ("failed", "cancelled"):
                raise H3Error(f"job {video_id} {job['status']}: {job.get('error')}")
            if time.time() > deadline:
                raise H3Error(f"job {video_id} still {job['status']} after {timeout_s}s")
            time.sleep(poll_s)

    def download(self, video_id: str, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with self._request("GET", f"/v1/videos/{video_id}/content") as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(out_path)
        return out_path
