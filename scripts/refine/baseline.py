"""Non-agentic baseline for the refinement agent: the agent's search tools run as a fixed script, without an agent.

  python scripts/refine/baseline.py VID [VID ...]      (fpose env, one GPU)

Per clip: polish_hand over the whole clip (keypoints + mask + depth + no penetration; no phase labels, so no contact
term), then fit_object_silhouette over the whole clip (default trust region). Writes
$PROJECT_STORAGE_ROOT/runs/refine_script/VID/{final.npz, phases.json, export/} like agent.py, so summarize.py
(REFINE_TAG=script) and compare.py read it. phases.json is copied from the agent's run (runs/refine_r2, else
runs/refine) for the held-phase metrics only; the script itself does not use phases.
"""

import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ops  # noqa: E402
from agent import export  # noqa: E402
from clip import Clip  # noqa: E402

STORAGE = os.environ.get("PROJECT_STORAGE_ROOT", "/data/sky/projects/CounterfactualHOI")
CHOIR_ROOT = os.environ.get("CHOIR_ROOT", os.path.join(STORAGE, "build", "CHOIR"))


def main():
    for vid in sys.argv[1:]:
        t0 = time.time()
        out = os.path.join(STORAGE, "runs", "refine_script", vid)
        os.makedirs(out, exist_ok=True)
        clip = Clip(os.path.join(CHOIR_ROOT, "output", vid), os.path.join(STORAGE, "runs", "fprecon", vid))
        s = clip.init_state.copy()
        s, _, n1 = ops.polish_hand(clip, s, 0, clip.N - 1)
        s, _, n2 = ops.fit_object_silhouette(clip, s, 0, clip.N - 1)
        ph = next((os.path.join(STORAGE, "runs", r, vid, "phases.json") for r in ("refine_r2", "refine")
                   if os.path.exists(os.path.join(STORAGE, "runs", r, vid, "phases.json"))), None)
        s.phases = json.load(open(ph)) if ph else []
        export(clip, s, out)
        if ph:
            shutil.copyfile(ph, os.path.join(out, "phases.json"))
        open(os.path.join(out, "transcript.jsonl"), "w").write(
            json.dumps({"note": f"agent finished at v2 after {(time.time() - t0) / 60:.1f} min; scripted: {n1}; {n2}"}) + "\n")
        print(f"{vid}: {n1}; {n2}; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
