"""Blind A/B comparison of two refined versions of the same clips (e.g. the agent vs the scripted baseline, or two
agent rounds), with the agent's verifier prompt, over the whole clip split at phase boundaries (<= 60 frames).

  python scripts/refine/compare.py --a refine_script --b refine_r2 VID [VID ...]      (fpose env, one GPU)

The judge sees overlays, side views and a plot of both versions' motion over each chunk.
--a/--b are run roots under $PROJECT_STORAGE_ROOT/runs (each <root>/VID/final.npz); "initial" means the unrefined
FoundationPose + Depth Pro 2 result. Chunks come from the phases of --b (else --a). Writes
runs/compare/<a>_vs_<b>/VID.json and prints per-clip and overall win counts.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from anthropic import AnthropicBedrock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agent import chunks, verify_run  # noqa: E402
from clip import Clip, load_state  # noqa: E402

STORAGE = os.environ.get("PROJECT_STORAGE_ROOT", "/data/sky/projects/CounterfactualHOI")
CHOIR_ROOT = os.environ.get("CHOIR_ROOT", os.path.join(STORAGE, "build", "CHOIR"))


def load(clip, root, vid):
    if root == "initial":
        return clip.init_state.copy()
    pf = os.path.join(STORAGE, "runs", root, vid, "phases.json")
    return load_state(os.path.join(STORAGE, "runs", root, vid, "final.npz"), json.load(open(pf)) if os.path.exists(pf) else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--model", default=os.environ.get("REFINE_MODEL", "global.anthropic.claude-opus-5-5"))
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--tag", default="", help="suffix of the output dir (runs/compare/<a>_vs_<b>_<tag>)")
    ap.add_argument("vids", nargs="+")
    args = ap.parse_args()
    out = os.path.join(STORAGE, "runs", "compare", f"{args.a}_vs_{args.b}" + (f"_{args.tag}" if args.tag else ""))
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    client = AnthropicBedrock(aws_region=args.region)
    total = {"a": 0, "b": 0, "tie": 0}
    for vid in args.vids:
        clip = Clip(os.path.join(CHOIR_ROOT, "output", vid), os.path.join(STORAGE, "runs", "fprecon", vid),
                    table=os.path.join(STORAGE, "runs", args.b, vid, "table_fit.json"))   # b's table when it has its own
        sa, sb = load(clip, args.a, vid), load(clip, args.b, vid)
        phases = sb.phases or sa.phases or [{"start": 0, "end": clip.N - 1, "label": "all"}]
        runs = chunks([0, clip.N - 1], phases)
        # verify_run(s0, s1) labels s1 "refined": here s0 = a, s1 = b; with both versions' per-frame metrics it also
        # shows the judge a plot of the motion (jitter / drift at rest that still frames do not show)
        sb.phases = phases
        ra, rb = clip.metrics(sa), clip.metrics(sb)
        with ThreadPoolExecutor(4) as ex:
            res = list(ex.map(lambda rn: verify_run(client, args, clip, sa, sb, rn, out, vid, ra, rb), runs))
        rows, wins = [], {"a": 0, "b": 0, "tie": 0}
        for run, v in res:
            w = "tie" if v["winner"] == "tie" else ("b" if v["winner"] == v["refined_label"] else "a")
            wins[w] += 1
            rows.append({"frames": run, "winner": w, "confidence": v["confidence"], "reason": v["reason"]})
        for k in total:
            total[k] += wins[k]
        json.dump({"a": args.a, "b": args.b, "wins": wins, "chunks": rows}, open(os.path.join(out, f"{vid}.json"), "w"), indent=1)
        print(f"{vid}: {args.a} {wins['a']}, {args.b} {wins['b']}, tie {wins['tie']}", flush=True)
    print(f"overall: {args.a} {total['a']}, {args.b} {total['b']}, tie {total['tie']}")


if __name__ == "__main__":
    main()
