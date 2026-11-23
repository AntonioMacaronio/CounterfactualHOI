"""Re-run the independent reviewer (agent.critic_review: absolute check of a result against the video, every phase
from four sides) on finished refinement runs, e.g. after changing its prompt. Rewrites VID/critic_final.json and the
reviewer paragraph of VID/report.md; then re-run summarize.py and build_choir_html.py --method refine.

  REFINE_TAG=r5 python scripts/refine/review.py [--samples 3] VID [VID ...]      (fpose env, one GPU)

One review is a noisy judgement (borderline clips flip between acceptable and bad), so --samples N runs it N times:
the verdict is the majority (ties go to the worse one), the problems are those of a run with that verdict, and all
N verdicts are kept ("votes").
"""

import argparse
import json
import os
import sys

from anthropic import AnthropicBedrock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agent import critic_review, critic_text  # noqa: E402
from clip import Clip, load_state  # noqa: E402

STORAGE = os.environ.get("PROJECT_STORAGE_ROOT", "/data/sky/projects/CounterfactualHOI")
CHOIR_ROOT = os.environ.get("CHOIR_ROOT", os.path.join(STORAGE, "build", "CHOIR"))
MARK = "Independent reviewer on the final result (what is still wrong): "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("REFINE_MODEL", "global.anthropic.claude-opus-5-5"))
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("vids", nargs="+")
    args = ap.parse_args()
    tag = "_" + os.environ["REFINE_TAG"] if os.environ.get("REFINE_TAG") else ""
    client = AnthropicBedrock(aws_region=args.region)
    for vid in args.vids:
        o = os.path.join(STORAGE, "runs", "refine" + tag, vid)
        clip = Clip(os.path.join(CHOIR_ROOT, "output", vid), os.path.join(STORAGE, "runs", "fprecon", vid))
        st = load_state(os.path.join(o, "final.npz"), json.load(open(os.path.join(o, "phases.json"))))
        revs = [critic_review(client, args, clip, st, o, f"final_s{k}") for k in range(args.samples)]
        votes = [r["overall"] for r in revs]
        rank = {"good": 0, "acceptable": 1, "bad": 2}
        overall = max(set(votes), key=lambda v: (votes.count(v), rank[v]))
        rev = dict(next(r for r in revs if r["overall"] == overall), votes=votes)
        json.dump(rev, open(os.path.join(o, "critic_final.json"), "w"), indent=1)
        rp = os.path.join(o, "report.md")
        if os.path.exists(rp):
            lines = open(rp).read().split("\n")
            lines = [MARK + f"(votes over {len(votes)} reviews: {', '.join(votes)}) " + critic_text(rev) if ln.startswith(MARK) else ln
                     for ln in lines]
            open(rp, "w").write("\n".join(lines))
        print(f"{vid}: votes {votes} -> {critic_text(rev)[:300]}", flush=True)


if __name__ == "__main__":
    main()
