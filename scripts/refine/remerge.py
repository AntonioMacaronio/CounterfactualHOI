"""Redo the last step of finished refinement runs with the current code: the merge of the agent's result with the
automatic fit on the ranges the verifier rejected (agent.merge), e.g. after a fix to it. No agent or LLM calls.

  REFINE_TAG=r8 python scripts/refine/remerge.py VID [VID ...]      (fpose env, one GPU)
  REFIT_TABLE=1 ...: fit the table again with the current code instead of using the run's (OUT/table_fit.json)

Per clip, from OUT = $PROJECT_STORAGE_ROOT/runs/refine[_TAG]/VID: the base version and the agent's last version
(OUT/versions; reverts in the report's edit list are followed back to a saved version), the verifier's decisions of
the last pass (OUT/report.md) and the phases (OUT/phases.json). Rewrites OUT/final.npz, OUT/export/, OUT/table.json
and, in OUT/report.md, the last pass's paragraph on the merge and the metrics table. Afterwards re-export the viewer
(scripts/fprecon/view_fprecon.py --run-dir OUT/export) and re-run review.py and summarize.py.
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agent import export, merge, merge_text, report_table  # noqa: E402
from clip import Clip, load_state  # noqa: E402

STORAGE = os.environ.get("PROJECT_STORAGE_ROOT", "/data/sky/projects/CounterfactualHOI")
CHOIR_ROOT = os.environ.get("CHOIR_ROOT", os.path.join(STORAGE, "build", "CHOIR"))
MARK = "Merge of the kept and the reverted ranges: "


def main():
    root = os.path.join(STORAGE, "runs", "refine" + ("_" + os.environ["REFINE_TAG"] if os.environ.get("REFINE_TAG") else ""))
    for vid in sys.argv[1:]:
        o = os.path.join(root, vid)
        clip = Clip(os.path.join(CHOIR_ROOT, "output", vid), os.path.join(STORAGE, "runs", "fprecon", vid),
                    table=None if os.environ.get("REFIT_TABLE") else os.path.join(o, "table_fit.json"))
        phases = json.load(open(os.path.join(o, "phases.json")))
        rep = open(os.path.join(o, "report.md")).read()
        edits = {int(k): n for k, n in re.findall(r"^- v(\d+): (.*)$", rep, flags=re.M)}
        k = max(edits)
        while edits[k].startswith("revert to v"):          # reverts are not saved: follow them to a saved version
            k = int(edits[k].split("revert to v")[1].split()[0])
        cut = rep.rindex("Verifier (v")                    # the last pass's section (earlier passes are saved versions)
        head, rep = rep[:cut], rep[cut:]
        base = int(re.search(r"Verifier \(v(\d+) vs refined", rep).group(1))
        ld = lambda j: clip.init_state.copy() if j == 0 else load_state(os.path.join(o, "versions", f"v{j:03d}.npz"), phases)
        s0, s1 = ld(base), ld(k)
        revert = [(int(a), int(b)) for a, b in re.findall(r"^- frames (\d+)-(\d+): REVERTED", rep, flags=re.M)]
        kept = [(int(a), int(b)) for a, b in re.findall(r"^- frames (\d+)-(\d+): kept", rep, flags=re.M)]
        final, seams = merge(clip, s0, s1, revert, kept)
        final.phases = phases
        export(clip, final, o)
        merged = merge_text(clip, final, seams)
        cols = [("initial", clip.metrics(clip.init_state))] + ([("automatic fit (v1)", clip.metrics(ld(1)))] if base else [])
        table = report_table(cols + [("final", clip.metrics(final))])
        rep = re.sub(r"\| metric \(median over frames\).*?(?=\n\n|\Z)", lambda m: table, rep, flags=re.S)
        if MARK in rep:
            rep = re.sub(re.escape(MARK) + r".*", lambda m: merged, rep)
        else:                                              # reports from before the merge paragraph existed
            rep = rep.replace("\n\nIndependent reviewer", "\n\n" + merged + "\n\nIndependent reviewer", 1)
        open(os.path.join(o, "report.md"), "w").write(head + rep)
        print(f"{vid}: agent v{k} vs base v{base}, reverted {revert}. {merged}", flush=True)


if __name__ == "__main__":
    main()
