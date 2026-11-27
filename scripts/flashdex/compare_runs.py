"""FlashDexRetarget runs side by side at equal env steps (not at equal wall time: runs differ in env count and speed).

  python scripts/flashdex/compare_runs.py NAME=RUN [NAME=RUN ...] [--every 10]      (RUN = $FDR/logs/flashsac/<run>)

Training side, from each RUN/metrics.jsonl (train_cfhoi.py), averaged over +-1M steps around every --every million:
the object-assist gain, the task reward and the clip completion rate the assist curriculum sees, the episode length.
Evaluation side, from RUN/curves.json when scripts/flashdex/eval_curves.py has been run on it: mean AdaDex progress
and the share of clips tracked end to end at each evaluation (evaluations run without the assist, so they only mean
something once the gain is near 0). With CFHOI_SGS=1 the completion rate is that of the episodes SGS chose to
start, which are harder than a uniform draw.
"""

import argparse
import json
from pathlib import Path

KEYS = [("kp", "Curriculum/xfrc_curriculum_adaptive/xfrc_kp_upper", 6, 1),
        ("task", "Curriculum/xfrc_curriculum_adaptive/xfrc_mean_task_rew", 7, 3),
        ("compl", "Curriculum/xfrc_curriculum_adaptive/xfrc_completion_rate", 7, 3),
        ("eplen", "Curriculum/xfrc_curriculum_adaptive/xfrc_mean_ep_len", 7, 1)]


def load(run):
    rows = {}
    for line in open(Path(run) / "metrics.jsonl"):
        try:
            r = json.loads(line)
        except ValueError:                                   # a half-written last line
            continue
        rows.setdefault(int(r["step"]), {}).update(r)
    return [rows[k] for k in sorted(rows)]


def at(rows, step, key, window=1.0e6):
    v = [r[key] for r in rows if key in r and abs(r["step"] - step) <= window]
    return sum(v) / len(v) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=RUN_DIR")
    ap.add_argument("--every", type=float, default=10.0, help="million env steps between rows")
    a = ap.parse_args()
    runs = [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in a.runs]
    data = {n: load(p) for n, p in runs}
    last = {n: max(r["step"] for r in rows) for n, rows in data.items()}
    width = sum(k[2] for k in KEYS)
    print("last logged env step: " + ", ".join(f"{n} {last[n] / 1e6:.1f}M" for n, _ in runs))
    print(f"{'step':>6} | " + " | ".join(f"{n[:width]:>{width}}" for n, _ in runs))
    print(f"{'':>6} | " + " | ".join("".join(f"{k[0]:>{k[2]}}" for k in KEYS) for _ in runs))
    step = a.every * 1e6
    while step <= max(last.values()):
        cells = []
        for n, _ in runs:
            cells.append("".join(f"{v:>{w}.{d}f}" if v is not None else f"{'-':>{w}}"
                                 for v, (_, _, w, d) in ((at(data[n], step, k[1]), k) for k in KEYS)))
        print(f"{step / 1e6:>5.0f}M | " + " | ".join(cells))
        step += a.every * 1e6
    curves = {n: json.load(open(p / "curves.json")) for n, p in runs if (p / "curves.json").exists()}
    if curves:
        steps = sorted({int(s) for c in curves.values() for s in c})
        print("\nevaluations (TrackEnv clips): mean progress % / clips tracked end to end %")
        print(f"{'step':>6} | " + " | ".join(f"{n[:width]:>{width}}" for n in curves))
        for s in steps:
            cells = []
            for n, c in curves.items():
                t = c.get(str(s), {}).get("summary", {}).get("track")
                cells.append(f"{100 * t['progress']:>{width - 8}.0f} / {100 * t['success']:>5.0f}" if t else f"{'-':>{width}}")
            print(f"{s / 1e6:>5.0f}M | " + " | ".join(cells))


if __name__ == "__main__":
    main()
