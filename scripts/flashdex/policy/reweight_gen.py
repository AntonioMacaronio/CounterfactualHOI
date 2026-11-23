"""Set the GenEnv share of a TrackEnv + GenEnv pack by repeating its TrackEnv clips.

  python -I scripts/flashdex/policy/reweight_gen.py SRC.pt DST.pt [--gen-share 0.25]     (flashdex env)

FDR's Isaac env pins each env to one object and draws uniformly among that object's clips at every reset, so with
1 TrackEnv clip and 2 GenEnv clips per object (make_gen_clips.py --variants 2) GenEnv gets 2/3 of the episodes.
This writes a pack where, per object, the TrackEnv clips are repeated so they make up 1 - gen_share of that object's
clips (rounded; e.g. 6 x 1 track : 2 gen = 25% gen). Repeats keep the clip's name, so evals score each clip once.
Per-frame arrays (first dim = total frames) are re-concatenated, per-clip ones (first dim = clips) re-indexed.
"""

import argparse
from collections import defaultdict

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--gen-share", type=float, default=0.25)
    args = ap.parse_args()
    pt = torch.load(args.src, map_location="cpu", weights_only=False)
    names = [str(x) for x in pt["motion_filename"]]
    M = len(names)
    F = int(pt["motion_num_frames"].sum())
    slot = pt["motion_object_slot_right"].tolist()
    by_slot = defaultdict(lambda: ([], []))
    for m, n in enumerate(names):
        by_slot[slot[m]][int("_gen" in n)].append(m)
    order = []
    for s, (track, gen) in sorted(by_slot.items()):
        reps = max(1, round(len(gen) * (1 - args.gen_share) / args.gen_share / max(len(track), 1))) if gen else 1
        order += [m for m in track for _ in range(reps)] + gen
        print(f"slot {s}: {len(track)} track x {reps}, {len(gen)} gen -> gen share "
              f"{len(gen) / (len(track) * reps + len(gen)):.2f}")
    starts, lens = pt["length_starts"].tolist(), pt["motion_num_frames"].tolist()
    frame_idx = torch.cat([torch.arange(starts[m], starts[m] + lens[m]) for m in order])
    out = {}
    for k, v in pt.items():
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == F:
            out[k] = v[frame_idx].contiguous()
        elif torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == M:
            out[k] = v[torch.tensor(order)].contiguous()
        elif isinstance(v, (list, tuple)) and len(v) == M and k in ("motion_filename",):
            out[k] = [v[m] for m in order]
        else:
            out[k] = v
    n = out["motion_num_frames"]
    out["length_starts"] = torch.cat([torch.zeros(1, dtype=n.dtype), n.cumsum(0)[:-1]])
    torch.save(out, args.dst)
    g = sum("_gen" in names[m] for m in order)
    print(f"wrote {args.dst}: {len(order)} clips ({g} GenEnv, share {g / len(order):.2f}), {len(frame_idx)} frames")


if __name__ == "__main__":
    main()
