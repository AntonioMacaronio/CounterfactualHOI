"""Agentic refinement of one 4D hand-object reconstruction: Claude inspects renders against the video and edits the
reconstruction's parameters with tools (no gradient-based optimization of the outputs).

  python scripts/refine/agent.py --video-dir $CHOIR_ROOT/output/VID --run-dir $PROJECT_STORAGE_ROOT/runs/fprecon/VID \
      --out $PROJECT_STORAGE_ROOT/runs/refine/VID [--effort high] [--max-turns 60]      (fpose env, Amazon Bedrock)

Loop: the agent gets an overview (contact sheet, side views, metric plots, flagged frames), labels the motion phases,
then fixes issues one at a time with the edit tools (ops.py); every edit makes a new version and returns before/after
metrics and images of the touched frames, so it can keep or revert. At the end an independent verifier (a fresh call,
same model) compares initial vs refined on every changed frame range (split at phases, at most 60 frames) in random A/B
order and reverts ranges where the initial reconstruction looks better. Outputs in OUT: final.npz (+ export/ for the viewer), versions/, transcript.jsonl,
images/, report.md.

Claude on Amazon Bedrock via the SDK's AnthropicBedrock client (InvokeModel, the account's global inference profile);
model, region and effort are flags. Prompt caching is automatic (top-level cache_control) and old tool results (the
images) are cleared server-side once the context grows (context editing), keeping the history append-only.
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from anthropic import AnthropicBedrock, APIConnectionError, APIStatusError, RateLimitError
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ops  # noqa: E402
import render as R  # noqa: E402
from clip import Clip, State, Versions, flag, phase_of, se3_interp, summarize  # noqa: E402

OBJECTS = {1: "small metal device (a RealSense-style depth camera)", 2: "blue tape roll", 3: "black spatula",
           4: "gray mug", 5: "dish soap bottle"}
TASKS = {"pp": "picks the object up from the table, carries it and places it back down",
         "prp": "picks the object up, rotates it in the hand and places it back down"}
PHASES = ["approach", "grasp", "lift", "carry", "rotate", "place", "release", "rest"]
AXES = ["object_long", "object_mid", "object_short", "world_up", "world_right", "world_forward",
        "camera_x", "camera_y", "camera_z", "ray"]
SIGNALS = ["coverage", "spill", "iou", "depth_res_cm", "depth_bias_cm", "hand_kp_px", "hand_coverage", "hand_spill",
           "hand_depth_bias_cm", "contact_cm", "penetration_cm", "obj_height_cm", "hand_height_cm",
           "ang_speed_deg_per_frame", "speed_cm_per_frame"]
REPORT_KEYS = ["coverage", "spill", "depth_res_cm", "hand_kp_px", "hand_depth_bias_cm", "contact_cm", "penetration_cm",
               "obj_height_cm", "ang_speed_deg_per_frame"]

SYSTEM = """You refine a 4D reconstruction of a hand manipulating a rigid object, filmed by a static camera. The
reconstruction will be retargeted to a robot hand and simulated, so it must match the video and be physically
plausible: the object rests on the table when nobody holds it, is held by the hand while carried, never interpenetrates
the hand or the table, and moves smoothly.

What you are editing (all metric, in the real camera's frame):
- the object's pose in every frame (a textured mesh reconstructed from the first frame);
- the hand's rigid motion in every frame (its finger articulation comes from a hand tracker and stays as is);
- motion phases (approach, grasp, lift, carry, rotate, place, release, rest) that you label.

How to read the images:
- Overlays show the real frame with the rendered object (orange outline and light orange fill), the rendered hand
  (blue outline), the segmentation mask of the object (green outline), detected 2D hand keypoints (white dots) joined
  to the hand model's joints (blue dots), and the object's axes from its center: long = red, mid = green, short = blue.
  "blend" mode paints the textured object render over the frame instead. A correct object pose puts the orange outline
  on the real object's silhouette, including thin parts (handles, the spatula's blade), with the texture lining up.
- "side", "top" and "front" are gravity-aligned virtual views (table in gray with a 5 cm grid): use them for heights,
  contact, and penetration, which the camera view cannot show.
- The segmentation masks and keypoints are evidence, not truth: masks sometimes miss parts of the object or the
  object entirely, keypoints of occluded fingers are guesses. When a mask is clearly wrong, trust what you see in the
  frame. Metrics (coverage/spill/depth vs the masks and the depth map, keypoint error, contact, height) are computed
  from that evidence; use them to find problems and confirm fixes, not as the goal. Heights (obj_height_cm,
  hand_height_cm) are measured from the table surface right around the object in that frame's depth.

Edits (each makes a new version and shows you before/after numbers and images of the touched frames):
- Object: rotate_object about its own long/mid/short axis or a gravity/camera axis, translate_object in world
  (right, forward = away from the camera, up) or camera axes, move_along_ray (changes depth only), interpolate_object
  between two good frames, hold_object_static, snap_object_to_table, smooth_object; propose_object_poses +
  apply_candidate give you the pose tracker's alternative hypotheses at a frame. For precision use the search tools,
  not hand-tuned numbers: fit_object_silhouette (search against the mask outside the hand + depth) and polish_object
  (the tracker's refiner) sharpen poses within a trust region; they will not fix a wrong orientation or a jump.
  Your own rotate/translate edits are for what search cannot do: flips, wrong sides, jumps, lag, structure.
- Hand: polish_hand fits the hand's rigid motion to its keypoints, mask and depth, and keeps it touching the object in
  the held phases you labeled (so label phases first); move_hand and attach_hand_to_object for what the fit cannot
  see. Keypoints of fingers hidden behind the object are guesses, so in held phases contact matters more than them.
- Range edits apply fully on [start, end] and fade out over `falloff` frames on each side.

How to work:
Unless the overview says otherwise, you start from v1, the automatic fit: polish_hand and fit_object_silhouette run
frame by frame over the whole clip. It is usually more precise per frame than the initial reconstruction (v0), and a
verifier will compare your result against v1, chunk by chunk, keeping v1 wherever yours looks worse. The automatic
fit knows nothing about the motion: it fits every frame on its own, so it can make the object jitter or drift while
it rests, leave the hand floating off the object while it is held (no phases yet, so no contact term), follow a bad
mask, or keep a wrong orientation it started from. Fixing that is your job:
1. Look at the overview and label the phases with set_phases first (frame ranges covering the whole clip).
2. Contact: re-run polish_hand over the held phases (grasp to place) so the hand touches the object there.
3. Rest: nobody moves the object before the grasp or after the release, so it should be still there. Where the
   rest frames jitter or drift (obj_height_cm, ang_speed / speed in plot), hold_object_static at a frame whose
   outline fits well, or smooth_object.
4. Structure: a wrong object orientation (a handle on the wrong side, the object flipped), the object lagging or
   jumping during fast rotation, the hand far from its keypoints, interpenetration. For each problem: inspect it
   closely (zoom, several frames, a virtual view), make the smallest edit that fixes it, and check the before/after
   result. Revert if it did not help; never stack edits on a version you have not checked. Compare with v0 (history
   / plot) when the automatic fit looks worse somewhere.
5. Heights: when the image fit and the depth agree, a height of under ~0.7 cm above the table at rest is
   measurement noise: leave it. If the object sits at about the same offset (up to ~2 cm, above or below) in every
   rest phase while its outline and depth fit well, that is usually the mesh's unseen underside or the depth of a
   thin object blending into the table, not a pose error. Leave it if one move_along_ray on the rest frames raises
   depth_res_cm past ~1 cm; the export puts the simulated table at the object's resting height, so a constant
   offset is harmless. What must be fixed is a rest height that differs between rest phases or drifts within one.
6. Before finishing, check the current version phase by phase, with outline and side views of frames from every
   phase and a plot of the motion: while the object is held the hand must touch it (contact_cm ~0, no visible gap in
   the side view) without passing through it; at rest the object sits still on the table; the object's outline
   follows the real one through the whole motion, with no jumps. Fix what fails, then finish. Finish when the
   remaining errors are small compared to the uncertainty of the evidence; do not chase tiny metric changes, and do
   not edit frames that already look right.
Batch your inspections: several frames per inspect call, several independent tool calls per turn."""


def tool(name, desc, props, required=None, strict=False):
    # Not strict: Bedrock caps strict tools (20) and their compiled grammar size; run_tool clamps and validates the
    # inputs and returns problems to the agent as tool errors instead.
    return {"name": name, "description": desc, "strict": strict,
            "input_schema": {"type": "object", "properties": props, "required": required or list(props),
                             "additionalProperties": False}}


INT, NUM = {"type": "integer"}, {"type": "number"}
RANGE = {"start": {"type": "integer", "description": "first frame"}, "end": {"type": "integer", "description": "last frame (inclusive)"}}
FALLOFF = {"falloff": {"type": "integer", "description": "frames over which the edit fades out on each side (0 = sharp)"}}
TOOLS = [
    tool("inspect", "Render overlays or virtual views of up to 9 frames as one image.",
         {"frames": {"type": "array", "items": INT, "description": "frame indices (1-9)"},
          "mode": {"type": "string", "enum": ["outline", "blend", "side", "top", "front"]}}),
    tool("zoom", "Full-resolution crop of one frame's overlay (box in original image pixels).",
         {"frame": INT, "box": {"type": "array", "items": INT, "description": "[x0, y0, x1, y1]"},
          "mode": {"type": "string", "enum": ["outline", "blend"]}}),
    tool("plot", "Plot per-frame signals over a frame range (current vs initial version).",
         {"signals": {"type": "array", "items": {"type": "string", "enum": SIGNALS}}, **RANGE}),
    tool("metrics", "Text table of metric medians over windows of a frame range, plus flagged frames.",
         {**RANGE, "window": {"type": "integer", "description": "frames per row"}}),
    tool("set_phases", "Label the motion phases (ranges should cover the clip).",
         {"phases": {"type": "array", "items": {"type": "object", "properties": {
             "start": INT, "end": INT, "label": {"type": "string", "enum": PHASES}},
             "required": ["start", "end", "label"], "additionalProperties": False}}}),
    tool("rotate_object", "Rotate the object about one of its axes or a world/camera axis through its center.",
         {**RANGE, "axis": {"type": "string", "enum": AXES}, "angle_deg": NUM, **FALLOFF}),
    tool("translate_object", "Translate the object (cm). frame=world: right, forward (away from camera), up; frame=camera: x right, y down, z depth.",
         {**RANGE, "dx_cm": NUM, "dy_cm": NUM, "dz_cm": NUM, "frame": {"type": "string", "enum": ["world", "camera"]}, **FALLOFF}),
    tool("move_along_ray", "Move the object along the camera ray (cm, + = farther): changes depth, keeps image position.",
         {**RANGE, "dist_cm": NUM, **FALLOFF}),
    tool("interpolate_object", "Replace poses strictly between start and end with a smooth interpolation of the two endpoint poses.", dict(RANGE)),
    tool("hold_object_static", "Set the object's pose on [start, end] to its pose at frame ref.", {**RANGE, "ref": INT}),
    tool("snap_object_to_table", "Translate along the table normal so the object's lowest point touches the table.",
         {**RANGE, "mode": {"type": "string", "enum": ["per_frame", "constant"]}}),
    tool("smooth_object", "Temporal smoothing of the object's poses.", {**RANGE, "window": INT}),
    tool("polish_object", "Local refinement of the object pose against mask + depth, per frame, within a trust region.",
         {**RANGE, "max_rot_deg": NUM, "max_trans_cm": NUM}),
    tool("fit_object_silhouette", "Fit the object's pose per frame to its mask (ignoring pixels the hand covers) and the depth by search, within a trust region; for small systematic offsets.",
         {**RANGE, "max_rot_deg": NUM, "max_trans_cm": NUM}),
    tool("propose_object_poses", "The pose tracker's top-k alternative object poses at a frame, rendered for you to choose from.",
         {"frame": INT, "k": {"type": "integer", "description": "2-9"}}),
    tool("apply_candidate", "Use candidate c (from propose_object_poses) at frame and apply the same change on [start, end].",
         {"frame": INT, "candidate": INT, **RANGE, **FALLOFF}),
    tool("move_hand", "Rigidly move the hand (cm; optional rotation about an axis through its centroid, rot_axis 'none' to skip).",
         {**RANGE, "dx_cm": NUM, "dy_cm": NUM, "dz_cm": NUM, "frame": {"type": "string", "enum": ["world", "camera"]},
          "rot_axis": {"type": "string", "enum": ["none"] + AXES}, "rot_deg": NUM, **FALLOFF}),
    tool("polish_hand", "Fit the hand's rigid motion to its 2D keypoints, mask and depth (no penetration), per frame.", dict(RANGE)),
    tool("attach_hand_to_object", "Keep the hand's position fixed relative to the object over [start, end], as at frame ref.", {**RANGE, "ref": INT}),
    tool("history", "List versions with their edits and key metrics.", {}),
    tool("revert", "Make version v the working version again (as a new version).", {"version": INT}),
    tool("finish", "Done: the current version is your result.", {"summary": {"type": "string"}}),
]


class Session:
    def __init__(self, clip, out, args):
        self.clip, self.out, self.args = clip, out, args
        self.v = Versions(clip.init_state)
        self.m = {0: clip.metrics(clip.init_state)}
        self.cands = {}
        self.n_img = 0
        self.done = None
        os.makedirs(os.path.join(out, "images"), exist_ok=True)
        os.makedirs(os.path.join(out, "versions"), exist_ok=True)
        self.base = 0                                  # the version the verifier compares the result against

    def auto_fit(self):
        """v1 = the search tools run over the whole clip (hand fit, then object fit): the agent starts from it."""
        c = self.clip
        st, _, n1 = ops.polish_hand(c, self.v.state, 0, c.N - 1)
        st, _, n2 = ops.fit_object_silhouette(c, st, 0, c.N - 1)
        k = self.v.add(st, f"automatic fit over the whole clip: {n1}; {n2}")
        np.savez(os.path.join(self.out, "versions", f"v{k:03d}.npz"), obj=st.obj, hand_off=st.hand_off, hand_rot=st.hand_rot)
        self.base = k
        return k

    def img(self, im, tag):
        self.n_img += 1
        cv2.imwrite(os.path.join(self.out, "images", f"{self.n_img:04d}_{tag}.jpg"), im)
        return R.encode(im)

    def metrics_of(self, k):
        if k not in self.m:
            self.m[k] = self.clip.metrics(self.v.states[k])
        return self.m[k]

    def clamp(self, i):
        return int(min(max(int(i), 0), self.clip.N - 1))

    def edit(self, fn, *a, **kw):
        """Apply an edit to the working version -> new version; report before/after on the touched frames."""
        before = self.v.cur
        st, touched, note = fn(self.clip, self.v.state, *a, **kw)
        k = self.v.add(st, note)
        np.savez(os.path.join(self.out, "versions", f"v{k:03d}.npz"), obj=st.obj, hand_off=st.hand_off, hand_rot=st.hand_rot)
        if not touched:
            return [{"type": "text", "text": f"v{k}: {note}. No frames changed."}]
        mb, ma = self.metrics_of(before), self.metrics_of(k)
        sel = set(touched)
        lines = [f"v{k} (from v{before}): {note}", f"touched frames {touched[0]}-{touched[-1]} ({len(touched)}). medians before -> after:"]
        for key in REPORT_KEYS:
            b = [r[key] for r in mb if r["frame"] in sel and r.get(key) is not None]
            a = [r[key] for r in ma if r["frame"] in sel and r.get(key) is not None]
            if b and a:
                lines.append(f"  {key}: {np.median(b):.2f} -> {np.median(a):.2f}")
        fb = sum(len(flag(r, phase_of(self.v.states[before], r['frame']))) for r in mb if r["frame"] in sel)
        fa = sum(len(flag(r, phase_of(st, r['frame']))) for r in ma if r["frame"] in sel)
        lines.append(f"  flags on touched frames: {fb} -> {fa}")
        show = [touched[int(j)] for j in np.linspace(0, len(touched) - 1, min(3, len(touched)))]
        tiles = []
        for i in dict.fromkeys(show):
            tiles += [R.overlay(self.clip, self.v.states[before], i, label=f"{i} before (v{before})"),
                      R.overlay(self.clip, st, i, label=f"{i} after (v{k})")]
        return [{"type": "text", "text": "\n".join(lines)}, self.img(R.sheet(tiles, ncols=2), f"v{k}")]

    def run_tool(self, name, a):
        spec = next((t for t in TOOLS if t["name"] == name), None)
        if spec is None:
            raise ValueError(f"unknown tool {name}")
        missing = [k for k in spec["input_schema"]["required"] if k not in a]
        if missing:                                  # tools are not strict, so check what the schema requires
            raise ValueError(f"{name} needs {', '.join(missing)} (got {', '.join(a) or 'no arguments'})")
        c, st = self.clip, self.v.state
        cl = self.clamp
        if name == "inspect":
            fr = [cl(i) for i in a["frames"][:9]]
            mode = a["mode"]
            ims = [R.overlay(c, st, i, mode=mode) if mode in ("outline", "blend") else R.view(c, st, i, mode) for i in fr]
            return [{"type": "text", "text": f"v{self.v.cur}, frames {fr}, mode {mode}"},
                    self.img(R.sheet(ims, ncols=3 if len(ims) > 4 else 2), f"inspect_{mode}")]
        if name == "zoom":
            return [self.img(R.crop(c, st, cl(a["frame"]), a["box"], a["mode"]), "zoom")]
        if name == "plot":
            s, e = cl(a["start"]), cl(a["end"])
            rows = {"initial (v0)": self.metrics_of(0)}
            if self.v.cur:
                rows[f"current (v{self.v.cur})"] = self.metrics_of(self.v.cur)
            return [self.img(R.plot(c, rows, a["signals"][:6], s, e, st.phases), "plot")]
        if name == "metrics":
            s, e = cl(a["start"]), cl(a["end"])
            rows = [r for r in self.metrics_of(self.v.cur) if s <= r["frame"] <= e]
            return [{"type": "text", "text": f"v{self.v.cur}\n" + summarize(c, st, rows, max(1, int(a["window"])))}]
        if name == "set_phases":
            def f(clip, state):
                s2 = state.copy()
                s2.phases = sorted(({"start": cl(p["start"]), "end": cl(p["end"]), "label": p["label"]} for p in a["phases"]),
                                   key=lambda p: p["start"])
                return s2, [], "phases: " + ", ".join(f"{p['label']} {p['start']}-{p['end']}" for p in s2.phases)
            out = self.edit(f)
            rows = self.metrics_of(self.v.cur)
            return out + [{"type": "text", "text": summarize(c, self.v.state, rows)}]
        if name == "rotate_object":
            return self.edit(ops.rotate_object, cl(a["start"]), cl(a["end"]), a["axis"], float(a["angle_deg"]), int(a["falloff"]))
        if name == "translate_object":
            return self.edit(ops.translate_object, cl(a["start"]), cl(a["end"]), a["dx_cm"], a["dy_cm"], a["dz_cm"], a["frame"], int(a["falloff"]))
        if name == "move_along_ray":
            return self.edit(ops.move_along_ray, cl(a["start"]), cl(a["end"]), a["dist_cm"], int(a["falloff"]))
        if name == "interpolate_object":
            return self.edit(ops.interpolate_object, cl(a["start"]), cl(a["end"]))
        if name == "hold_object_static":
            return self.edit(ops.hold_object_static, cl(a["start"]), cl(a["end"]), cl(a["ref"]))
        if name == "snap_object_to_table":
            return self.edit(ops.snap_object_to_table, cl(a["start"]), cl(a["end"]), a["mode"])
        if name == "smooth_object":
            return self.edit(ops.smooth_object, cl(a["start"]), cl(a["end"]), max(3, int(a["window"])))
        if name == "polish_object":
            return self.edit(ops.polish_object, cl(a["start"]), cl(a["end"]), float(a["max_rot_deg"]), float(a["max_trans_cm"]))
        if name == "fit_object_silhouette":
            return self.edit(ops.fit_object_silhouette, cl(a["start"]), cl(a["end"]), float(a["max_rot_deg"]), float(a["max_trans_cm"]))
        if name == "propose_object_poses":
            i, k = cl(a["frame"]), int(min(max(a["k"], 2), 9))
            cands = ops.propose_object_poses(c, st, i, k)
            self.cands[i] = cands
            ims = [R.overlay(c, st, i, label=f"{i} current (v{self.v.cur})")]
            for j, p in enumerate(cands):
                s2 = st.copy()
                s2.obj[i] = p
                ims.append(R.overlay(c, s2, i, label=f"{i} candidate {j}"))
            return [{"type": "text", "text": f"{len(cands)} candidates at frame {i} (0 = tracker's best score); apply one with apply_candidate"},
                    self.img(R.sheet(ims, ncols=3), f"cands_{i}")]
        if name == "apply_candidate":
            i, j = cl(a["frame"]), int(a["candidate"])
            if i not in self.cands or not 0 <= j < len(self.cands[i]):
                raise ValueError(f"no candidate {j} at frame {i}; call propose_object_poses first")
            return self.edit(ops.apply_candidate, i, self.cands[i][j], cl(a["start"]), cl(a["end"]), int(a["falloff"]))
        if name == "move_hand":
            ax = None if a["rot_axis"] == "none" else a["rot_axis"]
            return self.edit(ops.move_hand, cl(a["start"]), cl(a["end"]), a["dx_cm"], a["dy_cm"], a["dz_cm"], a["frame"],
                             ax, float(a["rot_deg"]), int(a["falloff"]))
        if name == "polish_hand":
            return self.edit(ops.polish_hand, cl(a["start"]), cl(a["end"]))
        if name == "attach_hand_to_object":
            return self.edit(ops.attach_hand_to_object, cl(a["start"]), cl(a["end"]), cl(a["ref"]))
        if name == "history":
            lines = []
            for k, note in enumerate(self.v.notes):
                rows = self.m.get(k)
                extra = ""
                if rows:
                    extra = " | " + ", ".join(f"{key} {np.nanmedian([r[key] if r.get(key) is not None else np.nan for r in rows]):.2f}"
                                              for key in ("coverage", "spill", "hand_kp_px", "penetration_cm"))
                lines.append(f"v{k}{' (current)' if k == self.v.cur else ''} <- v{self.v.parents[k]}: {note}{extra}")
            return [{"type": "text", "text": "\n".join(lines)}]
        if name == "revert":
            k = int(a["version"])
            if not 0 <= k < len(self.v.states):
                raise ValueError(f"no version {k}")
            st2 = self.v.states[k].copy()
            nk = self.v.add(st2, f"revert to v{k}")
            self.m[nk] = self.metrics_of(k)
            return [{"type": "text", "text": f"v{nk} = v{k} is now the working version"}]
        if name == "finish":
            self.done = a["summary"]
            return [{"type": "text", "text": "ok"}]
        raise ValueError(f"unknown tool {name}")


def overview(sess):
    c, st, k = sess.clip, sess.v.state, sess.v.cur
    fr = [int(x) for x in np.linspace(0, c.N - 1, 12)]
    side = [int(x) for x in np.linspace(0, c.N - 1, 6)]
    rows = sess.metrics_of(k)
    vid = c.vid
    task = TASKS[vid.split("_")[0]]
    obj = OBJECTS.get(int(vid.split("_")[-1]), "object")
    text = (f"Clip {vid}: a {c.side} hand {task}. Object: {obj}. {c.N} frames at {c.fps:.0f} fps, {c.W}x{c.H}. "
            f"Object size (long/mid/short axes): {', '.join(f'{100 * e:.1f}' for e in c.extent)} cm.\n"
            + (f"You start from v{k}, the automatic fit ({sess.v.notes[k]}); v0 is the initial reconstruction.\n"
               if k else "You start from v0, the initial reconstruction (no automatic fit; the verifier compares against "
                         "v0): polish_hand over the whole clip early on is usually worth it.\n")
            + f"v{k} metrics:\n" + summarize(c, st, rows) +
            f"\nImages (v{k}): (1) 12 frames, outline overlays; (2) 6 frames, side views; (3) per-frame signals"
            + (" (v0 for comparison)." if k else "."))
    return [{"type": "text", "text": text},
            sess.img(R.sheet([R.overlay(c, st, i) for i in fr], ncols=4), "overview"),
            sess.img(R.sheet([R.view(c, st, i, "side") for i in side], ncols=3), "overview_side"),
            sess.img(R.plot(c, {f"v{k}": rows, **({"initial (v0)": sess.metrics_of(0)} if k else {})},
                            ["coverage", "spill", "hand_kp_px", "contact_cm", "obj_height_cm", "ang_speed_deg_per_frame"]),
                     "overview_plot")]


def call(client, args, log, **kw):
    for attempt in range(6):
        try:
            with client.beta.messages.stream(model=args.model, max_tokens=32000, betas=["context-management-2025-06-27"],
                                             thinking={"type": "adaptive", "display": "summarized"},
                                             output_config={"effort": args.effort}, cache_control={"type": "ephemeral"},
                                             **kw) as s:
                return s.get_final_message()
        except (RateLimitError, APIConnectionError) as e:
            log(f"retry {attempt}: {type(e).__name__}")
            time.sleep(10 * (attempt + 1))
        except APIStatusError as e:
            if e.status_code >= 500 and attempt < 5:
                log(f"retry {attempt}: HTTP {e.status_code}")
                time.sleep(10 * (attempt + 1))
            else:
                raise
    raise RuntimeError("Bedrock: retries exhausted")


def agent_loop(sess, client, args, log):
    msgs = [{"role": "user", "content": overview(sess)}]
    ctx = {"edits": [{"type": "clear_tool_uses_20250919", "trigger": {"type": "input_tokens", "value": 140000},
                      "keep": {"type": "tool_uses", "value": 8}}]}
    t_start = time.time()
    for turn in range(args.max_turns):
        t0 = time.time()
        r = call(client, args, log, system=SYSTEM, tools=TOOLS, messages=msgs, context_management=ctx)
        msgs.append({"role": "assistant", "content": r.content})
        uses = [b for b in r.content if b.type == "tool_use"]
        think = " ".join(b.thinking for b in r.content if b.type == "thinking" and b.thinking)
        text = " ".join(b.text for b in r.content if b.type == "text")
        log({"turn": turn, "stop": r.stop_reason, "seconds": round(time.time() - t0, 1), "thinking": think[:3000],
             "text": text[:3000], "tools": [{"name": u.name, "input": u.input} for u in uses],
             "usage": {"in": r.usage.input_tokens, "out": r.usage.output_tokens,
                       "cache_read": r.usage.cache_read_input_tokens, "cache_write": r.usage.cache_creation_input_tokens}})
        if r.stop_reason == "refusal":
            log("refusal: stopping with the current version")
            break
        if not uses:
            if sess.done is None:
                msgs.append({"role": "user", "content": "Continue, or call finish if you are done."})
                continue
            break
        results = []
        for u in uses:
            try:
                content = sess.run_tool(u.name, u.input)
                results.append({"type": "tool_result", "tool_use_id": u.id, "content": content})
                log({"tool": u.name, "result": next((b["text"] for b in content if b["type"] == "text"), "")[:2000]})
            except Exception as e:                                   # report to the agent, keep going
                results.append({"type": "tool_result", "tool_use_id": u.id, "is_error": True, "content": f"{type(e).__name__}: {e}"})
                log({"tool": u.name, "error": f"{type(e).__name__}: {e}"})
        if sess.done is not None:
            msgs.append({"role": "user", "content": results})
            break
        elapsed = (time.time() - t_start) / 60
        if turn >= args.max_turns - 3 or elapsed > args.max_minutes:
            results.append({"type": "text", "text": "Budget almost used up: wrap up and call finish within the next turn."})
        msgs.append({"role": "user", "content": results})
    return sess.v.cur


def chunks(run, phases, max_len=60):
    """Split a changed frame range at phase boundaries and into pieces of at most max_len frames, so the verifier
    judges (and can revert) parts of a long edit separately."""
    a, b = run
    cuts = sorted({p["start"] for p in phases if a < p["start"] <= b})
    pieces, s = [], a
    for c in cuts + [b + 1]:
        while c - s > max_len:
            pieces.append([s, s + max_len - 1])
            s += max_len
        if c > s:
            pieces.append([s, c - 1])
        s = c
    return pieces


def changed_runs(clip, s0, s1, rot_deg=1.0, trans_cm=0.3, hand_cm=0.5, gap=6):
    ch = []
    for i in range(clip.N):
        dR = np.degrees(Rotation.from_matrix(s1.obj[i, :3, :3] @ s0.obj[i, :3, :3].T).magnitude())
        dt = 100 * np.linalg.norm(s1.obj[i, :3, 3] - s0.obj[i, :3, 3])
        dh = 100 * np.linalg.norm(clip.hand_verts(s1, i).mean(0) - clip.hand_verts(s0, i).mean(0))
        dhr = np.degrees(np.linalg.norm(s1.hand_rot[i] - s0.hand_rot[i]))
        if dR > rot_deg or dt > trans_cm or dh > hand_cm or dhr > 3:
            ch.append(i)
    runs = []
    for i in ch:
        if runs and i - runs[-1][1] <= gap:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return runs


VERIFY_SCHEMA = {"type": "object", "properties": {
    "winner": {"type": "string", "enum": ["A", "B", "tie"]},
    "confidence": {"type": "number", "description": "0-1"},
    "reason": {"type": "string"}}, "required": ["winner", "confidence", "reason"], "additionalProperties": False}


VERIFY_SIGNALS = ["obj_height_cm", "ang_speed_deg_per_frame", "speed_cm_per_frame", "contact_cm"]


def verify_run(client, args, clip, s0, s1, run, out, tag, rows0=None, rows1=None):
    """Blind A/B judgement of a base version (s0) vs the refined one (s1) on one frame range. rows0/rows1 (per-frame
    metrics of the whole clip) add a plot of the motion over the range, which still frames cannot show (jitter,
    drift at rest)."""
    a, b = run
    fr = sorted(set(int(x) for x in np.linspace(a, b, min(4, b - a + 1))))
    order = random.Random(hash((clip.vid, a, b))).random() < 0.5
    A, B = (s1, s0) if order else (s0, s1)
    tiles = []
    for i in fr:
        tiles += [R.overlay(clip, A, i, label=f"frame {i}: A"), R.overlay(clip, B, i, label=f"frame {i}: B")]
    views = []
    for i in fr[:2]:
        views += [R.view(clip, A, i, "side", label=f"frame {i} side: A"), R.view(clip, B, i, "side", label=f"frame {i} side: B")]
    im1, im2 = R.sheet(tiles, ncols=2), R.sheet(views, ncols=2)
    ims = [im1, im2]
    if rows0 is not None and rows1 is not None:
        lo, hi = max(0, a - 8), min(clip.N - 1, b + 8)
        ims.append(R.plot(clip, {"A": rows1 if order else rows0, "B": rows0 if order else rows1}, VERIFY_SIGNALS, lo, hi,
                          [p for p in s1.phases if p["end"] >= lo and p["start"] <= hi]))
    cv2.imwrite(os.path.join(out, "images", f"verify_{tag}_{a}-{b}.jpg"),
                np.vstack([cv2.resize(im, (1280, int(im.shape[0] * 1280 / im.shape[1]))) for im in ims]))
    prompt = (f"Two reconstructions (A and B) of the same video frames {a}-{b}: a hand and a rigid object "
              f"({OBJECTS.get(int(clip.vid.split('_')[-1]), 'object')}). Overlays: orange outline/fill = rendered object, blue = rendered "
              "hand, green = object segmentation mask (can be wrong), white dots = detected hand keypoints, blue dots = hand "
              "model joints. Side views: gravity-aligned, table in gray. Which reconstruction matches the video better and is "
              "more physically plausible (object on the table when not held, hand touching the object when holding it, no "
              "interpenetration)? Judge from the images; answer tie if they are equally good or the difference is negligible.")
    if len(ims) > 2:
        prompt += (f" The plot shows A's and B's motion over frames {max(0, a - 8)}-{min(clip.N - 1, b + 8)}: the object's lowest "
                   "point above the table, its rotation and translation speed per frame, and the hand's distance to the object "
                   "(0 = touching). A real object does not jitter or drift while it rests on the table, and moves smoothly "
                   "while held; spikes there are reconstruction errors.")
    r = client.messages.create(model=args.model, max_tokens=8000, output_config={"effort": "high", "format": {"type": "json_schema", "schema": VERIFY_SCHEMA}},
                               messages=[{"role": "user", "content": [R.encode(im) for im in ims] + [{"type": "text", "text": prompt}]}])
    v = json.loads(next(b.text for b in r.content if b.type == "text"))
    refined = "A" if order else "B"
    v["refined_label"] = refined
    v["keep_refined"] = not (v["winner"] not in ("tie", refined) and v["confidence"] >= 0.6)
    return run, v


def merge(clip, s0, s1, runs_to_revert, blend=3):
    s = s1.copy()
    for a, b in runs_to_revert:
        w = ops.weights(clip.N, a, b, blend)
        for i in np.nonzero(w > 0)[0]:
            s.obj[i] = se3_interp(s1.obj[i], s0.obj[i], w[i])
            s.hand_off[i] = (1 - w[i]) * s1.hand_off[i] + w[i] * s0.hand_off[i]
            s.hand_rot[i] = (Rotation.from_rotvec(s1.hand_rot[i]) if w[i] < 0.5 else Rotation.from_rotvec(s0.hand_rot[i])).as_rotvec()
    return s


def export(clip, state, out):
    """Write the result and an export/ run dir (the fprecon run with refined object poses + hand motion) for the viewer."""
    np.savez(os.path.join(out, "final.npz"), obj=state.obj, hand_off=state.hand_off, hand_rot=state.hand_rot)
    json.dump(state.phases, open(os.path.join(out, "phases.json"), "w"), indent=1)
    ex = os.path.join(out, "export")
    os.makedirs(ex, exist_ok=True)
    for f in os.listdir(clip.run):
        if f in ("object_poses.npy", "hand_rigid.npz"):
            continue
        dst = os.path.join(ex, f)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(clip.run, f), dst)
    np.save(os.path.join(ex, "object_poses.npy"), state.obj)
    np.savez(os.path.join(ex, "hand_rigid.npz"), hand_off=state.hand_off, hand_rot=state.hand_rot)
    # The table for simulation: the fitted plane's normal, at the height the object rests at (the mesh's unseen
    # underside and the depth near the object can put the object's lowest point ~1 cm off the measured surface).
    n = clip.table_n
    rest = [i for i in range(clip.N) if phase_of(state, i) in ("approach", "release", "rest")]
    low = [float((clip.sub_v @ state.obj[i, :3, :3].T + state.obj[i, :3, 3]).dot(n).min()) for i in rest]
    meas = [clip.local_table_c(state, i) for i in rest]
    c = float(np.median(low)) if low else float(n @ clip.table_p)
    json.dump({"normal_up": n.tolist(), "offset": c, "point": (c * n).tolist(),
               "rest_offset_vs_measured_cm": 100 * float(np.median(np.subtract(low, meas))) if low else None,
               "source": "object's lowest point at rest" if low else "fitted table plane"},
              open(os.path.join(out, "table.json"), "w"), indent=1)
    if os.path.lexists(os.path.join(ex, "table.json")):
        os.remove(os.path.join(ex, "table.json"))
    os.symlink(os.path.join(out, "table.json"), os.path.join(ex, "table.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("REFINE_MODEL", "global.anthropic.claude-opus-5-5"))
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--max-minutes", type=float, default=40)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--start", default="auto", choices=["auto", "initial"],
                    help="auto: start the agent from the search tools run over the whole clip (v1) and verify against it")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tlog = open(os.path.join(args.out, "transcript.jsonl"), "a")

    def log(x):
        rec = x if isinstance(x, dict) else {"note": x}
        rec["t"] = time.strftime("%H:%M:%S")
        tlog.write(json.dumps(rec, default=float) + "\n")
        tlog.flush()
        if "turn" in rec:
            print(f"[{rec['t']}] turn {rec['turn']} ({rec['seconds']}s): " + ", ".join(t["name"] for t in rec["tools"]) +
                  (f" | {rec['text'][:160]}" if rec["text"] else ""), flush=True)
        elif "error" in rec:
            print(f"   tool {rec['tool']} error: {rec['error'][:200]}", flush=True)
        elif "note" in rec:
            print(f"   {rec['note']}", flush=True)

    t0 = time.time()
    clip = Clip(args.video_dir, args.run_dir)
    sess = Session(clip, args.out, args)
    client = AnthropicBedrock(aws_region=args.region)
    log(f"loaded {clip.vid} ({clip.N} frames) in {time.time() - t0:.0f}s; model {args.model}, effort {args.effort}")
    if args.start == "auto":
        t1 = time.time()
        sess.auto_fit()
        log(f"automatic fit (v{sess.base}) in {time.time() - t1:.0f}s: {sess.v.notes[sess.base]}")
    final_k = agent_loop(sess, client, args, log)
    s0, s1 = sess.v.states[sess.base], sess.v.states[final_k]       # verify against the version the agent started from
    runs = [piece for run in changed_runs(clip, s0, s1) for piece in chunks(run, s1.phases)]
    log(f"agent finished at v{final_k} after {(time.time() - t0) / 60:.1f} min; changed ranges vs v{sess.base} {runs}")
    verdicts = []
    if runs and not args.no_verify:
        r0, r1 = sess.metrics_of(sess.base), sess.metrics_of(final_k)
        with ThreadPoolExecutor(4) as ex:
            verdicts = list(ex.map(lambda rn: verify_run(client, args, clip, s0, s1, rn, args.out, "final", r0, r1), runs))
        for run, v in verdicts:
            log(f"verify {run}: winner {v['winner']} (refined = {v['refined_label']}), confidence {v['confidence']:.2f}, keep {v['keep_refined']}: {v['reason'][:200]}")
    revert = [run for run, v in verdicts if not v["keep_refined"]]
    final = merge(clip, s0, s1, revert)
    final.phases = s1.phases
    export(clip, final, args.out)
    cols = [("initial", sess.metrics_of(0))] + ([(f"start (v{sess.base})", sess.metrics_of(sess.base))] if sess.base else [])
    cols.append(("final", clip.metrics(final)))
    med = lambda rows, k: np.nanmedian([r[k] if r.get(k) is not None else np.nan for r in rows])
    table = (f"| metric (median over frames) | {' | '.join(n for n, _ in cols)} |\n|---|{'---|' * len(cols)}\n"
             + "\n".join(f"| {k} | " + " | ".join(f"{med(m, k):.2f}" for _, m in cols) + " |" for k in REPORT_KEYS))
    report = (f"# Refinement of {clip.vid}\n\nModel {args.model}, effort {args.effort}; {sess.v.cur + 1} versions; "
              f"{(time.time() - t0) / 60:.1f} min.\n\nAgent summary: {sess.done or '(none)'}\n\n"
              f"Phases: {', '.join(p['label'] + ' ' + str(p['start']) + '-' + str(p['end']) for p in s1.phases)}\n\n"
              "Edits:\n" + "\n".join(f"- v{k}: {n}" for k, n in enumerate(sess.v.notes)) +
              f"\n\nVerifier (v{sess.base} vs refined, blind A/B):\n" +
              "\n".join(f"- frames {run[0]}-{run[1]}: {'kept' if v['keep_refined'] else 'REVERTED'} (winner {v['winner']}, refined = {v['refined_label']}, "
                        f"confidence {v['confidence']:.2f}): {v['reason']}" for run, v in verdicts) +
              f"\n\n{table}\n")
    open(os.path.join(args.out, "report.md"), "w").write(report)
    log(f"done in {(time.time() - t0) / 60:.1f} min; reverted {len(revert)}/{len(runs)} ranges")
    print(report)


if __name__ == "__main__":
    main()
