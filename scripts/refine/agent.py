"""Agentic refinement of one 4D hand-object reconstruction: Claude inspects renders against the video and edits the
reconstruction's parameters with tools. The hand is MANO parameters only (global orientation, translation, 15 joint
rotations; no vertex edits); fit_mano is a gradient-based fit of them that the agent runs on ranges it chooses.

  python scripts/refine/agent.py --video-dir $CHOIR_ROOT/output/VID --run-dir $PROJECT_STORAGE_ROOT/runs/fprecon/VID \
      --out $PROJECT_STORAGE_ROOT/runs/refine/VID [--effort high] [--max-turns 60]      (fpose env, Amazon Bedrock)

Loop: the agent gets an overview (contact sheet, side views, metric plots, flagged frames) and labels the motion
phases; the harness then searches the object's extent along the direction the first frame never saw (the mesh comes
from that frame; fit_object_thickness). The agent fixes issues one at a time with the edit tools (ops.py); every edit
makes a new version and returns before/after metrics and images of the touched frames, so it can keep or revert. An
independent reviewer (a fresh call, same model) judges the result in absolute terms before the agent may finish. Then
an independent verifier compares the result with the automatic fit on every changed frame range in random A/B order
(the span in which the hand holds the object is one range; the rest is split at phases, at most 60 frames) and the
ranges where the automatic fit looks better are set back to it. The reviewer checks that merged result, and a second
pass of the agent works on what it still finds (--passes). Outputs in OUT: final.npz, versions/, transcript.jsonl,
images/, report.md, phases.json, table.json and export/ (the fprecon run with the refined result, for the viewer and
retargeting: object_poses.npy, mano_params.npz = global_orient / transl / hand_pose per frame + betas, valid, side,
all in the camera frame, hand_verts.npy, object_shape.json and, when the object's shape was changed, the scaled
object_mesh.obj).

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
from clip import FINGERS, SHAPE_AXES, Clip, State, Versions, flag, jump_threshold, phase_of, se3_interp, summarize  # noqa: E402

OBJECTS = {1: "small metal device (a RealSense-style depth camera)", 2: "blue tape roll", 3: "black spatula",
           4: "gray mug", 5: "dish soap bottle"}
TASKS = {"pp": "picks the object up from the table, carries it and places it back down",
         "prp": "picks the object up, rotates it in the hand and places it back down"}
PHASES = ["approach", "grasp", "lift", "carry", "rotate", "place", "release", "rest"]
AXES = ["object_long", "object_mid", "object_short", "world_up", "world_right", "world_forward",
        "camera_x", "camera_y", "camera_z", "ray"]
SIGNALS = ["coverage", "spill", "iou", "depth_res_cm", "depth_bias_cm", "hand_kp_px", "hand_coverage", "hand_spill",
           "hand_depth_bias_cm", "contact_cm", "penetration_cm", "n_cross", "obj_height_cm", "hand_height_cm",
           "ang_speed_deg_per_frame", "speed_cm_per_frame", "hand_accel_mm", "hand_rel_accel_mm", "tip_accel_mm", "finger_accel_deg",
           "hand_over_obj", "obj_over_hand",
           "obj_accel_cm", "obj_ang_accel_deg", "thumb_gap_cm", "n_touching", "opposed"]
REPORT_KEYS = ["coverage", "spill", "depth_res_cm", "hand_kp_px", "hand_depth_bias_cm", "contact_cm", "penetration_cm",
               "n_cross", "hand_over_obj", "obj_over_hand", "obj_height_cm", "ang_speed_deg_per_frame", "hand_accel_mm", "tip_accel_mm",
               "finger_accel_deg", "obj_ang_accel_deg",
               "thumb_gap_cm", "n_touching", "opposed"]

SYSTEM = """You refine a 4D reconstruction of a hand manipulating a rigid object, filmed by a static camera. The
reconstruction will be retargeted to a robot hand and simulated, so it must match the video and be physically
plausible: the object rests on the table when nobody holds it, is held by the hand while carried, never interpenetrates
the hand or the table, and the object, the hand and every finger move smoothly (real hands do not jitter; a robot
replaying a jittery trajectory shakes).

What you are editing (all metric, in the real camera's frame):
- the object's pose in every frame (a textured mesh reconstructed from the first frame);
- the object's shape: because the mesh comes from that one frame, its extent along the direction that frame looked
  along (the side it never saw) is a guess. A scale of the mesh along that direction, and along the two directions
  across that view, is part of the reconstruction;
- the hand, as the MANO hand model's parameters in every frame: its global orientation and position and its 15
  joint rotations (the hand model's own articulation, so the hand always stays a valid human hand; there are no
  vertex edits);
- motion phases (approach, grasp, lift, carry, rotate, place, release, rest) that you label.

How to read the images:
- Overlays show the real frame with the rendered object (orange outline and light orange fill), the rendered hand
  (blue outline), the segmentation mask of the object (green outline), detected 2D hand keypoints (white dots) joined
  to the hand model's joints (blue dots), and the object's axes from its center: long = red, mid = green, short = blue.
  "blend" mode paints the textured object render over the frame instead. A correct object pose puts the orange outline
  on the real object's silhouette, including thin parts (handles, the spatula's blade), with the texture lining up.
- "side", "top" and "front" are gravity-aligned virtual views (table in gray with a 5 cm grid), and view3d shows the
  scene from any side (azimuth 0 = from the camera's right, 90 = from behind the object looking back at the camera,
  180 = from the left, 270 = from the camera's side; elevation; zoom "contact" for a close-up of where hand and object
  meet). In these views the hand is blue, RED where it passes through the object's surface or is inside it, and GREEN
  where it touches it. The camera view cannot show depth: fingers that look wrapped around an object in the video can
  be inside it or through its wall (a mug, a tape roll) in 3D, which only other viewpoints reveal. n_cross counts the
  hand's triangles that cross the object's surface (0 is right; a few at a firm grip are fine).
- Grip: contact_cm is the closest hand point, so one fingertip resting on the object reads as contact. opposed says
  whether the thumb or the palm and a finger touch on opposite sides (a grip that could lift the object; a hook grip,
  fingers through a mug's handle, holds without it, so check the video);
  n_touching counts fingers touching; thumb_gap_cm is the thumb's distance. While the object is lifted, carried or
  rotated, opposed must be true. hand_depth_bias_cm is how far the hand is in front of (< 0) or behind the depth map:
  a hand floating in front of the object while its fingers really wrap the far side shows a large negative value.
- Jitter: still images cannot show it. hand_accel_mm (frame-to-frame acceleration of the hand mesh, mm/frame^2),
  hand_rel_accel_mm (the same relative to the object), finger_accel_deg and obj_ang_accel_deg / obj_accel_cm measure
  it; the overview gives the hand tracker's own smoothness as the reference (refined motion should not be much
  jitterier). view_motion shows a range's wrist and fingertip paths in one fixed 3D view (smooth curves = good,
  zig-zags = jitter) and a filmstrip of consecutive frames from one viewpoint.
- Occlusion order: hand_over_obj is the share of the rendered hand that is drawn in front of the object where the
  video shows the object (the real fingers are behind it there); obj_over_hand the share that is hidden behind the
  object where the video shows the hand. Both should be ~0; a flag at 5%. This is direct evidence from the video
  of which side of the object the fingers are on, which the depth map cannot give at this scale.
- Table: up for the world axes, the side / top / front views and the export is the normal of the table plane, fitted
  to the depth points around the object where it rests at the start and at the end of the clip (GeoCalib's gravity
  estimate is only its starting point; the depth map's table is not exactly one plane, so the plane is fitted through
  both places). Heights are measured from that plane right around the object.
- Shape: obj_height_cm at rest says whether the object sits on the table (0), in it (< 0) or above it. An object that
  rests in the table by the same amount in every rest phase while its outline and depth fit, or that the hand can only
  grip by leaving its keypoints (the fingers cannot get behind it, the hand stays in front of the depth map:
  hand_depth_bias_cm < 0), has the wrong extent on the side the camera never saw: no hand pose can grip it as the
  video shows, and a simulated copy would sink into the table. That is an error of the object's shape, not of the
  hand and not noise.
- Jumps: tip_accel_mm is the acceleration of the fastest fingertip. hand_accel_mm is a mean over the whole hand and a
  median over frames hides single frames, so a hand or a finger that jumps between two frames (the end of an edit's
  range, a fit that changed its solution from one frame to the next) only shows in tip_accel_mm: a spike of 10+
  mm/frame^2 on one to three frames, flagged "finger jump". In the viewer and on a robot such a jump is the most
  visible error of all. Every edit reports the fastest fingertip's acceleration around the frames it touched and
  WARNS when the edit added a jump (to the hand or the object): fix a jump right away, before any other work.
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
- Object shape: fit_object_thickness searches the object's extent along the first frame's viewing direction and shows
  its evidence for every candidate (rest height, mask and depth fit, and whether a grip fit reaches an opposed grip
  and at what keypoint error); scale_object scales the mesh by a factor you choose along "unseen" (that viewing
  direction) or along "seen_long" / "seen_short" (the object's longer / shorter extent across that view). Both keep
  the side of the object that faces the camera in place and re-fit the object's poses and the hand over the whole
  clip, so do them before detailed work (they undo hold_object_static / smooth_object edits).
- Hand: fit_mano is a gradient-based fit of all MANO parameters of a frame range at once: 2D keypoints, the hand's
  mask and the depth map, MANO's hand-pose prior, smoothness over time, no penetration into the object or the table,
  and in the held phases you labeled (so label phases first) a grip: the thumb or palm and a finger on the object's
  surface. mode "fingers" fits only the joint rotations, "rigid" only the global pose; contact false turns the grip
  term off. Because it is smooth over time and fits the whole range jointly, it is the main hand tool: use it per
  phase (grasp to place for the grip). It keeps the frames just outside its range fixed and joins them smoothly, and
  its smoothness covers the fingertips and the wrist, so it leaves no seam; a pose that must change at the range's
  end changes over the first / last few frames inside the range (include a few more frames if those matter). separate_hand pushes the hand out along the surface normal where the palm
  passes through the object; move_hand_to_depth jumps the hand along the camera ray by its measured depth error (to
  the side of the object the depth map puts it on, which a local fit cannot reach without passing through the
  object; then fit_mano); curl_finger (turns the named fingers' MANO joints) and move_hand / attach_hand_to_object
  for what the fit cannot see (move_hand_to_depth and attach_hand_to_object fade their change out over 6 frames
  beyond the range; move_hand and curl_finger need a falloff of 4+ frames unless the range's ends are already
  jumps you are removing); smooth_hand smooths the MANO parameters over a range (it re-checks that the hand stays
  out of the object). Keypoints of fingers hidden behind the object are guesses, so in held phases contact and
  no crossing matter more than them.
- Range edits apply fully on [start, end] and fade out over `falloff` frames on each side.

How to work:
Unless the overview says otherwise, you start from v1, the automatic fit: fit_object_silhouette frame by frame
(smoothed over time), smooth_object over the whole clip (window 5: the pose tracker's frame-to-frame noise), then
fit_mano over the whole clip without the grip term (it knows no phases yet), and once more with the occlusion term
where that puts the hand on the right side of the object without costing keypoint fit.
It is usually more precise per frame than the initial reconstruction (v0), and a
verifier will compare your result against v1, chunk by chunk, keeping v1 wherever yours looks worse. The automatic
fit knows nothing about the motion: it fits the object frame by frame, so it can make it jitter or drift while it
rests, leaves the hand floating off the object while it is held (no phases yet, so no grip term), follows a bad
mask, or keeps a wrong orientation it started from. Fixing that is your job:
1. Look at the overview and label the phases with set_phases first (frame ranges covering the whole clip). Right
   after your first set_phases the harness runs fit_object_thickness once and shows you its evidence table. Check its
   decision against what you know: compare the object's size in cm with what such an object really measures and with
   the hand next to it, and look at a rest frame and a held frame from the side (view3d). Correct it with
   scale_object if it is clearly off (too thick to be held as in the video, sunk into the table, or squashed).
2. Grip: while held, the hand must grip the object (opposed), not rest a fingertip on it. fit_mano over the held
   phases (grasp to place) so the hand and its fingers grip the object there; it already weighs the depth map against
   the keypoints, so a remaining hand_depth_bias_cm of 1-3 cm is usually the evidence disagreeing, not an error. Use
   move_hand_to_depth only when the hand is clearly on the wrong side of the object (then fit_mano again). Where held
   frames are still not opposed, raise contact_weight (3-10), or move_hand first so the thumb and the fingers can
   reach opposite sides, then fit_mano again. If a grip is only reached by pulling the hand off its keypoints
   (hand_kp_px up by 5+ px, fingers where the video shows none), the fingers are not the problem: the object is too
   thick or at the wrong depth for the hand (step 5). Fix that, not the hand. Where the hand is flagged in front of
   the object while the video shows the object (hand_over_obj), the hand is on the wrong side of it: a fit cannot
   carry it through the object, so move_hand_to_depth (or move_hand along camera z) first, then fit_mano with
   occlusion_weight 1-3 and check that hand_kp_px stays. Then look at held frames in 3D from at least three sides (view3d, azimuths
   0, 90, 180, 270; zoom "contact"): fingers must wrap on the outside of the object, none red; where the palm is red
   use separate_hand (or fit_mano again), where a finger is red or floats use fit_mano mode fingers or curl_finger.
3. Rest: nobody moves the object before the grasp or after the release, so it should be still there. Where the
   rest frames jitter or drift (obj_height_cm, ang_speed / speed in plot), hold_object_static at a frame whose
   outline fits well, or smooth_object.
4. Structure: a wrong object orientation (a handle on the wrong side, the object flipped), the object lagging or
   jumping during fast rotation, the hand far from its keypoints, interpenetration. For each problem: inspect it
   closely (zoom, several frames, a virtual view), make the smallest edit that fixes it, and check the before/after
   result. Revert if it did not help; never stack edits on a version you have not checked. Compare with v0 (history
   / plot) when the automatic fit looks worse somewhere.
5. Object on the table and in the hand: at rest the object's lowest point belongs at the table (within ~0.5 cm) with
   its outline and depth still fitting, and while held the object must fit between the thumb and the fingers. The
   same offset in every rest phase, or an object too thick to be gripped as the video shows, is the object's unseen
   side: fit_object_thickness, or scale_object when you can tell by how much; move_along_ray only when the object's
   visible surface is off the depth map too (depth_bias_cm). Afterwards the rest height must be ~0 with coverage,
   spill and depth_res_cm about unchanged, and a grip fit must reach an opposed grip without the keypoint error
   rising; revert a shape change that does not do that. snap_object_to_table moves the object along the table's
   normal, off its outline: use it only where one rest phase is off. A rest height that differs between rest phases
   or drifts within one is a pose error: hold_object_static or smooth_object.
6. Smoothness: per-frame fits (separate_hand, fit_object_silhouette, polish_object) and hand edits can add jitter
   (fit_mano has its own smoothness term; raise smooth_weight if it is still jittery), and any range edit with a
   sharp end can add a jump. After such edits plot tip_accel_mm / hand_accel_mm and look at view_motion over the
   edited range and a few frames beyond; where the hand or fingers jitter or jump, fit_mano with a higher
   smooth_weight or smooth_hand (window ~8-15 frames) across those frames; where the object does, smooth_object or
   interpolate_object.
   Do not smooth through real fast motion (a quick grasp or release): compare with the hand tracker's reference.
7. Before finishing, check the current version phase by phase, with outline overlays, view3d from several sides
   and a plot of the motion: while the object is held the hand must touch it (contact_cm ~0, green, no gap) without
   passing through it (n_cross ~0, no red); hand, fingers and object move smoothly with no "finger jump" flag left
   (view_motion, plot tip_accel_mm over the whole clip); at rest the object sits still on the table; the object's outline
   follows the real one through the whole motion, with no jumps. Fix what fails, then finish. Finish when the
   remaining errors are small compared to the uncertainty of the evidence; do not chase tiny metric changes, and do
   not edit frames that already look right.
Batch your inspections: several frames per inspect call, several independent tool calls per turn."""


def frames_text(frames):
    """[3, 4, 5, 9] -> '3-5, 9'."""
    return ", ".join(f"{r[0]}-{r[-1]}" if len(r) > 1 else str(r[0]) for r in ops.runs_of(sorted(frames)))


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
    tool("fit_object_thickness", "Search the object's extent along the first frame's viewing direction, the side that frame could not see (thinner when the object rests in the table or cannot be gripped, thicker when it floats): each candidate gets the object's poses re-fitted and a grip fit on the held phases, and you get the evidence for each. Applies the most consistent one. Needs phases.", {}),
    tool("scale_object", "Scale the object's mesh (factor < 1 = smaller) along 'unseen' (the first frame's viewing direction: the extent that frame could not see) or 'seen_long' / 'seen_short' (its longer / shorter extent across that view). The side facing the camera stays in place; the object's poses are re-fitted to mask + depth and the hand is re-fitted over the whole clip.",
         {"axis": {"type": "string", "enum": ["unseen", "seen_long", "seen_short"]}, "factor": {"type": "number", "description": "0.3-3 (e.g. 0.5 halves that extent)"}}),
    tool("fit_table", "Re-fit the table plane from the depth around the object in the frames you give: one plane through the table around the object in all of them (choose frames where the table around the object is visible, from every rest phase). Up for edits, views and the export is the table's normal. Use it when the table drawn in the side views is not where the video's table is under the object at rest.",
         {"frames": {"type": "array", "items": INT, "description": "1-8 frames"}}),
    tool("propose_object_poses", "The pose tracker's top-k alternative object poses at a frame, rendered for you to choose from.",
         {"frame": INT, "k": {"type": "integer", "description": "2-9"}}),
    tool("apply_candidate", "Use candidate c (from propose_object_poses) at frame and apply the same change on [start, end].",
         {"frame": INT, "candidate": INT, **RANGE, **FALLOFF}),
    tool("move_hand", "Rigidly move the hand (cm; optional rotation about an axis through its centroid, rot_axis 'none' to skip).",
         {**RANGE, "dx_cm": NUM, "dy_cm": NUM, "dz_cm": NUM, "frame": {"type": "string", "enum": ["world", "camera"]},
          "rot_axis": {"type": "string", "enum": ["none"] + AXES}, "rot_deg": NUM, **FALLOFF}),
    tool("fit_mano", "Gradient-based fit of the MANO parameters of every frame in [start, end] jointly: 2D keypoints, hand mask, depth, hand-pose prior, smoothness over time, no penetration into the object or table, and in held phases a grip (thumb or palm + a finger on the object). The frames just outside the range stay fixed and the fit joins them without a jump. mode: all, fingers (joint rotations only) or rigid (global pose only).",
         {**RANGE, "mode": {"type": "string", "enum": ["all", "fingers", "rigid"]},
          "contact": {"type": "boolean", "description": "grip term in held phases (default true)"},
          "smooth_weight": {"type": "number", "description": "smoothness over time (default 1; 3-10 for a jittery range)"},
          "contact_weight": {"type": "number", "description": "grip term weight (default 1; 3-10 when held frames still float off the object)"},
          "occlusion_weight": {"type": "number", "description": "0 (default) to 3: keep the hand behind the object where the video shows the object and in front of it where it shows the hand"}},
         required=["start", "end", "mode"]),
    tool("separate_hand", "Per frame, push the hand out of the object along the surface normal where the palm passes through it (at most max_cm; keeps the keypoint fit).",
         {**RANGE, "max_cm": NUM}),
    tool("move_hand_to_depth", "Move the hand along the camera ray by its measured depth error (hand vs depth map), per frame, smoothed: one jump to the side of the object the depth puts it on, fading out over 6 frames beyond the range; follow with fit_mano.",
         dict(RANGE)),
    tool("smooth_hand", "Remove jitter from the hand on [start, end]: smooth its MANO parameters (global pose and/or joint rotations) over time (motion slower than `window` frames is kept), then re-check that it stays out of the object.",
         {**RANGE, "window": {"type": "number", "description": "smoothing period in frames (8-15 typical)"},
          "what": {"type": "string", "enum": ["all", "rigid", "fingers"]}}),
    tool("view_motion", "Motion over [start, end] (<= 90 frames) in one fixed 3D view: wrist + fingertip paths (zig-zag = jitter), plus a filmstrip of 8 consecutive frames at the range's most jittery spot.",
         {**RANGE, "azimuth": NUM, "elevation": NUM, "zoom": {"type": "string", "enum": ["action", "contact"]}}),
    tool("curl_finger", "Turn the named fingers' MANO joints about their bending direction (degrees spread over the 3 joints, curl > 0, open < 0) on [start, end].",
         {**RANGE, "fingers": {"type": "array", "items": {"type": "string", "enum": FINGERS + ["all"]}}, "delta_deg": NUM, **FALLOFF}),
    tool("view3d", "3D views of object + hand + table from several sides (red = hand through / inside the object, green = touching): up to 4 frames x 4 azimuths.",
         {"frames": {"type": "array", "items": INT, "description": "1-4 frames"},
          "azimuths": {"type": "array", "items": NUM, "description": "degrees: 0 camera's right, 90 behind, 180 left, 270 camera side (1-4)"},
          "elevation": {"type": "number", "description": "degrees above level (0-90, default view 25)"},
          "zoom": {"type": "string", "enum": ["action", "contact"]}}),
    tool("attach_hand_to_object", "Keep the hand's position fixed relative to the object over [start, end], as at frame ref (the change fades out over 6 frames beyond the range).", {**RANGE, "ref": INT}),
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
        self.client, self.critic_rounds, self.critique = None, 0, None
        self.shape_done, self.shape_table, self.edits_since_base, self.table_note = False, None, 0, None

    def auto_fit(self):
        """v1 = the fits run over the whole clip (object silhouette, a light smoothing of the object's poses, then
        MANO without the grip term, then with the occlusion term where that helps): the agent starts from it."""
        c = self.clip
        st, _, n1 = ops.fit_object_silhouette(c, self.v.state, 0, c.N - 1)
        st, _, n1b = ops.smooth_object(c, st, 0, c.N - 1, 5)       # the tracker's frame-to-frame pose noise
        st, _, n2 = ops.fit_mano(c, st, 0, c.N - 1, contact=False, shape=True)
        st, n3 = ops.fit_occlusion(c, st, 0, c.N - 1, contact=False)
        k = self.v.add(st, f"automatic fit over the whole clip: {n1}; {n1b}; {n2}; {n3}")
        self.save(k, st)
        self.base = k
        return k

    def save(self, k, st):
        np.savez(os.path.join(self.out, "versions", f"v{k:03d}.npz"), obj=st.obj, global_orient=st.go, transl=st.tr,
                 hand_pose=st.hp, betas=self.clip.betas_of(st), shape=self.clip.shape_of(st))

    def rebase(self, direct=False):
        """The verifier's base must have the working version's object shape (one rigid object per clip, and the
        verifier judges the agent's edits, not the shape). After an edit that changed the shape the base becomes the
        automatic fit under the new shape: the working version itself when nothing else was edited since the base
        (direct), else the old base re-shaped, with its object and hand re-fitted (no grip term)."""
        c, st = self.clip, self.v.state
        if np.allclose(c.shape_of(st), c.shape_of(self.v.states[self.base])):
            return ""
        if direct:
            self.base = self.v.cur
        else:
            nb = ops.reshape_object(c, self.v.states[self.base], c.shape_of(st))
            nb = ops.fit_object_silhouette(c, nb, 0, c.N - 1)[0]
            nb = ops.fit_mano(c, nb, 0, c.N - 1, contact=False)[0]
            cur = self.v.cur
            self.base = self.v.add(nb, f"automatic fit (v{self.base}) under the new object shape: object and hand re-fitted")
            self.save(self.base, nb)
            self.v.cur = cur
        self.edits_since_base = 0
        return (f"\nThe object's shape is one per clip, so the verifier now compares your result against v{self.base}: the "
                "automatic fit under this shape.")

    def thickness(self):
        """fit_object_thickness on the working version (the harness runs it once after the first set_phases)."""
        self.shape_done = True
        direct = self.edits_since_base == 0
        st, touched, note, table = ops.fit_object_thickness(self.clip, self.v.state)
        self.shape_table = table
        if not touched:
            return [{"type": "text", "text": note + "\n" + table}]
        out = self.edit(lambda clip, state: (st, touched, note))
        return out + [{"type": "text", "text": table + self.rebase(direct)}]

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
        self.clip.use(self.v.state)
        st, touched, note = fn(self.clip, self.v.state, *a, **kw)
        k = self.v.add(st, note)
        self.save(k, st)
        self.edits_since_base += bool(touched)
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
        lines += self.jump_report(mb, ma, touched)
        show = [touched[int(j)] for j in np.linspace(0, len(touched) - 1, min(3, len(touched)))]
        tiles = []
        for i in dict.fromkeys(show):
            tiles += [R.overlay(self.clip, self.v.states[before], i, label=f"{i} before (v{before})"),
                      R.overlay(self.clip, st, i, label=f"{i} after (v{k})")]
        return [{"type": "text", "text": "\n".join(lines)}, self.img(R.sheet(tiles, ncols=2), f"v{k}")]

    def jump_report(self, mb, ma, touched):
        """Did the edit make the hand or the object jump? The fastest fingertip's and the object's acceleration on the
        touched frames and two beyond each end (a sharp range end shows there), before vs after."""
        lo, hi = max(0, touched[0] - 2), min(self.clip.N - 1, touched[-1] + 2)
        val = lambda rows, i, k: rows[i].get(k) if rows[i].get(k) is not None else np.nan
        fr = list(range(lo, hi + 1))
        tb, ta = (np.array([val(rows, i, "tip_accel_mm") for i in fr]) for rows in (mb, ma))
        out = []
        if np.isfinite(ta).any() and np.isfinite(tb).any():
            thr = jump_threshold(self.clip.tip_ref)
            out.append(f"  fastest fingertip's acceleration, max over frames {lo}-{hi}: {np.nanmax(tb):.1f} -> {np.nanmax(ta):.1f} "
                       f"mm/frame^2 (at frame {fr[int(np.nanargmax(ta))]}; a jump is over {thr:.0f})")
            new = [i for i, b_, a_ in zip(fr, tb, ta) if np.isfinite(a_) and a_ > thr and a_ > 1.5 * np.nan_to_num(b_) + 2]
            if new:
                out.append(f"  WARNING: this edit made the hand JUMP at frames {frames_text(new)} (fingertip acceleration up to "
                           f"{max(a_ for i, a_ in zip(fr, ta) if i in new):.0f} mm/frame^2). Fix it before anything else: revert and redo "
                           "the edit with a falloff or a range that ends where the hand already agrees, or smooth_hand / "
                           "fit_mano across those frames.")
        ob = [i for i in fr if not np.isnan(val(ma, i, "obj_accel_cm")) and not np.isnan(val(mb, i, "obj_accel_cm")) and (
            (val(ma, i, "obj_accel_cm") > 0.6 and val(ma, i, "obj_accel_cm") > 1.5 * val(mb, i, "obj_accel_cm") + 0.2) or
            (val(ma, i, "obj_ang_accel_deg") > 4 and val(ma, i, "obj_ang_accel_deg") > 1.5 * val(mb, i, "obj_ang_accel_deg") + 1))]
        if ob:
            out.append(f"  WARNING: this edit made the OBJECT jump at frames {frames_text(ob)} (acceleration up to "
                       f"{max(val(ma, i, 'obj_accel_cm') for i in ob):.1f} cm/frame^2, {max(val(ma, i, 'obj_ang_accel_deg') for i in ob):.0f} "
                       "deg/frame^2). Fix it before anything else: redo the edit with a falloff, or interpolate_object / "
                       "smooth_object across those frames.")
        return out

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
            if not self.shape_done:                      # the object's unseen extent, as soon as rest / held are known
                out += [{"type": "text", "text": "The harness ran fit_object_thickness (once, after the first set_phases):"}] + self.thickness()
            rows = self.metrics_of(self.v.cur)
            return out + [{"type": "text", "text": summarize(c, self.v.state, rows)}]
        if name == "fit_object_thickness":
            return self.thickness()
        if name == "scale_object":
            if a["axis"] not in ops.SHAPE_AXES:
                raise ValueError("axis must be unseen, seen_long or seen_short")
            direct = self.edits_since_base == 0
            out = self.edit(ops.scale_object, a["axis"], float(min(max(a["factor"], 0.3), 3.0)))
            return out + [{"type": "text", "text": self.rebase(direct)}]
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
        if name == "fit_mano":
            if a["mode"] not in ("all", "fingers", "rigid"):
                raise ValueError("mode must be all, fingers or rigid")
            return self.edit(ops.fit_mano, cl(a["start"]), cl(a["end"]), a["mode"], None if a.get("contact", True) else False,
                             smooth_weight=float(min(max(a.get("smooth_weight", 1.0), 0.1), 30)),
                             contact_weight=float(min(max(a.get("contact_weight", 1.0), 0.0), 30)),
                             occl_weight=float(min(max(a.get("occlusion_weight", 0.0), 0.0), 3.0)))
        if name == "fit_table":
            fr = sorted({cl(i) for i in a["frames"][:8]})
            before = self.metrics_of(self.v.cur)
            note = c.fit_table(st, fr)
            self.m.clear()                               # heights of every version are measured against the table
            self.table_note = note
            rows = self.metrics_of(self.v.cur)
            rest = [r["frame"] for r in rows if phase_of(st, r["frame"]) in ops.REST] or fr
            h = lambda rr: np.median([rr[i]["obj_height_cm"] for i in rest])
            return [{"type": "text", "text": f"{note} The object's lowest point at rest: {h(before):+.1f} -> {h(rows):+.1f} cm from the table."},
                    self.img(R.sheet([R.view(c, st, i, "side") for i in fr[:4]], ncols=2), "table")]
        if name == "separate_hand":
            return self.edit(ops.separate_hand, cl(a["start"]), cl(a["end"]), float(min(max(a["max_cm"], 0.2), 3.0)))
        if name == "move_hand_to_depth":
            return self.edit(ops.move_hand_to_depth, cl(a["start"]), cl(a["end"]))
        if name == "smooth_hand":
            what = a["what"]
            return self.edit(ops.smooth_hand, cl(a["start"]), cl(a["end"]), float(min(max(a["window"], 3), 40)),
                             what in ("all", "fingers"), what in ("all", "rigid"))
        if name == "view_motion":
            s_, e_ = cl(a["start"]), cl(a["end"])
            e_ = min(e_, s_ + 89)
            az, el = float(a["azimuth"]), float(min(max(a["elevation"], 0), 90))
            mo = c.motion(st, range(s_, e_ + 1))
            ha = {i: m["hand_accel_mm"] for i, m in mo.items() if m.get("hand_accel_mm") is not None}
            fa = {i: m["finger_accel_deg"] for i, m in mo.items() if m.get("finger_accel_deg") is not None}
            peak = max(ha, key=ha.get) if ha else (s_ + e_) // 2
            f0 = int(min(max(peak - 4, 0), c.N - 8))
            txt = (f"v{self.v.cur}, frames {s_}-{e_}: hand jitter median {np.median(list(ha.values())) if ha else 0:.1f}, max "
                   f"{max(ha.values()) if ha else 0:.1f} mm/frame^2 at frame {peak} (hand tracker's reference {c.jitter_ref:.1f}); "
                   f"finger jitter median {np.median(list(fa.values())) if fa else 0:.1f}, max {max(fa.values()) if fa else 0:.1f} deg/frame^2; "
                   + (lambda ta: f"fastest fingertip's acceleration max {max(ta.values()):.1f} mm/frame^2 at frame {max(ta, key=ta.get)} "
                      f"(a jump is over {jump_threshold(c.tip_ref):.0f}). " if ta else "")(
                       {i: m["tip_accel_mm"] for i, m in mo.items() if m.get("tip_accel_mm") is not None})
                   + f"Images: paths over the range; filmstrip of frames {f0}-{f0 + 7}.")
            return [{"type": "text", "text": txt}, self.img(R.trace(c, st, s_, e_, az, el), "trace"),
                    self.img(R.filmstrip(c, st, f0, 8, az, el, a["zoom"]), "filmstrip")]
        if name == "curl_finger":
            return self.edit(ops.curl_finger, cl(a["start"]), cl(a["end"]), a["fingers"], float(a["delta_deg"]), int(a["falloff"]))
        if name == "view3d":
            fr = [cl(i) for i in a["frames"][:4]]
            az = [float(x) for x in a["azimuths"][:4]] or [0.0]
            el = float(min(max(a["elevation"], 0), 90))
            ims = [R.view_orbit(c, st, i, z, el, a["zoom"]) for i in fr for z in az]
            rows = self.metrics_of(self.v.cur)
            txt = "; ".join(f"frame {i}: n_cross {rows[i]['n_cross']}" + (f" ({', '.join(rows[i]['cross_fingers'])})" if rows[i]["cross_fingers"] else "")
                            + ", fingertip gaps cm " + ", ".join(f"{f} {g:.1f}" if g < 5 else f"{f} >5" for f, g in zip(FINGERS, rows[i]["tip_gap_cm"])) for i in fr)
            return [{"type": "text", "text": f"v{self.v.cur}: {txt}"}, self.img(R.sheet(ims, ncols=len(az), tile_w=480), "view3d")]
        if name == "attach_hand_to_object":
            return self.edit(ops.attach_hand_to_object, cl(a["start"]), cl(a["end"]), cl(a["ref"]))
        if name == "history":
            lines = []
            for k, note in enumerate(self.v.notes):
                rows = self.m.get(k)
                extra = ""
                if rows:
                    extra = " | " + ", ".join(f"{key} {np.nanmedian([r[key] if r.get(key) is not None else np.nan for r in rows]):.2f}"
                                              for key in ("coverage", "spill", "hand_kp_px", "penetration_cm", "n_cross"))
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
            if self.client is not None and not self.critic_rounds:
                self.critic_rounds = 1
                rev = critic_review(self.client, self.args, c, st, self.out, "agent")
                self.critique = rev
                serious = [p for p in rev["problems"] if p["severity"] in ("high", "medium")]
                if serious:
                    return [{"type": "text", "text": (
                        f"Not finished yet: an independent reviewer checked v{self.v.cur} against the video from four sides "
                        f"(it judges absolute quality, not changes) and found {len(serious)} high/medium problems:\n"
                        + "\n".join(f"- [{p['severity']}] frames {p['start']}-{p['end']}: {p['issue']} (hint: {p['fix_hint']})"
                                     for p in serious)
                        + "\nFix them with your tools and check the result (view3d, view_motion, metrics). If a problem "
                          "cannot be fixed with your tools, say why in your finish summary. Then call finish again.")}]
            self.done = a["summary"]
            return [{"type": "text", "text": "ok"}]
        raise ValueError(f"unknown tool {name}")


def overview(sess):
    c, st, k = sess.clip, sess.v.state, sess.v.cur
    fr = [int(x) for x in np.linspace(0, c.N - 1, 12)]
    side = [int(x) for x in np.linspace(0, c.N - 1, 6)]
    rows = sess.metrics_of(k)
    close = [r["frame"] for r in rows if r["hand_valid"] and r["contact_cm"] < 1.0]
    near = [close[int(j)] for j in np.linspace(0, len(close) - 1, 3)] if len(close) >= 3 else (close or [c.N // 2])
    near = list(dict.fromkeys(near))
    span = (close[0], min(close[-1], close[0] + 89)) if close else (0, min(c.N - 1, 60))
    vid = c.vid
    task = TASKS[vid.split("_")[0]]
    obj = OBJECTS.get(int(vid.split("_")[-1]), "object")
    text = (f"Clip {vid}: a {c.side} hand {task}. Object: {obj}. {c.N} frames at {c.fps:.0f} fps, {c.W}x{c.H}. "
            f"Object size (long/mid/short axes): {', '.join(f'{100 * e:.1f}' for e in c.extent)} cm; reconstructed from the "
            f"first frame, so its extent along that frame's viewing direction ({100 * c.shape_extent[0]:.1f} cm) is a guess, its "
            f"extents across that view ({100 * c.shape_extent[1]:.1f} and {100 * c.shape_extent[2]:.1f} cm) were seen.\n"
            + (f"You start from v{k} ({sess.v.notes[k]}); v0 is the initial reconstruction.\n"
               if k else "You start from v0, the initial reconstruction (no automatic fit; the verifier compares against "
                         "v0): fit_mano over the whole clip early on is usually worth it.\n")
            + f"v{k} metrics:\n" + summarize(c, st, rows) +
            (lambda j0, jk: f"\nHand jitter (median hand_accel_mm): hand tracker's own {c.jitter_ref:.1f}, v{k} {jk:.1f} mm/frame^2; "
             f"finger jitter (median finger_accel_deg) v{k} {j0:.1f} deg/frame^2.")(
                np.nanmedian([r.get("finger_accel_deg") if r.get("finger_accel_deg") is not None else np.nan for r in rows]),
                np.nanmedian([r.get("hand_accel_mm") if r.get("hand_accel_mm") is not None else np.nan for r in rows])) +
            (lambda jf: f"\nFinger jumps (fastest fingertip's acceleration over {jump_threshold(c.tip_ref):.0f} mm/frame^2; hand tracker's "
             f"median {c.tip_ref:.1f}): " + (f"frames {frames_text(jf)}" if jf else "none") + f" in v{k}.")(
                [r["frame"] for r in rows if r.get("tip_accel_mm") is not None and r["tip_accel_mm"] > jump_threshold(c.tip_ref)]) +
            f"\nImages (v{k}): (1) 12 frames, outline overlays; (2) 6 frames, side views; (3) per-frame signals"
            + (" (v0 for comparison)" if k else "") + f"; (4) frames {near} where the hand is at the object, in 3D from four "
            f"sides (red = through / inside the object, green = touching); (5) wrist + fingertip paths over frames "
            f"{span[0]}-{span[1]} in one fixed view (zig-zag = jitter).")
    return [{"type": "text", "text": text},
            sess.img(R.sheet([R.overlay(c, st, i) for i in fr], ncols=4), "overview"),
            sess.img(R.sheet([R.view(c, st, i, "side") for i in side], ncols=3), "overview_side"),
            sess.img(R.plot(c, {f"v{k}": rows, **({"initial (v0)": sess.metrics_of(0)} if k else {})},
                            ["coverage", "hand_kp_px", "contact_cm", "n_cross", "obj_height_cm", "hand_accel_mm", "tip_accel_mm"]),
                     "overview_plot"),
            sess.img(R.sheet([R.view_orbit(c, st, i, z) for i in near for z in (0, 90, 180, 270)], ncols=4, tile_w=440),
                     "overview_3d"),
            sess.img(R.trace(c, st, span[0], span[1], 0), "overview_trace")]


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


def followup(sess, rev, verdicts):
    """First message of another pass: what the reviewer still finds in the merged result of the pass before, and what
    the verifier rejected of it."""
    k = sess.v.cur
    text = (f"Your earlier pass is finished. Its result is v{k}: your last version with the ranges the verifier rejected set "
            "back to the version you started from. An independent reviewer (fresh context, absolute standards, every phase "
            "next to the video and from four sides) still finds these problems in it:\n"
            + "\n".join(f"- [{p['severity']}] frames {p['start']}-{p['end']}: {p['issue']} (hint: {p['fix_hint']})" for p in rev["problems"])
            + ("\nThe verifier rejected these ranges of your earlier pass:\n" + "\n".join(
                f"- frames {run[0]}-{run[1]}: {v['reason']}" for run, v in verdicts if not v["keep_refined"])
               if any(not v["keep_refined"] for _, v in verdicts) else "")
            + f"\nThis pass starts from v{k} and the verifier compares what you change now against v{k}, so make the edits that "
              "clearly fix a listed problem, most severe first, and check each one from several sides. What was rejected "
              "before will be rejected again unless you do it differently: when a grip could only be reached by pulling "
              "the hand off the video, the object's shape or depth is the likely cause (fit_object_thickness, scale_object). "
              "If a problem cannot be fixed with your tools, say why in your finish summary.\n\n")
    ov = overview(sess)
    return [{"type": "text", "text": text + ov[0]["text"]}] + ov[1:]


def agent_loop(sess, client, args, log, first=None, max_turns=None, max_minutes=None):
    msgs = [{"role": "user", "content": first or overview(sess)}]
    max_turns, max_minutes = max_turns or args.max_turns, max_minutes or args.max_minutes
    ctx = {"edits": [{"type": "clear_tool_uses_20250919", "trigger": {"type": "input_tokens", "value": 140000},
                      "keep": {"type": "tool_uses", "value": 8}}]}
    t_start = time.time()
    for turn in range(max_turns):
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
        if turn >= max_turns - 3 or elapsed > max_minutes:
            results.append({"type": "text", "text": "Budget almost used up: wrap up and call finish within the next turn."})
        msgs.append({"role": "user", "content": results})
    return sess.v.cur


def held_spans(phases):
    """Maximal runs of consecutive held phases (grasp ... place) as [start, end]."""
    spans = []
    for p in sorted(phases, key=lambda p: p["start"]):
        if p["label"] in HELD_PHASES:
            if spans and p["start"] <= spans[-1][1] + 1:
                spans[-1][1] = max(spans[-1][1], p["end"])
            else:
                spans.append([p["start"], p["end"]])
    return spans


def chunks(run, phases, max_len=60):
    """Split a changed frame range for the verifier. The span in which the hand holds the object (consecutive held
    phases) is one piece, judged as a whole: a grip is one thing from grasp to place, and pieces of it that are kept
    and rejected in turn cannot be joined. The rest is split at phase boundaries and into pieces of at most max_len
    frames, so parts of a long edit are judged (and can be rejected) separately."""
    a, b = run
    spans = held_spans(phases)
    inside = lambda f: any(s0 < f <= s1 for s0, s1 in spans)          # a boundary within a held span is no cut
    cuts = sorted({p["start"] for p in phases if a < p["start"] <= b and not inside(p["start"])}
                  | {s1 + 1 for s0, s1 in spans if a < s1 + 1 <= b})
    pieces, s = [], a
    for c in cuts + [b + 1]:
        held = any(s0 <= s and c - 1 <= s1 for s0, s1 in spans)
        while c - s > max_len and not held:
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
        dhr = np.degrees((Rotation.from_rotvec(s1.go[i]) * Rotation.from_rotvec(s0.go[i]).inv()).magnitude())
        dfl = np.degrees(np.abs(s1.hp[i] - s0.hp[i]).max())
        if dR > rot_deg or dt > trans_cm or dh > hand_cm or dhr > 3 or dfl > 5:
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


CRITIC_SCHEMA = {"type": "object", "properties": {
    "overall": {"type": "string", "enum": ["good", "acceptable", "bad"]},
    "problems": {"type": "array", "items": {"type": "object", "properties": {
        "start": {"type": "integer"}, "end": {"type": "integer"}, "issue": {"type": "string"},
        "severity": {"type": "string", "enum": ["high", "medium", "low"]}, "fix_hint": {"type": "string"}},
        "required": ["start", "end", "issue", "severity", "fix_hint"], "additionalProperties": False}}},
    "required": ["overall", "problems"], "additionalProperties": False}
HELD_PHASES = ("grasp", "lift", "carry", "rotate", "place")


def critic_review(client, args, clip, state, out, tag):
    """Absolute check (not a comparison): a fresh-context reviewer sees, per phase, the video frame with the
    reconstruction and the 3D scene from four sides, the wrist/fingertip paths of the held span and the flagged
    frames, and lists what is wrong for a robot to replay."""
    phases = state.phases or [{"start": 0, "end": clip.N - 1, "label": "all"}]
    sheets = []
    for p in phases:
        n = p["end"] - p["start"] + 1
        frs = sorted({p["start"] + n // 3, p["start"] + (2 * n) // 3}) if p["label"] in HELD_PHASES and n >= 3 else [p["start"] + n // 2]
        tiles = []
        for i in frs:
            tiles += [R.overlay(clip, state, i, label=f"{i} {p['label']}: video + reconstruction")] + \
                     [R.view_orbit(clip, state, i, z) for z in (0, 90, 180, 270)]
        sheets.append(R.sheet(tiles, ncols=5, tile_w=340))
    held = [p for p in phases if p["label"] in HELD_PHASES]
    if held:
        a0 = held[0]["start"]
        sheets.append(R.trace(clip, state, a0, min(held[-1]["end"], a0 + 89), 0))
    rows = clip.metrics(state)
    for k, im in enumerate(sheets):
        cv2.imwrite(os.path.join(out, "images", f"critic_{tag}_{k}.jpg"), im)
    prompt = (f"You are reviewing a 4D reconstruction of a {clip.side} hand manipulating a rigid object "
              f"({OBJECTS.get(int(clip.vid.split('_')[-1]), 'object')}; the person {TASKS[clip.vid.split('_')[0]]}). "
              "It will be retargeted to a robot hand and simulated, so it must be physically right, not just line up with "
              "the video. One image per motion phase: each row is one frame, first the video frame with the reconstruction "
              "drawn on it (orange = object, blue = hand), then the 3D scene from the camera's right, from behind, from the "
              "left and from the camera's side (hand blue, red where it passes through the object or is inside it, green "
              "where it touches it). The last image (if any) shows the wrist and fingertip paths over the held span (smooth "
              "curves = good, zig-zags = jitter). Judge by absolute standards, not relative to anything: while the object is "
              "lifted, carried or rotated it must be gripped as the video shows: thumb or palm on one side and fingers on the "
              "other, or fingers hooked through a handle (a mug held by its handle needs no thumb), or fingers wrapped around "
              "it; one fingertip resting on it is not a grip; no part of the hand may be inside the object; the object's pose must match the "
              "video; at rest it must sit on the table; the motion must be smooth. List every problem you can see, with "
              "its frame range, severity (high: a robot could not reproduce the manipulation from this, e.g. no grip while "
              "lifted, fingers through the object, wrong object orientation; medium: clearly wrong but local; low: "
              "cosmetic) and a fix hint. Report only what the images or numbers show.\n\nPer-frame numbers and flags:\n"
              + summarize(clip, state, rows))
    r = client.messages.create(model=args.model, max_tokens=8000,
                               output_config={"effort": "high", "format": {"type": "json_schema", "schema": CRITIC_SCHEMA}},
                               messages=[{"role": "user", "content": [R.encode(im) for im in sheets[:9]] + [{"type": "text", "text": prompt}]}])
    rev = json.loads(next(b.text for b in r.content if b.type == "text"))
    json.dump(rev, open(os.path.join(out, f"critic_{tag}.json"), "w"), indent=1)
    return rev


def critic_text(rev):
    return f"overall {rev['overall']}; " + "; ".join(f"[{p['severity']}] {p['start']}-{p['end']}: {p['issue']} (hint: {p['fix_hint']})"
                                                  for p in rev["problems"]) if rev["problems"] else f"overall {rev['overall']}; no problems"


VERIFY_SIGNALS = ["obj_height_cm", "ang_speed_deg_per_frame", "contact_cm", "n_cross", "hand_accel_mm", "tip_accel_mm"]


def verify_run(client, args, clip, s0, s1, run, out, tag, rows0=None, rows1=None):
    """Blind A/B judgement of a base version (s0) vs the refined one (s1) on one frame range. rows0/rows1 (per-frame
    metrics of the whole clip) add a plot of the motion over the range, which still frames cannot show (jitter,
    drift at rest)."""
    a, b = run
    held = all(phase_of(s1, i) in HELD_PHASES for i in (a, b))       # the span in which the hand holds the object
    n_fr = 6 if held and b - a > 40 else 4
    fr = sorted(set(int(x) for x in np.linspace(a, b, min(n_fr, b - a + 1))))
    order = random.Random(hash((clip.vid, a, b))).random() < 0.5
    A, B = (s1, s0) if order else (s0, s1)
    tiles = []
    for i in fr:
        tiles += [R.overlay(clip, A, i, label=f"frame {i}: A"), R.overlay(clip, B, i, label=f"frame {i}: B")]
    views = []                                            # 3D from three sides, A's row then B's row, per frame
    for i in ((fr[len(fr) // 4], fr[len(fr) // 2], fr[(3 * len(fr)) // 4]) if held and len(fr) >= 4 else
              (fr[0], fr[-1]) if len(fr) > 1 else fr):
        for lab, S_ in (("A", A), ("B", B)):
            views += [R.view_orbit(clip, S_, i, z, label=f"frame {i} {R.AZ_NAMES[z]}: {lab}") for z in (0, 90, 180)]
    im1, im2 = R.sheet(tiles, ncols=2), R.sheet(views, ncols=3, tile_w=440)
    lo_, hi_ = max(0, a - 4), min(clip.N - 1, b + 4)             # wrist + fingertip paths: A | B, same fixed view
    im3 = R.sheet([R.trace(clip, A, lo_, hi_, 0, label=f"A: paths over frames {lo_}-{hi_}"),
                   R.trace(clip, B, lo_, hi_, 0, label=f"B: paths over frames {lo_}-{hi_}")], ncols=2, tile_w=560)
    ims = [im1, im2, im3]
    if rows0 is not None and rows1 is not None:
        lo, hi = max(0, a - 8), min(clip.N - 1, b + 8)
        ims.append(R.plot(clip, {"A": rows1 if order else rows0, "B": rows0 if order else rows1}, VERIFY_SIGNALS, lo, hi,
                          [p for p in s1.phases if p["end"] >= lo and p["start"] <= hi]))
    cv2.imwrite(os.path.join(out, "images", f"verify_{tag}_{a}-{b}.jpg"),
                np.vstack([cv2.resize(im, (1280, int(im.shape[0] * 1280 / im.shape[1]))) for im in ims]))
    prompt = (f"Two reconstructions (A and B) of the same video frames {a}-{b}: a hand and a rigid object "
              f"({OBJECTS.get(int(clip.vid.split('_')[-1]), 'object')}). Overlays: orange outline/fill = rendered object, blue = rendered "
              "hand, green = object segmentation mask (can be wrong), white dots = detected hand keypoints, blue dots = hand "
              "model joints. 3D views: the scene from three sides (table in gray), the hand blue, red where it passes "
              "through the object's surface or is inside it, green where it touches it. Which reconstruction matches the "
              "video better and is "
              "more physically plausible (object on the table when not held, hand touching the object when holding it, no "
              "interpenetration, smooth motion of the object, hand and fingers)? Judge from the images; answer tie if they are equally good or the difference is negligible.")
    prompt += (" The paths image shows, for A and B in the same fixed 3D view, the wrist and fingertips over the frames, "
               "one dot per frame: real motion draws smooth curves, a jittery reconstruction draws zig-zags.")
    if len(ims) > 3:
        prompt += (f" The plot shows A's and B's motion over frames {max(0, a - 8)}-{min(clip.N - 1, b + 8)}: the object's lowest "
                   "point above the table, its rotation speed per frame, the hand's distance to the object "
                   "(0 = touching), the hand's triangles through the object, and the frame-to-frame acceleration of the hand "
                   "(mean over its surface) and of its fastest fingertip (jitter; a spike on one to three frames is the hand "
                   "or a finger jumping). A real object does not jitter or drift while it rests on the table and moves "
                   "smoothly while held, and real hands and fingers move smoothly; spikes there are reconstruction errors.")
    if held and rows0 is not None and rows1 is not None:
        def numbers(rows, st):
            sel = [r for r in rows if a <= r["frame"] <= b]
            lifted = [r for r in sel if phase_of(st, r["frame"]) in ("lift", "carry", "rotate")]
            kp = [r["hand_kp_px"] for r in sel if r.get("hand_kp_px") is not None]
            return ((f"thumb or palm against the fingers on {np.mean([bool(r['opposed']) for r in lifted]):.0%} of the lifted frames, "
                     if lifted else "") +
                    f"{np.mean([r['n_cross'] for r in sel]):.1f} hand triangles through the object on average, keypoint "
                    f"error {np.median(kp) if kp else 0:.0f} px")
        prompt += (f" Frames {a}-{b} are the whole span in which the hand holds the object (grasp to place), and what "
                   "decides here is whether the object is really held. While it is off the table it must be gripped as "
                   "the video shows: thumb or palm on one side and fingers on the other with the object between them, or "
                   "fingers hooked through a handle, or wrapped around it. Fingertips resting on one side of an object "
                   "that hangs in the air is not a grip: nothing would hold it, and a robot replaying it drops the "
                   "object. Prefer the reconstruction in which the object is held, even when its hand lies a little "
                   "further from the white keypoints (keypoints of fingers hidden behind the object are guesses). Do not "
                   "prefer one whose hand plainly contradicts the video (a finger or the thumb where the video shows "
                   "none, the hand through the object). Measured over this span, A: "
                   + numbers(rows1 if order else rows0, A) + "; B: " + numbers(rows0 if order else rows1, B)
                   + " (a hook grip through a handle counts as no thumb-against-fingers grip in these numbers).")
    r = client.messages.create(model=args.model, max_tokens=8000, output_config={"effort": "high", "format": {"type": "json_schema", "schema": VERIFY_SCHEMA}},
                               messages=[{"role": "user", "content": [R.encode(im) for im in ims] + [{"type": "text", "text": prompt}]}])
    v = json.loads(next(b.text for b in r.content if b.type == "text"))
    refined = "A" if order else "B"
    v["refined_label"] = refined
    v["keep_refined"] = not (v["winner"] not in ("tie", refined) and v["confidence"] >= 0.6)
    return run, v


def revert_weights(N, runs, blend=6):
    """Per-frame weight of the base version: 1 on the rejected ranges, falling smoothly to 0 over `blend` frames into
    the kept frames. The 0/1 profile is widened by blend/2 and low-passed with a Hann window, so it has no kink:
    the largest of two overlapping fades has one in the middle of a short kept range (the hand jumps there), and a
    range's own fade must not overwrite its neighbour's frames (the hand jumps to the other version and back)."""
    h = max(1, blend // 2)
    r = np.zeros(N + 4 * h)                                # padded by 2h on both sides
    for a, b in runs:
        r[a + h:b + 3 * h + 1] = 1                         # [a, b] widened by h
    k = np.hanning(2 * h + 3)[1:-1]
    return np.clip(np.convolve(r, k / k.sum(), mode="same")[2 * h:2 * h + N], 0, 1)


def merge(clip, s0, s1, runs_to_revert, kept=(), blend=6):
    """The refined result (s1) with the verifier's rejected ranges set back to the base (s0), by one smooth weight
    per frame (revert_weights). Then a check on the result: wherever its hand jumps more than both versions do (a
    seam), it is smoothed; and where a jump is left next to a kept range (`kept`: the verifier's kept ranges), that
    range is rejected as well and the merge redone: the two versions are too far apart there for the hand to move to
    the other one and back within the range (pp_5: 7 cm at the fingertips, 11 frames kept between rejected ranges).
    Returns (state, notes)."""
    revert, kept, notes = sorted(runs_to_revert), sorted(kept), []
    if not revert:
        return s1.copy(), notes
    if not np.allclose(clip.shape_of(s0), clip.shape_of(s1)):       # one rigid object per clip (Session.rebase)
        s0 = ops.reshape_object(clip, s0, clip.shape_of(s1))
    while True:
        w = revert_weights(clip.N, revert, blend)
        s = s1.copy()
        for i in np.nonzero(w > 0)[0]:
            s.obj[i] = se3_interp(s1.obj[i], s0.obj[i], w[i])
            s = ops._blend(s1, s0, i, w[i], into=s)
        s, seams = ops.smooth_seams(clip, s, [s0, s1])
        bad, a, _ = ops.seam_frames(clip, s, [s0, s1])
        jumps = [i for i in bad if a[i] > jump_threshold(clip.tip_ref)]
        near = [r for r in kept if any(r[0] - blend <= i <= r[1] + blend for i in jumps)]
        if not near:
            return s, notes + seams
        r = min(near, key=lambda r: r[1] - r[0])
        notes.append(f"kept range {r[0]}-{r[1]} rejected as well: joining it to its rejected neighbours left the hand jumping "
                     f"(fingertip acceleration {max(a[i] for i in jumps):.0f} mm/frame^2 at frames {frames_text(jumps)})")
        kept.remove(r)
        revert = sorted(revert + [r])


def report_table(cols):
    """Markdown table of the per-frame metrics' medians; cols = [(name, rows), ...]."""
    med = lambda rows, k: np.nanmedian([r[k] if r.get(k) is not None else np.nan for r in rows])
    return (f"| metric (median over frames) | {' | '.join(n for n, _ in cols)} |\n|---|{'---|' * len(cols)}\n"
            + "\n".join(f"| {k} | " + " | ".join(f"{med(m, k):.2f}" for _, m in cols) + " |" for k in REPORT_KEYS))


def merge_text(clip, final, seams):
    """The report's paragraph on the merge: seams smoothed and finger jumps left in the result."""
    thr = jump_threshold(clip.tip_ref)
    jumps = [r["frame"] for r in clip.metrics(final) if r.get("tip_accel_mm") is not None and r["tip_accel_mm"] > thr]
    return ("Merge of the kept and the reverted ranges: " + ("; ".join(seams) if seams else "no seams to smooth")
            + f". Finger jumps left (fastest fingertip over {thr:.0f} mm/frame^2): "
            + (f"frames {frames_text(jumps)}" if jumps else "none") + ".")


def write_scaled_obj(clip, scale, src, dst):
    """The reconstructed OBJ with its vertices (and normals) scaled along the object's shape axes about its centre
    (Clip.scaled); texture coordinates, faces and the material reference are kept as they are."""
    B, c = clip.shape_axes, clip.center_obj
    S, Sn = B @ np.diag(scale) @ B.T, B @ np.diag(1.0 / np.asarray(scale)) @ B.T
    with open(src) as fi, open(dst, "w") as fo:
        for ln in fi:
            if ln.startswith("v "):
                t = ln.split()
                v = c + S @ (np.array(t[1:4], float) - c)
                ln = "v " + " ".join(f"{x:.6f}" for x in v) + (" " + " ".join(t[4:]) if len(t) > 4 else "") + "\n"
            elif ln.startswith("vn "):
                n = Sn @ np.array(ln.split()[1:4], float)
                ln = "vn " + " ".join(f"{x:.6f}" for x in n / max(np.linalg.norm(n), 1e-12)) + "\n"
            fo.write(ln)


def export(clip, state, out):
    """Write the result and an export/ run dir (the fprecon run with refined object poses + hand motion) for the viewer."""
    clip.use(state)
    scale = clip.shape_of(state)
    np.savez(os.path.join(out, "final.npz"), obj=state.obj, global_orient=state.go, transl=state.tr, hand_pose=state.hp,
             betas=clip.betas_of(state), shape=scale)
    json.dump(state.phases, open(os.path.join(out, "phases.json"), "w"), indent=1)
    ex = os.path.join(out, "export")
    os.makedirs(ex, exist_ok=True)
    meshes = ("object_mesh.obj", "object_mesh.ply")
    for f in os.listdir(clip.run):
        if f in ("object_poses.npy", "hand_rigid.npz", "hand_verts.npy", "mano_params.npz", "object_shape.json") + meshes:
            continue
        dst = os.path.join(ex, f)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(clip.run, f), dst)
    # the object's mesh: the reconstructed one (a link), or a copy scaled along its axes (State.shape)
    for f in meshes:
        src, dst = os.path.join(clip.run, f), os.path.join(ex, f)
        if not os.path.exists(src):
            continue
        if os.path.lexists(dst):
            os.remove(dst)                               # a link or a scaled copy from an earlier export
        if np.allclose(scale, 1):
            os.symlink(src, dst)
        elif f.endswith(".obj"):
            write_scaled_obj(clip, scale, src, dst)
        else:
            clip.mesh.export(dst)
    json.dump({"scale": scale.tolist(), "axes": list(SHAPE_AXES), "axes_object_frame": clip.shape_axes.T.tolist(),
               "center": clip.center_obj.tolist(), "extent_cm_reconstructed": (100 * clip.shape_extent0).tolist(),
               "extent_cm": (100 * clip.shape_extent).tolist(), "extent_cm_long_mid_short": (100 * clip.extent).tolist()},
              open(os.path.join(ex, "object_shape.json"), "w"), indent=1)
    np.save(os.path.join(ex, "object_poses.npy"), state.obj)
    # the hand: MANO parameters per frame (camera frame; mano.py's convention) and the mesh they give
    np.savez(os.path.join(ex, "mano_params.npz"), global_orient=state.go, transl=state.tr, hand_pose=state.hp,
             betas=clip.betas_of(state), valid=clip.hand_valid, side=clip.side, fps=clip.fps)
    np.save(os.path.join(ex, "hand_verts.npy"), clip.hand_verts_batch(state, range(clip.N)).astype(np.float32))
    # The table for simulation: the fitted plane's normal, at the height the object rests at (the mesh's unseen
    # underside and the depth near the object can put the object's lowest point ~1 cm off the measured surface).
    n = clip.table_n
    rest = [i for i in range(clip.N) if phase_of(state, i) in ("approach", "release", "rest")]
    low = [float((clip.sub_v @ state.obj[i, :3, :3].T + state.obj[i, :3, 3]).dot(n).min()) for i in rest]
    meas = [clip.local_table_c(state, i) for i in rest]
    # each stretch of rest frames counts the same (the object rests in two places, before the grasp and after the
    # release; the longer stretch must not decide alone)
    spans = [[rest[0]]] if rest else []
    for i in rest[1:]:
        spans[-1].append(i) if i == spans[-1][-1] + 1 else spans.append([i])
    per = [float(np.median([low[rest.index(i)] for i in sp])) for sp in spans]
    c = float(np.mean(per)) if low else float(n @ clip.table_p)
    json.dump({"normal_up": n.tolist(), "offset": c, "point": (c * n).tolist(),
               "rest_offset_vs_measured_cm": 100 * float(np.median(np.subtract(low, meas))) if low else None,
               "rest_heights_cm": [100 * (h - c) for h in per],
               "source": "object's lowest point at rest (mean over the rest stretches)" if low else "fitted table plane"},
              open(os.path.join(out, "table.json"), "w"), indent=1)
    if os.path.lexists(os.path.join(ex, "table.json")):
        os.remove(os.path.join(ex, "table.json"))
    os.symlink(os.path.join(out, "table.json"), os.path.join(ex, "table.json"))
    # the fitted plane itself (table.json is that plane moved to the object's resting height), for Clip(table=...)
    json.dump({"normal": clip.table_n.tolist(), "point": clip.table_p.tolist(), "info": clip.table_info},
              open(os.path.join(out, "table_fit.json"), "w"), indent=1)


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
    ap.add_argument("--passes", type=int, default=2,
                    help="agent passes: after each, the reviewer checks the merged result; another pass works on what it finds")
    ap.add_argument("--max-turns2", type=int, default=25, help="turns of a later pass")
    ap.add_argument("--max-minutes2", type=float, default=20, help="minutes of a later pass")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--no-critic", action="store_true", help="skip the absolute reviewer (before finish and on the result)")
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
    if not args.no_critic:
        sess.client = client
    log(f"loaded {clip.vid} ({clip.N} frames) in {time.time() - t0:.0f}s; model {args.model}, effort {args.effort}")
    if args.start == "auto":
        t1 = time.time()
        sess.auto_fit()
        log(f"automatic fit (v{sess.base}) in {time.time() - t1:.0f}s: {sess.v.notes[sess.base]}")
    sections, summaries, n_rev, n_runs = [], [], 0, 0
    for n_pass in range(max(1, args.passes)):
        if n_pass == 0:
            final_k = agent_loop(sess, client, args, log)
        else:                                         # on what the reviewer still finds in the merged result
            final_k = agent_loop(sess, client, args, log, first=followup(sess, final_rev, verdicts),
                                 max_turns=args.max_turns2, max_minutes=args.max_minutes2)
        summaries.append(sess.done or "(none)")
        s0, s1 = sess.v.states[sess.base], sess.v.states[final_k]       # verify against the version the pass started from
        runs = [piece for run in changed_runs(clip, s0, s1) for piece in chunks(run, s1.phases)]
        log(f"agent finished at v{final_k} after {(time.time() - t0) / 60:.1f} min; changed ranges vs v{sess.base} {runs}")
        verdicts = []
        if runs and not args.no_verify:
            r0, r1 = sess.metrics_of(sess.base), sess.metrics_of(final_k)
            with ThreadPoolExecutor(4) as ex:
                verdicts = list(ex.map(lambda rn: verify_run(client, args, clip, s0, s1, rn, args.out, f"pass{n_pass + 1}", r0, r1), runs))
            for run, v in verdicts:
                log(f"verify {run}: winner {v['winner']} (refined = {v['refined_label']}), confidence {v['confidence']:.2f}, keep {v['keep_refined']}: {v['reason'][:200]}")
        revert = [run for run, v in verdicts if not v["keep_refined"]]
        n_rev, n_runs = n_rev + len(revert), n_runs + len(runs)
        final, seams = merge(clip, s0, s1, revert, [run for run, v in verdicts if v["keep_refined"]])
        final.phases = s1.phases
        merged = merge_text(clip, final, seams)
        log(merged)
        export(clip, final, args.out)
        final_rev = None
        if not args.no_critic:                        # what is still wrong with the result
            final_rev = critic_review(client, args, clip, final, args.out, "final")
            log(f"reviewer on the {'final' if n_pass else 'first'} result: {critic_text(final_rev)[:600]}")
        sections.append(
            f"Verifier (v{sess.base} vs refined, blind A/B):\n" +
            "\n".join(f"- frames {run[0]}-{run[1]}: {'kept' if v['keep_refined'] else 'REVERTED'} (winner {v['winner']}, refined = {v['refined_label']}, "
                      f"confidence {v['confidence']:.2f}): {v['reason']}" for run, v in verdicts)
            + "\n\n" + merged
            + (f"\n\nIndependent reviewer, before the agent could finish: {critic_text(sess.critique)}" if sess.critique else ""))
        serious = [q for q in final_rev["problems"] if q["severity"] in ("high", "medium")] if final_rev else []
        if n_pass + 1 >= args.passes or not serious:
            break
        # another pass: it starts from the merged result and is verified against it
        json.dump(final_rev, open(os.path.join(args.out, f"critic_pass{n_pass + 1}.json"), "w"), indent=1)
        sections[-1] += f"\n\nReviewer on this pass's result (the next pass works on it): {critic_text(final_rev)}"
        k = sess.v.add(final, f"result of pass {n_pass + 1}: v{final_k} with the ranges the verifier rejected set back to v{sess.base}")
        sess.save(k, final)
        sess.base, sess.done, sess.critic_rounds, sess.critique, sess.edits_since_base = k, None, 0, None, 0
        log(f"pass {n_pass + 2} starts from v{k}: the reviewer found {len(serious)} high/medium problems")
    cols = [("initial", sess.metrics_of(0))] + [("automatic fit (v1)", sess.metrics_of(1))] * (args.start == "auto")
    cols.append(("final", clip.metrics(final)))
    table = report_table(cols)
    clip.use(final)
    shape = clip.shape_of(final)
    report = (f"# Refinement of {clip.vid}\n\nModel {args.model}, effort {args.effort}; {sess.v.cur + 1} versions; "
              f"{len(sections)} pass{'es' if len(sections) > 1 else ''}; {(time.time() - t0) / 60:.1f} min.\n\nAgent summary: "
              + "\n\nSecond pass, on the reviewer's findings: ".join(summaries) + "\n\n"
              f"Phases: {', '.join(p['label'] + ' ' + str(p['start']) + '-' + str(p['end']) for p in final.phases)}\n\n"
              "Edits:\n" + "\n".join(f"- v{k}: {n}" for k, n in enumerate(sess.v.notes)) +
              f"\n\nObject shape: x{shape[0]:.2f} of the reconstructed mesh along the first frame's viewing direction (the unseen "
              f"extent), x{shape[1]:.2f} / x{shape[2]:.2f} across it; the object is {' x '.join(f'{100 * e:.1f}' for e in clip.extent)} cm "
              "along its long / mid / short axes.\n" + (sess.shape_table or "No thickness search ran.")
              + f"\n\nTable: {clip.table_info['source']}"
              + (f" ({clip.table_info['points']} points, {clip.table_info['rms_cm']:.2f} cm rms off the plane)" if clip.table_info["rms_cm"] is not None else "")
              + f"; its normal is up, {clip.table_info['tilt_vs_geocalib_deg']:.1f} deg from GeoCalib's gravity estimate."
              + (f" {sess.table_note}" if sess.table_note else "")
              + "\n\n" + "\n\n".join(sections)
              + (f"\n\nIndependent reviewer on the final result (what is still wrong): {critic_text(final_rev)}" if final_rev else "")
              + f"\n\n{table}\n")
    open(os.path.join(args.out, "report.md"), "w").write(report)
    log(f"done in {(time.time() - t0) / 60:.1f} min; reverted {n_rev}/{n_runs} ranges")
    print(report)


if __name__ == "__main__":
    main()
