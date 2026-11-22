# Project Goal
We want to train a dexterous hand policy that intakes RGB and learns to pick up, in-hand manipulate, and then set down an object solely through video generation data.

## 1.0 Generated videos
We film a few real reference videos per task, then use MiniMax H3 (video-to-video) to make counterfactual versions of each one. Two kinds of edit, following Alvin's plan:

- **Type A (geometry):** same object, with one geometric change (shorter, narrower, larger). This exists to compare directly against sim augmentation, which changes the same properties inside the simulator. [`prompt_prp_type_a.md`](prompt_prp_type_a.md)
- **Type B (object swap):** replace the object with a different rigid object from a fixed list of 10. This is the "as diverse as possible" condition. [`prompt_prp_type_b.md`](prompt_prp_type_b.md)

My clips (filmed 11/19):

| i | Object | pp | prp |
|---|---|---|---|
| 1 | small metal case | `pp_anthony_real_1` | `prp_anthony_real_1` |
| 2 | blue tape roll | `pp_anthony_real_2` | `prp_anthony_real_2` |
| 3 | black spatula | `pp_anthony_real_3` | `prp_anthony_real_3` |
| 4 | gray mug | `pp_anthony_real_4` | `prp_anthony_real_4` |
| 5 | dish soap bottle | `pp_anthony_real_5` | `prp_anthony_real_5` |

Each real video gets 5 Type A and 5 Type B edits at seed 0, so 10 * 10 = 100 counterfactual videos (generated 11/21).
Type A uses A01-A03 from the group doc plus two edits of our own: A04 taller (height 120%) and A05 smaller overall
(all dimensions 80%). Type B splits the 10 objects into interleaved halves, {B01,B03,B05,B07,B09} and
{B02,B04,B06,B08,B10}, alternating over pp_1..pp_5, prp_1..prp_5. So pp_i and prp_i get opposite halves (each source
object meets all 10 targets), and each target is used 5 times. The group prompts are written for prp. The pp clips use
the same text with the in-hand rotation steps replaced by a carry. Both versions are in `videogen/group_prompts.py`,
and the clip list is `configs/videogen/real_20261119.json`.

Results: all 100 succeeded first try, 1344x768 at 24 fps, 56.5 H200-hours in total (about 5 min per pp clip and
12 min per prp clip on 4 H200s). They ran as 10 SkyPilot managed jobs on ll-smo (`scripts/sky/h3_videogen.yaml`).
Videos, prompts and source|generated compare clips are in
`s3://far-research-internal/antzhan/counterfactualhoi/videogen/real_20261119/<clip>/<type_a|type_b>/<ID>/seed0.mp4`.

## 1.1 Begin HOI reconstruction on real / generated videos
We use **Open-CHOIR** ([hxwork/CHOIR](https://github.com/hxwork/CHOIR), SIGGRAPH Asia 2026 / TOG) for hand + object reconstruction. It replaces Do-as-I-Do, which we used first and now keep as the baseline.

### Goal / interface
For each real video, and each generated video that passes 1.0, produce a 4D hand-object trajectory that 1.2 can consume:

- **In:** real or generated video (H3: 1344x768, 24 fps, 4-15 s) + `<CLS>` text from its prompt. `<CLS>` doubles as the object text prompt for segmentation (SAM3), which gives CHOIR its first-frame object mask, so no clicking is needed.
- **Out (hand):** per-frame MANO (pose, shape, global wrist pose). Right hand only for now.
- **Out (object):** one rigid mesh + per-frame 6-DoF pose, in the same camera/world frame as the hand.
- **Out (scene):** camera intrinsics, gravity direction, metric scale. 1.2 needs the table plane (all videos are static camera, tabletop manipulation) and gravity.

### Pipeline, status
Old pipeline in [`choir.md`](choir.md).

How it works (one command per clip: scripts/fprecon/run_fprecon.sh VID GPU, about 5–8 min each):
0. Masks and mesh: SAM3 (image model, text prompts such as "mug,cup") segments the object and hand on every frame,
   replacing CHOIR's SAM2 tracks (those drifted: a "hand" on the whole table, a hand mask that swallowed the mug, an
   object mask missing the camera's black panel). SAM3D then builds a textured mesh from frame 0 (CHOIR's copy of
   SAM3D strips the texture and 95% of the triangles, which is why the earlier viewers looked bad).
1. Depth: Depth Pro 2 gives metric depth for every frame, using CHOIR's camera focal length.
2. Scale from the hand: I compare the MANO hand's depth with the depth map on the visible hand pixels. Depth Pro 2 was already close to metric: the correction factor is 0.95–1.16 across clips.
3. Object: FoundationPose tracks the SAM3D mesh on that depth. It sizes the mesh by trying several sizes on the first frame, and re‑registers from the mask when a frame‑by‑frame check says the fit has slipped.
4. Hand: each frame's hand is moved onto the depth map, so hand, object and table all share one metric depth.

Results on the 10 real videos (first/last-frame object mask IoU, both methods scored on the SAM3 masks): 0.80–0.95
vs CHOIR's 0.31–0.74; the object ends 0.4–2.5 cm from the table (CHOIR 3–20 cm); 0–1% of frames flagged by the
per-frame check. Project website: `scripts/site/build_site.py` -> `outputs/site` (repo `CounterfactualHOI-site`).


## 1.2 Agentic Refinement of Reconstructions
Why do we need an agent? These are the failures we've seen, and gradient descent handles most of them badly:
1. **Discrete ambiguities:** the mug tracked with its handle on the wrong side. A gradient can't cross that gap; the agent can look and choose.
2. **Lag in fast motion:** the spatula flip (frames 51–56), where the tracker falls behind and then recovers.
3. **Bad evidence:** SAM2/SAM3 masks missing the camera's black panel or the whole spatula. An optimizer fits wrong masks faithfully; an agent can see that the mask is wrong and down‑weight it.
4. **Physical and semantic constraints:** an object at rest should sit on the table, a grasped object should move rigidly with the hand, and fingers shouldn't pass through the object. These come from understanding the video's phases (approach → grasp → carry/rotate → release), which the agent can label.

The parameters the agent edits:
- Phase labels: frame ranges for approach, grasp, carry, rotate, release and rest. These drive the constraints.
- Object: a few keyframe poses with smooth interpolation between them, plus one global scale.
- Hand: per‑phase position offsets, plus a "lock to object" mode during grasps (the hand's pose stays fixed relative to the object).
- Table: height and tilt.


| Type | Tool | Returns |
|---|---|---|
| look | `inspect(frames, mode=outline\|blend)` | up to 9 frames, cropped to the hand and object: rendered object (orange), hand (blue), SAM3 mask (green), 2D keypoints vs MANO joints, object axes |
| look | `inspect(frames, mode=side\|top\|front)` | gravity-aligned virtual views with the table (5 cm grid): heights, contact |
| look | **`view3d(frames, azimuths, elevation, zoom=action\|contact)`** (new) | up to 4 frames × 4 sides (0 = camera's right, 90 = from behind, 180 = left, 270 = camera side), or a close-up of where hand and object meet; hand red where it passes through / is inside the object, green where it touches |
| look | `zoom(frame, box)` | full-resolution crop of one overlay |
| look | `plot(signals, range)` | per-frame curves, current vs initial: fit, keypoint error, contact, penetration, **triangles through the object** (new), height, speed |
| measure | `metrics(range, window)` | windowed medians + flagged frame runs (incl. "hand through the object's surface (fingers)") |
| phases | `set_phases(phases)` | approach / grasp / lift / carry / rotate / place / release / rest; drives the contact terms |
| edit object | `rotate_object`, `translate_object`, `move_along_ray`, `interpolate_object`, `hold_object_static`, `snap_object_to_table`, `smooth_object` | a new version with before/after numbers and overlays |
| search object | `fit_object_silhouette(range)` | per-frame 6-DoF fit to the mask outside the hand + depth (gradient-free, trust region) |
| search object | `polish_object(range)` | FoundationPose's refiner per frame, kept only where the fit improves |
| search object | `propose_object_poses(frame, k)`, `apply_candidate` | FoundationPose's top-k registration hypotheses, rendered to choose from |
| edit hand | `move_hand`, `attach_hand_to_object` | rigid hand edits |
| edit hand | **`curl_finger(range, fingers, delta_deg)`** (new) | flex / open fingers on top of Dyn-HaMR's articulation (MANO skinning) |
| search hand | `polish_hand(range)` | rigid hand fit to 2D keypoints (MANO joint regressor) + mask + depth, no penetration, contact while held |
| search hand | **`fit_fingers(range, fingers)`** (new) | per-finger flexion search: keypoints, no triangles through the object's surface (FCL mesh-mesh), contact of the last segment while held |
| search hand | **`separate_hand(range, max_cm)`** (new) | pushes the hand out along the surface normal where the palm passes through the object (keeps the keypoint fit) |
| control | `history`, `revert(version)`, `finish(summary)` | |
| verify (no agent) | blind A/B judge per changed range | overlays + 3D views from three sides + motion plot, vs the automatic fit; keeps the fit where the agent's edit loses |

Agentic Framework Loop:
1. Automatic start, no LLM: run the current pipeline, compute metrics, flag suspect frames.
2. Triage (one call): the agent sees a contact sheet, motion plots and the metrics. It returns phase labels and a ranked list of issues with frame ranges, as structured JSON.
3. Fix loop, one issue at a time: form a guess, edit or search, check before/after on the same frames, then commit or revert. The harness enforces the acceptance rule, not the agent: no metric regression outside the edited range.
4. Independent verifier: a separate, fresh‑context call sees the before/after pair in random order and says which matches the video better. This stops the main agent from talking itself into an edit.
5. Finish: a global consistency pass (smoothing, contacts, table), final QC, and the parameter diff with a written rationale.

It plugs in as a new refine step after run_fprecon.sh, and the website gets a fourth section, "agent‑refined".

### Implementation (`scripts/refine`) and results
`bash scripts/refine/run_refine.sh VID [GPU]` (Claude Opus 5.5 on Bedrock, `AnthropicBedrock`, us-west-2), 5–19 min
per clip; outputs in `/data/.../runs/refine_r4/VID` (report.md, transcript, versions/, export/ for the viewer and
retargeting incl. hand_verts.npy = the final hand mesh per frame, table.json = the table at the object's resting
height) and `outputs/refine_viser_r4/`.
- **State:** per-frame object SE(3), rigid hand offset/rotation, per-finger flexion (MANO skinning on top of
  Dyn-HaMR's articulation), phase labels; every edit is a new version.
- **Final design (round 4):** (1) automatic fit, no LLM: `polish_hand` (hand to ViTPose keypoints via the MANO joint
  regressor + SAM3 mask + depth), `fit_object_silhouette` (object pose vs mask outside the hand + depth),
  `fit_fingers` (no hand triangle through the object's surface: FCL mesh-mesh tests), `separate_hand` (palm out along
  the surface normal); (2) the agent starts from that, labels phases, looks at the 3D scene from several sides
  (`view3d`: red = through / inside, green = touching) and fixes what per-frame fits can't see: contact while held
  (`polish_hand` + `fit_fingers` with contact terms, `curl_finger`), a still object at rest, jumps, flips; (3) a blind
  A/B verifier (overlays + 3D views from three sides + a motion plot) compares each changed range with the automatic
  fit and keeps the fit where the agent's edit loses.
- **Metrics (initial → final, medians):** hand keypoint error 19–41 → 9–13 px; object mask IoU (hand = occluder)
  0.86–0.93 → 0.90–0.96; hand triangles through the object while held 143 → 0 (pp_4 mug), 221 → 11 (prp_4 mug),
  170 → 0 (prp_5 bottle), 122 → 0 (prp_2 tape roll), median over clips 73 → 0; penetration ~0 everywhere; flagged
  frames 261 → 43 (prp_1), 119 → 27 (pp_4). Table: `runs/refine_r4/summary.md`.
- **Blind A/B over all frames of the 10 clips** (`scripts/refine/compare.py --tag final4`, chunks ≤ 60 frames at
  phase boundaries, judge sees 3D views): final vs initial 78:0; final vs round 3 (no finger tools, no 3D views)
  64:13 (1 tie); final vs automatic fit 43:7 (28 ties; not independent, the verifier selected with the same judge).
  A human check is still needed.
- **What we learned:** (a) rounds 1–2 started the agent from the initial reconstruction; a non-agentic script of its
  own search tools (`scripts/refine/baseline.py`) matched it (22:19, 34 ties). Agents are bad at hand-tuning numbers
  and good at what needs understanding of the motion; the judge only credits that when it sees the motion. (b) Round
  3 changed the 3D reconstruction by ~0 cm on top of the automatic fit: it looked at 2–3 virtual views per clip, all
  from the camera's right, and could not move fingers; from behind / the left the fingers were inside the mug.
  Measured afterwards, round 3 still had hand triangles through the object while held on 9 of 10 clips (median
  6-219, e.g. 219 prp_4, 98 pp_4, 75 prp_5). With 3D views from any side, finger tools and a mesh-mesh crossing
  measure, round 4 gets 0-11 (64:13 vs round 3 in the blind A/B).
  (c) The rigid hand fit cannot see thin walls (the SDF grid is ~3-4 mm): it now keeps the previous pose where it
  would push the hand through the object (after round 4, which the verifier had to correct on prp_4).
- **Known issues:** on a few frames of the mug clips a fingertip is still inside the cup (only the ring of triangles
  where a finger crosses the wall counts, so the measure undercounts a finger fully through it; next: a per-finger
  parity test along the bones); the object can rest at heights up to ~1 cm apart before and after a move (e.g. pp_2);
  object 1's mesh underside sits ~1 cm into the measured table (handled by table.json).

## 1.3 Retarget to robot morphology
We will retarget to a Sharpa URDF using `do-as-i-do`'s retargeting pipeline.

- [x] `retargeting` env (`scripts/daid/setup.sh env retargeting`); inputs from a Do-as-I-Do run via
  `scripts/retarget/prepare_daid_input.py`, run with `scripts/retarget/run_retarget.sh RAW_DIR TASK` (outputs under
  `/data/.../runs/retarget`; replay with `third_party/do-as-i-do/retargeting/replay_viser.py --run-dir ...`)
- [x] Test on IMG_5710_real (Do-as-I-Do recon): stages 1-4 (dataset, scene, IK) fine; stage 5 (MuJoCo Warp physics)
  took 64 min and terminated early at sim step 1900/2567: the Sharpa hand never lifts the case (pushes it off its
  pedestal at ~4.5 s), final object error 0.13 m. Pick-up + in-palm spin is a hard first test.
- [ ] CHOIR -> retargeting adapter (CHOIR's MANO/object poses into `all_hand_meshes.npz` + layout + mesh), then retry
  on a plain pick-and-place clip

## 1.3 Train a policy

