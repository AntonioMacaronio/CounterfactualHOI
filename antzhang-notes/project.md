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
| look | `contact_sheet(frames)` | grid of video frames with the render, masks and hand drawn on |
| look | `render_overlay(frame)`, `render_view(frame, side\|top)` | one frame up close; a gravity‑aligned 3D view (table contact, hand–object gap) |
| look | `crop(frame, box)` | zoomed region; this is the single biggest lever for vision accuracy |
| look | `plot(signal, frames)` | chart of object velocity, hand–object distance, height above the table |
| measure | `metrics(range)` | coverage, spill, depth residual, table height, contact distance, penetration, velocity spikes |
| edit | `set_phases`, `edit_keyframe`, `flip_about_axis`, `lock_hand_to_object`, `snap_to_table`, `set_scale` | a new version, with before/after metrics |
| search | `reregister(frame, k)` | FoundationPose's top‑k pose guesses as rendered images, for the agent to pick from |
| search | `local_search(params, range, objective)` | gradient‑free fine search (e.g. CMA‑ES) over a subset the agent chooses |
| control | `compare(a, b, frames)`, `commit`, `revert`, `finish(report)` | |

Agentic Framework Loop:
1. Automatic start, no LLM: run the current pipeline, compute metrics, flag suspect frames.
2. Triage (one call): the agent sees a contact sheet, motion plots and the metrics. It returns phase labels and a ranked list of issues with frame ranges, as structured JSON.
3. Fix loop, one issue at a time: form a guess, edit or search, check before/after on the same frames, then commit or revert. The harness enforces the acceptance rule, not the agent: no metric regression outside the edited range.
4. Independent verifier: a separate, fresh‑context call sees the before/after pair in random order and says which matches the video better. This stops the main agent from talking itself into an edit.
5. Finish: a global consistency pass (smoothing, contacts, table), final QC, and the parameter diff with a written rationale.

It plugs in as a new refine step after run_fprecon.sh, and the website gets a fourth section, "agent‑refined".

### Implementation (`scripts/refine`) and results
`bash scripts/refine/run_refine.sh VID [GPU]` (Claude Opus 5.5 on Bedrock, `AnthropicBedrock`, us-west-2), 4–9 min
per clip; outputs in `/data/.../runs/refine_r3/VID` (report.md, transcript, versions/, export/ for the viewer and
retargeting, table.json = the table at the object's resting height) and `outputs/refine_viser_r3/`.
- **State:** per-frame object SE(3) + rigid hand offset/rotation + phase labels; every edit is a new version.
- **Final design (round 3):** (1) automatic fit, no LLM: `polish_hand` (gradient-free fit of the hand to ViTPose
  keypoints via the MANO joint regressor + SAM3 mask + depth, no penetration) and `fit_object_silhouette` (object
  pose vs mask outside the hand + depth) over the whole clip; (2) the agent starts from that, labels phases, and fixes
  what per-frame fits can't see: contact while held (re-fit with a contact term), a still object at rest
  (`hold_object_static` / `smooth_object`), jumps, flips; (3) a blind A/B verifier (overlays + side views + a plot of
  the motion) compares each changed range against the automatic fit and keeps the fit where the agent's edit loses.
- **Metrics (initial → final, medians):** hand keypoint error 19–41 → 8–12 px; object mask IoU (hand = occluder)
  0.86–0.93 → 0.90–0.96; penetration while held e.g. 2.5 → 0.1 cm (pp_5), 0.9 → 0.01 cm (prp_2); flagged frames
  235 → 80 (prp_1), 89 → 1 (pp_4). Table: `runs/refine_r3/summary.md`.
- **Blind A/B over all frames of the 10 clips** (`scripts/refine/compare.py`, chunks ≤ 60 frames at phase
  boundaries): automatic fit vs initial 58:16 (1 tie); final vs initial 66:11 (1 tie); final vs automatic fit 55:2
  (21 ties). The last is not independent (the verifier used the same judge to select). A human check is still needed.
- **What we learned:** rounds 1–2 started the agent from the initial reconstruction; a non-agentic script of its own
  search tools (`scripts/refine/baseline.py`) matched it (22:19, 34 ties). Agents are bad at hand-tuning numbers and
  good at what needs understanding of the motion; the judge only credits that when it sees the motion (plot), not
  just still frames.
- **Known issues:** the object can rest at heights up to ~1 cm apart before and after a move (e.g. pp_2); object 1's
  mesh underside sits ~1 cm into the measured table (handled by table.json); the table estimate near the object needs
  table pixels around it (fixed: widen the ring before falling back to the global plane, which can be 5 cm off).

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

