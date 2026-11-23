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


| Type | Tools |
|---|---|
| look | `inspect` (overlays on the video; gravity-aligned side/top/front views), `view3d` (any side or a contact close-up; hand red where it passes through the object, green where it touches), `view_motion` (wrist + fingertip paths and a filmstrip: shows jitter), `zoom`, `plot`, `metrics` |
| phases | `set_phases` (approach / grasp / lift / carry / rotate / place / release / rest; drive the contact and grip terms) |
| object | edits: `rotate_object`, `translate_object`, `move_along_ray`, `interpolate_object`, `hold_object_static`, `snap_object_to_table`, `smooth_object`; search: `fit_object_silhouette`, `polish_object` (FoundationPose refiner), `propose_object_poses` + `apply_candidate` |
| hand (MANO parameters only: global pose, 15 joint rotations, shape) | edits: `move_hand`, `move_hand_to_depth`, `separate_hand`, `attach_hand_to_object`, `curl_finger` (MANO joint rotations), `smooth_hand`; **`fit_mano`** (new): gradient-based fit of the range's MANO parameters to 2D keypoints, masks and depth, with a hand-pose prior, temporal smoothness, no penetration, and contact / grip in the held phases |
| control | `history`, `revert`, `finish` |
| checks (no agent) | independent reviewer (absolute, every phase vs the video from four sides, 3 votes); blind A/B verifier per changed range vs the automatic fit |

Agentic Framework Loop:
1. Automatic start, no LLM: run the current pipeline, compute metrics, flag suspect frames.
2. Triage (one call): the agent sees a contact sheet, motion plots and the metrics. It returns phase labels and a ranked list of issues with frame ranges, as structured JSON.
3. Fix loop, one issue at a time: form a guess, edit or search, check before/after on the same frames, then commit or revert. The harness enforces the acceptance rule, not the agent: no metric regression outside the edited range.
4. Independent verifier: a separate, fresh‑context call sees the before/after pair in random order and says which matches the video better. This stops the main agent from talking itself into an edit.
5. Finish: a global consistency pass (smoothing, contacts, table), final QC, and the parameter diff with a written rationale.

It plugs in as a new refine step after run_fprecon.sh, and the website gets a fourth section, "agent‑refined".

## 1.3 Retarget to robot morphology and agentic retargeting refinement

After refining, we will add a retargeting agent that closes the loop. First, we retarget withHow

1. **Physics check, per time window (the verifier).** For each window: does the simulated object follow the reconstruction (within 2 cm and 10°)? Did it drop? How much penetration, and did the hand hit the pedestal? Plus images of that window: the simulated robot and object next to the reconstruction and the video frame, from several sides, with contacts marked.
2. **Diagnosis: which layer caused it.**

   | Layer | Example | What the agent changes |
   |---|---|---|
   | reconstruction | no real grip, fingers inside the object, jittery thumb | our existing refinement tools, then re-prepare the window |
   | retargeting targets | the Sharpa hand can't make the human's grasp (different size and joints) | which fingertips it tracks and how strongly, wrist offset, contact guidance (built in, currently off) |
   | physics optimizer | too short a planning horizon for a regrasp, too little weight on carrying the object | horizon, samples, iterations and reward weights, per window |
   | simulated world | object mass or friction, the collision mesh (simplifying a mug into convex pieces can fill in its handle or cavity), table height | the object's physical properties and collision shape, the table |

3. **Fix Re-simulate only the failing window.** Restart the physics from the saved state at the start of the failing window and keep a fix if it helps that window and doesn't ruin later windows.

4. **Repeat** until every window passes or the budget runs out


## 1.4 Train a robot policy with the following embodiment and I/O 
Imagine a Tianji 7 DoF arm with a 22 DoF sharpa connected to the end and a D405 on the wrist like there is in spd.bot. Since spd.bot uses a YAM, we have to mount the wrist camera on the palm side of the hand but it will be near the flange.


| Item | Value |
|---|---|
| Hand | Sharpa Wave, 22 DoF |
| Arm | Tianji 7-DoF force-controlled arm with anthropomorphic cross-axis wrist (see the arm section) |
| Controlled joints | `N = 22 + 7 =` **29** |
| Policy rate | 50 Hz (one step every 20 ms) |
| Low-level control | Built-in joint controllers on hand and arm, servoing faster than the policy; arm targets interpolated between policy steps (see the arm section) |
| Wrist camera | Intel RealSense D405, mounted on the wrist, looking along the hand (as in SPD's per-hand views) |
| Exo camera | Fixed depth camera viewing the workspace (see the exo camera section) |
| Tactile | Not used |

Here is the input design spec of the policy. The outputs are pretty simple, just the 29 joint angle action space.
 
| # | Group | Contents | Frame / encoding | Dim (N = 29) |
|---|---|---|---|---|
| 1 | **Proprioception** | Last 3 frames of measured joint angles `q`, and last 3 frames of the policy's own joint targets `q*` | Each joint normalized to [-1, 1] by its limits | 3 × 2N = **174** |
| 2 | **Object pose (current)** | Position (3) + rotation (6D: first two columns of the rotation matrix) | Robot base frame | **9** |
| 3 | **Future object trajectory** | K = 5 waypoints spaced 100 ms apart (0.1–0.5 s ahead) | Each as a change from the current object pose | 5 × 9 = **45** |
| 4 | **Future hand/arm trajectory** | Reference joint angles at the same 5 time points | Each as a change from current measured `q` | 5 × N = **145** |
| 5 | **Wrist depth (D405)** | Depth image + validity mask → CNN encoder | See the wrist camera section | latent **64** |
| 6 | **Exo depth** | Depth image + validity mask → separate CNN encoder | See the exo camera section | latent **64** |