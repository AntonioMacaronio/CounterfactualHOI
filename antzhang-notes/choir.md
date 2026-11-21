# CHOIR hand-object reconstruction (1.1)

## Pipeline (CHOIR)
1. **Stage 1, initialization:** SAM2 object/hand masks from the first-frame mask, SAM3D object mesh + initial pose,
   Dyn-HaMR MANO hand sequence (HaMeR + ViTPose, camera from VIPE).
2. **Stage 2, grasp-aware depth correction:** GraspFlowMatching predicts a ray-depth offset that fixes where the hand
   sits relative to the object.
3. **Stage 3, contact-aware temporal fitting:** object pose from CoTracker point tracks + PnP, with SAM3D keyframe
   resets and rotation protection; amodal masks (Diffusion-VAS) for the hand-occluded object; then joint
   hand-object optimization with penetration and contact losses and temporal smoothness.

Our own steps on top, ported from the Do-as-I-Do setup:
- **Table:** fit the support plane under the resting object, in the same metric frame as hand + object.
- **QC:** silhouette IoU, rotation flips, hand-object penetration/gaps, object/hand height above the table.
- **Viewer:** viser 4D viewer + offline HTML.

Known risks: very new code (released Oct 2026), 3 conda envs (choir, Dyn-HaMR, VIPE), and the table/scale must be
re-derived from CHOIR's depth (Depth Anything V2 / VIPE) rather than MoGe. Gravity: use the table normal when GeoCalib
disagrees by >10° (GeoCalib was 26-33° off on 2 of 4 IMG_5710 clips).

## Status / TODO
- [x] CHOIR pinned (`third_party/CHOIR`), working copy + all envs on `/data` (`scripts/choir/setup.sh`); all three envs run on the H200
- [x] Assets: CHOIR release weights, public checkpoints, MANO links, BMC arrays
- [x] HMP motion-prior model: uploaded by hand, linked into Dyn-HaMR `_DATA/hmp_model` (CHOIR runs Dyn-HaMR with
  `run_prior=False`, so it is not used by default)
- [x] Run on the 10 real videos (`pp/prp_anthony_real_{1..5}`, `scripts/choir/run_choir.sh VID GPU`). Seed masks
  from SAM3 text prompts (`first_mask.sh`; the spatula needed point prompts). Roughly 25-60 min per clip on one H200
  with 2-3 clips per GPU; Dyn-HaMR (HaMeR + VIPE, CPU-heavy) and stage 3 take ~10-30 min each.
  Fixes needed on top of upstream (all in `setup.sh` / `run_choir.sh`): EGL instead of OSMesa for Dyn-HaMR,
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` for its checkpoints, missing deps (OpenEXR, loguru, bundled human_body_prior,
  pytorch-lightning, rtree), `runpy alter_sys` + an undefined global `args` in stage 3's `mp.spawn` workers,
  per-GPU GraspFlowMatching torchrun port, `--hand_side` for clips where Dyn-HaMR sees two hands (`pick_hand_side.py`).
- [x] Viewer: `scripts/choir/view_choir.py` (hand + object + frame-0 depth points, GeoCalib gravity-aligned) and
  `build_choir_html.py` -> `outputs/choir_viser/index.html` (pp / prp tabs). GeoCalib is 1-4° from the table plane
  on all of these clips (lots of vertical lines), so no table-normal fallback was needed.
- [x] CHOIR depth drift: the object is right on the table at the start, but CHOIR places it 6-20 cm short of the
  table along the camera ray when it is set down farther away (pp_2/3/4). Its object silhouette IoU is only
  0.3-0.75 even where the hand is off the object. -> replaced by the pipeline below.
- [x] **FoundationPose + Depth Pro 2 pipeline** (`scripts/fprecon/run_fprecon.sh VID GPU`, on top of CHOIR's
  stage 1: frames, SAM2 masks, intrinsics, SAM3D mesh, Dyn-HaMR hands):
  1. Depth Pro 2 (H+, temporal mode) metric video depth with CHOIR's focal (~1 min per clip).
  2. Metric scale from the MANO hand: median over frames of z_hand / z_depth on the visible hand pixels. Depth Pro 2
     is already close to metric: factor 0.95-1.16 over the 10 clips, per-frame spread 2-7%.
  3. FoundationPose (official weights) on the scaled depth, with depth blanked under the hand: SAM3D mesh sized by
     a size search at frame 0 (render vs mask + depth), vertex-colored from frame 0, tracked; occlusion-aware check
     (coverage / spill vs masks) re-registers from the mask (best of the top-10 hypotheses) and re-tracks backwards.
  4. Hand moved per frame onto the depth map (smoothed). Viewer + QC: `verify_track.py`, `overlay.py`,
     `outputs/fprecon_viser/index.html`, `scripts/fprecon/compare_choir.py`.
  Results (first/last-frame mask IoU, CHOIR -> new): 0.40-0.72 -> 0.85-0.94 (pp_1/prp_1 0.6-0.8: SAM2 masks only
  the silver top of that object, a RealSense-style camera); object ends -0.4..+2.3 cm from the table (CHOIR +3..+20
  cm). Verified by eye on 12 frames per clip; in-hand rotations (prp_2 roll on edge, prp_3 flip, prp_4 mug handle
  passing behind) track correctly. Gotcha: SAM2's hand mask can swallow the held object (all of the mug in prp_4),
  so the object mask must win when blanking hand depth.
- [ ] Port table fit and QC to CHOIR outputs
- [ ] A/B vs Do-as-I-Do on the 4 IMG_5710 clips (same QC metrics)
- [ ] Batch over the 10 real + 50 generated videos
- [ ] Export to the 1.2 retargeting format (Sharpa)
