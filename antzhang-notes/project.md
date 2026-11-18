# Project Goal
We want to train a dexterous hand policy that intakes RGB and learns to pick up, in-hand manipulate, and then set down an object solely through video generation data.

## 1.0 Generated videos
Here's how we are performing the data augmentation:
1. We first film one video which will be our "reference_video"
2. We augment the reference_video by using the text-only video-to-video (V2V) template from PRISM with Minimax H3.
```
<Video 1> In a real-world continuous footage. Preserve the reference video's original background, lighting and camera viewpoint. Replace the <OBJ> with <CLS>, <ACTION>.
```
- `<OBJ>`: the object in the reference_video (e.g. `white case`)
- `<CLS>`: the replacement object, as text (e.g. `a red apple`)
- `<ACTION>`: a short description of what the hand does in the reference_video

#### IMG_5710.MOV: pick up, in-hand rotation, set down

Reference_video: the right hand picks up the gray AirPods case, spins it around in the palm with the
fingers, and sets it back down. Seed 0; about 13 min per clip on 4x H200.

| Object | Prompt | Result |
|---|---|---|
| red apple | `<Video 1> In a real-world continuous footage. Preserve the reference video's original background, lighting and camera viewpoint. Replace the gray AirPods case with a red apple, pick it up, spin it around in the palm with the fingers, and set it back down on the table.` | Apple swap and in-hand rotation look good, but the background (living room, towel) and the camera (zoomed in) changed |
| wooden block | `<Video 1> In a real-world continuous footage. Preserve the reference video's original background, lighting and camera viewpoint. Replace the gray AirPods case with a wooden block, pick it up, spin it around in the palm with the fingers, and set it back down on the table.` | Best in-hand rotation (the block turns through several faces), but the background (black table) and the camera changed |
| lemon | `<Video 1> In a real-world continuous footage. Preserve the reference video's original background, lighting and camera viewpoint. Replace the gray AirPods case with a lemon, pick it up, spin it around in the palm with the fingers, and set it back down on the table.` | Lemon rotation looks natural, but the background (kitchen) changed and the AirPods case stays in the scene as a second, enlarged object |

Videos: `outputs/minimax_h3/IMG_5710/prism/<object>/seed0.mp4`, plus `seed0_compare.mp4` (source | generated).

## 1.1 Begin HOI reconstruction on generated videos
**Decision: we use Do-as-I-Do** (`third_party/do-as-i-do/reconstruction`). We also considered HOLD; see the
comparison below for why we didn't pick it.

We parametrize the hands with MANO and then visualize with viser the hand and object mesh. We reconstruct the object mesh with SAM3D because of its robustness and the fact that it's state of the art.

### Goal / interface
For each generated video that passes 1.0, produce a 4D hand-object trajectory that 1.2 and 1.3 can consume:

- **In:** generated video (H3: 1344x768, 24 fps, 4-15 s) + `<CLS>` text from its prompt. `<CLS>` doubles as the object text prompt for segmentation, so no clicking is needed.
- **Out (hand):** per-frame MANO (pose, shape, global wrist pose). Right hand only for now.
- **Out (object):** one rigid mesh + per-frame 6-DoF pose, in the same camera/world frame as the hand.
- **Out (scene):** camera intrinsics, gravity direction, metric scale. 1.3 needs the table plane and gravity.
- **Out (QC):** per-clip validity metrics (see below), written back next to the 1.0 manifest.

### Option comparison (decided: Do-as-I-Do)

| | Do-as-I-Do (`third_party/do-as-i-do/reconstruction`) | HOLD (`zc-alexfan/hold`, CVPR'24) |
|---|---|---|
| Object geometry | SAM3D mesh from one reference frame (template, then tracked) | template-free: neural SDF optimized jointly with the hand |
| Object pose | TAPIR velocity tracking + guided-diffusion pose tracking, then translation/scale optimization | HLoc SfM on object-only images, then hand-object alignment fitting |
| Hand | HaWoR (MANO, world frame) | METRO (right hand only) or HaMeR, then MANO registration |
| Metric scale | MoGe pointmaps + GeoCalib gravity | SfM is up to scale; scale comes from fitting to the hand |
| Cost per clip | feed-forward stages plus optimization (TODO: measure) | per-video optimization (200 epochs by default), likely much slower |
| Segmentation | SAM3: click GUI (needs an X display) or text prompt | SAM-Track click GUI |
| Viewer | viser (`scripts/visualize_3d.py`) | aitviewer-based |
| Downstream | output goes straight into DAID `retargeting/` (MuJoCo, Sharpa hand) | needs a converter to the retargeting input format |
| Weak spots | single-frame mesh can be wrong on the hidden back side | SfM needs texture and many views: risky on plain objects (white case, lemon); in-hand rotation helps |
| License | HaWoR CC BY-NC-ND, SAM License (research use OK) | MIT |

Why Do-as-I-Do: it's faster per clip, uses SAM3D for the object mesh, and plugs straight into the retargeting
in 1.3, which matters at our target of 50+ clips per person. HOLD avoids an object template but needs a long
optimization for every video, and its SfM step is risky on plain objects (white case, lemon).
Known risk we accept: SAM3D builds the mesh from a single frame, so the hidden back side can be wrong. Check it in
the viser views of the in-hand rotation clips.

### Things specific to generated videos
- **Reference frame for SAM3D:** use a frame before the grasp, where the object is on the table and unoccluded
  (around frames 0-20 in IMG_5710 outputs).
- **Camera and scene change per clip** (as seen in the IMG_5710 outputs), so estimate intrinsics, gravity and scale
  per video and never reuse them from the reference_video.
- **Object size is whatever H3 drew**, not the size in the prompt (the case_large run came out ~10x too big). Read
  dimensions from the reconstruction, and flag clips that are wildly off.
- **Leftover source object:** e.g. the lemon run still shows the AirPods case. Segment by `<CLS>` text and reject
  clips with two candidate objects.
- **Temporal consistency:** check that the object keeps one identity and shape for the whole clip.

### Validity checks (feeds "fraction of attempted clips that yield demos")
- reconstruction completes (all stages run, no lost tracking)
- hand-object penetration depth below a threshold (mm, TBD)
- no contact loss while the object is off the table (in-hand phase)
- object reprojection IoU vs. the SAM3 masks, and pose jitter (frame-to-frame)
- record GPU time per clip and add it to the 1.0 cost, giving GPU time per valid demo

### TODO
- [ ] Set up DAID recon: 4 conda envs (sam3, sam3d, hawor, tapnet) under `/data`, weights, HF access to
      `facebook/sam3` + `facebook/sam-3d-objects`, MANO download; `git submodule update --init --recursive`
- [ ] Sanity check on the **real** IMG_5710 first, then on the 3 generated IMG_5710 clips
- [ ] Headless SAM3 with the `<CLS>` text prompt instead of the click GUI
- [ ] Batch script: read the 1.0 `manifest.jsonl` → run recon per `ok` clip → write QC metrics back
- [ ] Recon outputs go to `/data` (per-frame MoGe pointmaps are roughly 12 MB/frame, so a few GB per clip); only
      viser screenshots and QC summaries go in `outputs/`

## 1.2 Physical refinement with VLM contact priors

## 1.3 Retarget to robot morphology

## 1.4 Train depth-based or RGB-based policy
