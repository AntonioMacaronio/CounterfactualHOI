# Embodiment for §1.4: arm + Sharpa hand + D405 mount + camera definitions

What `antzhang-notes/project.md` lists as missing between §1.3 and §1.4 (rows 14 and 16 of "Data this stage has to
hand to 1.4"):

- **Row 16**, the arm's kinematic model and the hand's mount on its flange: a MuJoCo model per arm (7 arm joints +
  the 22 Sharpa joints = 29), checked against the vendor's numbers, and the IK step that turns FDR's floating wrist
  into arm joint angles.
- **Row 14**, the camera definitions: where the wrist D405 sits on the palm, where the exo camera stands, both sets
  of intrinsics, depth range and noise, plus the renderer that turns them into the policy's depth images.
- The **D405 wrist mount** as CAD (STEP for machining, STL in the model).

Two arms are supported; everything below is the same code driven by `configs/embodiment/arms/<arm>.yaml`:

| config | robot | moving | source of the model |
|---|---|---|---|
| `marvin_m6ccs` | Tianji Marvin M6-S-R-CCS-696-V4.0, right arm (cross-axis wrist) | 7 arm joints | `cynthia-you/TJ_FX_ROBOT_CONTRL_SDK`, Apache-2.0 |
| `vega_1p` | Dexmate Vega 1 Pro, right arm, wrist with F/T sensor | 7 arm joints; chassis, torso, head, left arm frozen | `dexmate-ai/dexmate-urdf`, Apache-2.0 |
| `vega_1p_no_ft` | the same without the F/T sensor (flange 30.2 mm shorter) | as above | as above |

Nothing here needs a GPU. Code and configs are in the repo; vendor files, generated models, CAD and results are on
`/data` (`$PROJECT_STORAGE_ROOT/build/...`, `runs/embodiment/...`).

## Run

```sh
bash scripts/flashdex/setup.sh       # once: FlashDexRetarget (its Sharpa hand model is reused here)
bash scripts/embodiment/setup.sh     # venv (venvs/arm), vendor models at pinned commits, mount CAD, the three models
PY=$PROJECT_STORAGE_ROOT/venvs/arm/bin/python; D=$PROJECT_STORAGE_ROOT/runs/flashdex/data_r10_flat
cd scripts/embodiment
$PY ik.py --arm vega_1p --dataset $D [--rollouts RUN/evals/step<N>]     # the IK step -> runs/embodiment/ik/<arm>/<tag>/
$PY ik_pyroki.py --arm vega_1p --mode fingertips --dataset $D [--rollouts ...]   # the same with collision avoidance
                                                                        #   -> runs/embodiment/ik_pyroki/<arm>/<tag>_<mode>/
$PY render_depth.py --dataset $D [--rollouts ...] [--arm vega_1p [--ik-dir DIR]]   # wrist + exo depth -> runs/embodiment/depth/
$PY preview.py --arm marvin_m6ccs                                       # pictures of the model
bash view_embodiment.sh --arm vega_1p [--clips VID ...]                 # interactive 3D viewer of the robot playing the
                                                                        #   clips -> runs/embodiment/viewer/<arm>/.../<clip>.html
# design tools (rerun when clips, table or hardware change; they print what to put into the configs)
$PY place_base.py --arm marvin_m6ccs [--hand-yaw 0 90 180 270]          # where the arm stands
$PY design_wrist_cam.py --dataset $D [--evaluate]                       # wrist camera candidates / the configured pose
$PY design_exo_cam.py --dataset $D                                      # exo camera candidates
$PY d405_mount.py --arm marvin_m6ccs; $PY build_model.py --arm marvin_m6ccs      # after any config change
```

| file | what |
|---|---|
| `common.py` | paths, SE(3), config loading, URDF reader, D-H forward kinematics, FDR datasets and rollouts |
| `build_model.py` | vendor URDF + FDR's Sharpa hand + mount + cameras -> `robot.xml`, `scene.xml`, `frames.json`, `wrist_stub/`; checks |
| `ik.py` | `ArmIK`: palm pose -> 7 arm joints (QP steps, limits, coupled wrist limit); clip conversion and report |
| `ik_pyroki.py` | the IK step with collision avoidance on PyRoki: `palm` and `fingertips` modes, report on the exact meshes |
| `spheres.py` | what `ik_pyroki.py` needs from a model: `ik_arm.urdf`, `ik_full.urdf`, collision spheres and hull corners |
| `place_base.py` | placement search: every candidate scored with the IK on all clips |
| `cameras.py` | camera definitions as objects, ray-cast depth renderer, stereo validity, noise model, policy encoding |
| `scene.py` | MuJoCo scene of a clip (floating hand or an arm, table, object), point-based clearances |
| `design_wrist_cam.py`, `design_exo_cam.py` | camera pose searches |
| `render_depth.py` | depth images of both cameras for every step of a clip |
| `d405_mount.py` | the mount as parametric CAD (CadQuery) |
| `preview.py` | shaded pictures without OpenGL |
| `view_embodiment.py`, `.sh` | self-contained viser HTML per clip: the robot with an IK result, FDR's hand as a ghost, both cameras with their depth images, the Vega's head camera view, checkboxes to hide each frustum (runs in the flashdex env, like `scripts/flashdex/export_viser.sh`; patches viser's client to expose its scene tree, see `client_page()`) |
| `viewer_meshes.py` | the viewer's copies of the visual meshes: thinned with MeshLab's quadric edge collapse to the smallest version within 1 mm (moving parts) / 2 mm (frozen body) of the CAD, split at sharp edges; run by `build_model.py` |
| `build_embodiment_html.py` | the website's tab 6 page around the viewers (`scripts/site/build_site.py --embodiment`) |

Visual meshes keep their vendor colours: a GLB's parts are grouped by material colour and each group is one geom
(the Vega: white shell, black panels, the lights); the Marvin's and Sharpa's STL files carry none and are drawn in one
colour.

Outputs per arm, in `$PROJECT_STORAGE_ROOT/build/embodiment/<arm>_sharpa_r/`: `robot.xml` (root link at the origin),
`scene.xml` (placed in the clip world, with table and exo camera), `frames.json` (joint names, limits, the transforms
below), `wrist_stub/`, `view_*.png`.

## Frames and the chain at the wrist

- **Clip world** (FDR's): z up, table top at z = 0.8 m, the object's first-frame centre over the origin, +y the way the
  source video's camera looked. The human's right arm reaches in from +x / −y.
- **Palm** = the body FDR's floating wrist moves, `R_forearm_rot_y_link` = Sharpa's `right_hand_C_MC`: +z from the wrist
  to the fingers, +x out of the palm, +y to the thumb. The IK puts the arm's palm where FDR's palm is.
- The hand keeps FDR's joint, body, site and actuator names, so FDR's 22 hand joint values go in unchanged.
  Joint order of the models: `arm_j1..arm_j7`, then the 22 hand joints in FDR's order.

```
last arm link ── arm_flange (site: the vendor's flange frame, z out of the arm)
             └── hand_mount (body: the Sharpa flange's frame F = flange + plate_mm along z, turned yaw_deg about z)
                   └── R_forearm_rot_y_link (palm: F + 29.5 mm; 0.5 + 29 mm from Sharpa's with_flange URDF)
                         └── wrist_cam (D405, pose from cameras.yaml)
```

### Marvin M6-CCS

- Kinematics: the vendor URDF (SolidWorks export) equals the manual's D-H table 4-2 and the SDK's `ccs_m6_40.MvKDCfg`
  (174.5 / 287 / 314 mm, 18 mm elbow offsets, 95 mm to the flange). The URDF writes π/2 as 1.5708; the builder snaps
  that. Check (`build_report.json`, 200 random configurations): model vs D-H rows 6e-13 mm; the unsnapped vendor URDF is
  0.006 mm / 0.0016° off the D-H rows.
- Limits used: the SDK's (±170, ±120, ±170, −145..60, ±170, ±60, ±90°; 180°/s). The manual's table and the URDF allow
  ±178° on J1, J3, J5. The two wrist joints share their travel, `|J7| ≤ 110.5° − 1.025 |J6|` (manual fig. 4-4); the IK
  enforces it.
- Flange (manual fig. 4-6): ISO 9409-1-50-4-M6, 6 mm dowel hole, 31.5 mm centring recess, 63 mm boss on a 75 mm
  body. The 8-pin connector and the drag button are on the rim along J7's axis, which is the hand's thumb–little-finger
  direction; the palm side is free.
- **Hand mount**: the Sharpa Wave's default flange is the same ISO pattern, so the hand bolts on without an adapter.
  The dowel pin allows one clocking (`yaw_deg: 90`): J7 (±90°) then flexes / extends the hand and J6 (±60°) is radial /
  ulnar deviation, as on a human wrist. With the camera plate (6 mm) the palm frame is 35.5 mm in front of the arm's
  flange face.

### Vega 1 Pro

- Kinematics: Dexmate's URDF, right arm `R_arm_j1..j7`; every other joint is frozen at `fixed_joints` (torso 0 / 60 /
  30°, which leans the shoulders 30° forward over the table edge; left arm in Dexmate's "folded" pose). Check: model
  vs the URDF's own numbers 1.5e-6 mm (there is no D-H table to compare with).
- Flange: `R_ee`. **The bolt pattern is not from a drawing** (Dexmate's hardware docs need a login). Measured on the
  URDF mesh of the no-sensor wrist: six 4.1 mm holes on a 49 mm circle, 18.4 mm bore. The sensor wrist's mesh shows
  no holes; it is assumed to repeat the pattern. The mesh's end face is 3 mm behind `R_ee`'s origin. Check all
  three on the robot before machining.
- **Hand mount**: the Sharpa ISO flange needs an adapter; the camera plate (8 mm) is that adapter. Four M4 screws
  (four of the six holes) hold it and four M6 threads take the Sharpa flange. That only fits for clockings of
  15° + 30° k; `yaw_deg: 285` follows the most clips (below).

## The IK step (`ik.py`)

Per clip: many starts for the first frame, the most different solutions are each followed along the clip at 10 Hz, the
one whose path stays furthest from limits and table is kept, and the clip is then solved at 60 Hz with every step
starting from the previous solution. A step is a small QP (joint limits, the Marvin's J6/J7 limit, a small pull to
the home posture), iterated to 0.05 mm / 0.01°. Speeds are reported, not clamped. The same call converts joint
*targets* once FDR logs them (`--field`).

**Placement** (`place_base.py`, scored on all ten clips at 10 Hz):

| arm | searched | best | follows |
|---|---|---|---|
| Marvin, upright on a pedestal | 2,016 poses × 4 hand clockings | base at (0.60, −0.40, 0.80), beside the table's near edge | 7 of 10 clips |
| Marvin, as on Tianji's dual-arm body (base axis horizontal) | 2,688 poses × 4 clockings | shoulder at (0.42, −0.25, 1.30), clocking 270° | 8 of 10 clips |
| Vega (chassis + torso posture) | 243 shoulder targets × 3 clockings | chassis at (0.29, −0.55), heading 105°, right shoulder at (0.40, −0.30, 1.20), clocking 285° | 9 of 10 clips |

The configs hold the pedestal placement for the Marvin (clocking 90°, the pin's) and the Vega placement above.

**Result at 60 Hz** (round-10 clips; "solved" = within 1 mm / 0.5°; "arm–table" = lowest point of the arm's moving
links above the table, negative = inside it):

| clip | Marvin: reference | Marvin: FDR rollout | Vega: reference | Vega: FDR rollout |
|---|---|---|---|---|
| pp_1 | 100%, +27 mm | 98%, +35 | 100%, +27 | 100%, +34 |
| pp_2 | 59%, −30 | 63%, −13 | 77%, −8 | 84%, −24 |
| pp_3 | 69%, 0 | 57%, +14 | 100%, −12 | 100%, +13 |
| pp_4 | 100%, −1 | 100%, −9 | 100%, +6 | 100%, +1 |
| pp_5 | 100%, −9 | 100%, +10 | 100%, −3 | 100%, +14 |
| prp_1 | 100%, +62 | 100%, +46 | 100%, +28 | 100%, +32 |
| prp_2 | 65%, −50 | 47%, −34 | 100%, −52 | 100%, −49 |
| prp_3 | 100%, −10 | 100%, −35 | 100%, −8 | 100%, −43 |
| prp_4 | 100%, −74 | 82%, −79 | 100%, −69 | 45%, −105 |
| prp_5 | 99.8%, −6 | 96%, −25 | 100%, +3 | 100%, −34 |
| fully solved and clear of the table | 2 | 2 | 4 | 5 |

(FDR rollout: `cfhoi_r10_flat-1124-060850`, evaluation at 110M steps.) Joint speeds exceed the arms' limits in part of
most clips (up to 2–4× in solved clips), mostly where a joint rides along a limit.

**What this means.** With the palm kept exactly where FDR put it, the floating-wrist data cannot be converted for
either arm (the next section lets the palm move and keeps the fingertips instead). FDR's hand has no arm behind
it, so its wrist follows the human's: 32–85 mm above the table. The hardware behind the palm does not fit there: the
Marvin's last link is 82 × 106 mm across around the hand's axis, the Vega's 83 × 78 mm, and the D405 reaches 75 mm
below the axis. The fix belongs in the retargeting, not in the IK: give FDR's hand that hardware as collision geometry
(`wrist_stub/` holds it in the palm frame: the last arm link's hull, the mount, the D405 box), or retarget with the
arm model itself. The models here are ready for either.

## The IK step with collision avoidance (`ik_pyroki.py`)

The same step on [PyRoki](https://github.com/chungmin99/pyroki) (JAX; CPU, one thread), with the table, the joint
limits, the Marvin's J6/J7 limit and the arm's joint speed limits as hard constraints and self-collision as a cost.
PyRoki works on a URDF, which `spheres.py` exports from the placed MuJoCo model together with the collision model;
its forward kinematics agree with MuJoCo's to 1e-8 mm on every link and fingertip of both arms.

**Collision model** (`spheres.json`):

- robot against robot: a few spheres per link (76 on the Marvin, 171 on the Vega with its body), fitted to each
  link's convex hull. They overshoot the hull by 1-3 cm on the arm (more on the Vega's torso), so the arm just keeps a
  little more distance from the body. Link pairs that cannot move against each other or always overlap are left out.
- robot against the table: not spheres. The clips leave millimetres between the wrist hardware and the table;
  spheres with centimetres of overshoot would invent collisions. A convex shape is above a plane exactly when the
  corners of its hull are, so each moving link keeps its hull corners (the camera's eight exactly, the last arm link's
  within 2.9 mm), which must stay 2 mm above the table.

**Two modes.** In ik.py's results, the part inside the table is rigid with the palm (last arm link, mount, camera)
in about 90% of the colliding frames: 11.7% of all frames on the Marvin and 9.2% on the Vega, against 1.0% and 1.4%
where only links the arm can move are inside. No solver can move the former while the palm stays where FDR put it.

- `palm`: 7 arm joints, the palm must reach FDR's pose, the arm's other links avoid the table and the body.
- `fingertips`: 7 arm + 22 hand joints. The five fingertips must stay where FDR put them (position; orientation
  loosely); the palm may lift and tilt and the fingers re-bend, as far as needed to get the wrist hardware off the
  table. The fingers may not go deeper into the table than FDR's did. FDR's palm pose and finger angles are
  preferences, and ik.py's arm solution is the posture preference, so where nothing collides the result is ik.py's.

A frame is **usable** when the fingertips are within 3 mm of FDR's, arm and wrist hardware are above the table, the
fingers are no deeper in the table and the hand no deeper in the object than FDR's (2 mm slack), and no arm joint
exceeds its speed limit. Round-10 clips, measured on the exact meshes:

| | Marvin: clips usable in every frame | Marvin: frames usable | Vega: clips | Vega: frames |
|---|---|---|---|---|
| `palm`, reference clips | 2 of 10 | 78.4% | 2 of 10 | 85.5% |
| `fingertips`, reference clips | **6 of 10** | 94.1% | **8 of 10** | 99.6% |
| `fingertips`, FDR rollouts (110M steps) | 3 of 10 | 90.2% | 4 of 10 | 95.6% |

Usable frames per clip, `fingertips` mode on the reference clips:

| | pp_1 | pp_2 | pp_3 | pp_4 | pp_5 | prp_1 | prp_2 | prp_3 | prp_4 | prp_5 |
|---|---|---|---|---|---|---|---|---|---|---|
| Marvin | 100% | 71% | 86% | 100% | 100% | 100% | 72% | 100% | 99.3% | 100% |
| Vega | 100% | 100% | 100% | 100% | 100% | 100% | 98.4% | 100% | 98.7% | 100% |

- **The table problem is gone in `fingertips` mode**: arm or wrist hardware is inside the table in 0.0% (Marvin) and
  0.1% (Vega) of the reference frames, against 9.5% and 5.7% in `palm` mode. `prp_4`, where the Marvin's wrist was
  74 mm inside the table, is clear in every frame with the fingertips within 2.5 mm.
- What it costs: the palm leaves FDR's pose by up to 37 mm / 20° (Marvin, `prp_4`) and 48 mm / 27° (Vega, `pp_2`)
  in the worst frame; in a typical frame it does not move (median 0 mm in every fully usable Marvin clip, at most
  2.9 mm on the Vega). The fingers re-bend by up to 59° (Marvin, `prp_4`) and 52° (Vega, `pp_2`) in the clips
  that needed the largest correction, and by 5-32° at most in the other usable clips. The hand is no deeper in
  the object than FDR's (within 0.2 mm) in every reference clip.
- What is left on the Marvin is reach, not collision: in `pp_2`, `pp_3` and `prp_2` the fingertips cannot be
  reached within the joint limits in 14-29% of the frames.
- FDR's rollouts are harder than the reference clips (its wrist jitters and the object slips): fewer clips are usable
  in every frame, but 90-96% of the frames are.
- Speed: about 50-330 ms per frame in `fingertips` mode, 8-85 ms in `palm` mode, on one CPU thread.

**What `fingertips` mode is and is not.** It is a kinematic re-retargeting: the fingertips touch the object where
FDR's did, but the hand's pose around them changed, and the result has not been through physics again. Contacts
made by other parts of the fingers or the palm can differ, most where the fingers re-bend a lot. Before training on
it, replay the joint targets in the simulator and keep the clips whose object still follows. Self-collision is in
the solver as a cost on coarse spheres; it has not been checked on exact geometry (the links' convex hulls overlap
too much to tell).

## Wrist camera (`design_wrist_cam.py`)

504 poses of the D405 housing beside the wrist, scored on every third frame of all ten clips (1,119 frames): what the
camera measures (share of 96 object-surface points and of the fingertips with a valid stereo depth; an object's far
side is never visible, so about 50% is the ceiling) and how close the housing comes to table, fingers and object.
Best pose per side:

| side of the hand | best pose (r, z, tilt / pan) | object measured | object seen at all | fingertips | lowest point vs table | frames within 8 mm of the table | clips affected |
|---|---|---|---|---|---|---|---|
| **palm** | 62 mm, −40 mm, 0° | **27.3%** | 94% of frames | 47% | −51 mm | 16.7% | 7 |
| palm–thumb diagonal | 62 mm, −40 mm, 0° / 35° | 22.5% | 87% | 35% | −34 mm | 4.2% | 3 |
| thumb | 62 mm, −40 mm, 10° / 20° | 7.8% | 33% | 5% | +17 mm | 0% | 0 |
| back of the hand | 62 mm, +6 mm, 10° | 1.6% | 4% | 0% | +16 mm | 0% | 0 |
| little finger (where SPD mounts it) | 62 mm, −40 mm, 10° / 35° | 16.4% | 68% | 44% | −69 mm | 14.2% | 7 |
| palm–little-finger diagonal | 62 mm, −40 mm, 0° / 20° | 24.9% | 87% | 53% | −61 mm | 20.5% | 9 |

(r = distance of the housing's centre from the hand's axis, z = position along the hand, 0 at the palm's wrist end,
negative towards the arm; "within 8 mm" counts frames in which the housing is closer than 8 mm to the table or inside it.)

- **The palm side is the only place that sees the grasp**, and looking straight along the fingers is best: tilting the
  camera towards the palm loses the object (25% → 20% → 10% → 3% at 0 / 10 / 20 / 30°).
- The D405's 7 cm minimum range at 848×480 costs nothing here (a 4 cm minimum would add 0.7 points).
- **No side with a useful view clears the table in these clips.** That is the same problem as the arm's wrist link,
  and has the same fix.

**Chosen** (`cameras.yaml`): housing centre at (54, 0, −18) mm in the palm frame, no tilt, its back flush with the
mount plate, which makes the mount one flat part. On all frames of the reference clips it measures 24.0% of the
object's surface (the object is seen in 87% of the frames in which it is within 25 cm), 50% of the fingertips; it stays
11 mm from the fingers and 21 mm from the object, and is inside the table in 5 clips (worst −37 mm; within 8 mm in
11.7% of frames). On FDR's rollouts: 4 clips (worst −26 mm, 9.6%). The search's best palm pose sits 22 mm further
back beside the arm's flange: 3 points more of the object, but 14 mm deeper into the table and a bracket instead of
a flat plate.

### The mount (`d405_mount.py`)

One flat plate between the arm's flange and the Sharpa flange, with a tab on the palm side; the D405 is screwed to
the tab's hand-side face through its two rear M3 threads. Flat, so it can be cut from sheet (aluminium 6061; SLS PA12
also works). Files: `build/embodiment/mount/<arm>/d405_mount.step` (mm), `.stl`, `.json` (dimensions, hardware,
clearances), `_top/_side/_iso.png`.

| | Marvin | Vega 1 Pro |
|---|---|---|
| plate | 75 mm disc, 6 mm; 53 × 44 mm tab | 64 mm disc, 8 mm; 53 × 44 mm tab |
| mass (aluminium) | 101 g | 113 g |
| to the arm | 4 × M6×18 through Sharpa flange and plate into the arm; 6×16 dowel pin; 31.4 mm centring boss | 4 × M4×9, heads sunk |
| to the hand | (the same four M6) | 4 × M6×10 into the plate's M6 threads; no dowel |
| camera | 2 × M3×6, heads sunk 3 mm | 2 × M3×8, heads sunk 3 mm |
| camera to Sharpa flange / wrist / palm | 4.8 / 12.0 / 8.1 mm | the same |

The D405's USB socket (USB 3 Micro-B with two M2 locking screws) faces the thumb side: use a right-angle plug; two
3.2 mm holes beside it take a cable tie.

## Exo camera (`design_exo_cam.py`)

360 viewpoints around the workspace (azimuth × elevation × distance, looking at the middle of the object paths),
scored on the same frames with a generic forearm behind the hand. The view from the side opposite the arm wins
clearly; elevation between 20° and 35° makes no difference. Best elevation and distance per azimuth:

| stands at (azimuth) | object measured | fingertips measured |
|---|---|---|
| 0–45° (the arm's side, +x) | 35–37% | 22–38% |
| 90° (far side of the table) | 39% | 34% |
| 180° (left of the table) | 43% | 54% |
| **195–240° (left-front)** | **42–43%** | **58–61%** |
| 270° (where the video camera stood) | 39% | 46% |
| 300–345° | 36% | 19–21% |

**Chosen**: a RealSense D435 at (−0.544, −0.276, 1.230): azimuth 210°, 25° elevation, 0.90 m from the look-at point
(0.162, 0.132, 0.85), 0.43 m above the table at its left-front corner, clear of both robots. It measures 42.7% of the
object's surface (worst clip 26.7%) and 60.5% of the fingertips. The policy image is a 530 × 300 window of the
stream, 5× smaller (the smallest window that holds object and fingertips in every frame); the object covers about 160
pixels of it. A D405 cannot be used here (rated to 0.5 m).

## Camera definitions (`configs/embodiment/cameras.yaml`, `cameras.py`)

| | wrist | exo |
|---|---|---|
| sensor, stream | D405, 848×480 at 60 fps | D435, 848×480 at 60 fps |
| pose | palm frame: (0.054, 0.009, −0.0102), view along +z, image right = −y | world: (−0.5443, −0.2758, 1.2304), view (0.785, 0.453, −0.423) |
| intrinsics (nominal) | fx = fy = 430.6, c = (424, 240): 89.1° × 58.3° | fx = fy = 425, c = (424, 240): 89.9° × 58.9° |
| baseline | 18 mm | 50 mm |
| valid depth | 0.07–0.50 m | 0.30–1.50 m |
| policy image | 106 × 60, whole frame | 106 × 60, window (192, 67, 530, 300): 63.6° × 38.8° |
| depth noise (1σ) | 0.1 mm at 10 cm, 0.8 mm at 30 cm, 2.3 mm at 50 cm | 1.9 mm at 0.75 m, 2.7 mm at 0.9 m, 4.8 mm at 1.2 m |

- Poses are optical frames (z view, x right, y down) of the **left imager**, where a RealSense measures depth from.
- **Validity**: a pixel is valid when its depth is in range and *both* imagers see the point. The renderer tests the
  right imager explicitly, which gives the shadows beside near objects and the invalid band on the image's left
  that real D400 cameras have (wide on the wrist camera, where everything is near).
- **Noise** (per camera, parameters in the YAML): disparity noise (white + smooth over the image), a per-episode depth
  scale error (σ 1%), isolated dropouts, dropouts on depth edges and at grazing angles, round holes (more on the
  D405, which has no projector). Depth error follows `z² · σ_disparity / (fx · baseline)`. The numbers sit inside the
  D400 datasheet's bounds; they are a model, not a measurement of our cameras.
- **Policy input**: two channels per camera, depth scaled to [0, 1] over the valid range (0 where invalid) and the mask.
- Intrinsics are nominal (square pixels, the datasheet's vertical field of view). Replace them with each camera's
  factory calibration; randomise them in simulation.

`render_depth.py` renders both images for every step from joint angles, object pose and scene only (rows 1, 3, 11),
on CPU: the 3,350 steps of the ten reference clips take about 6 minutes on 10 cores. Output per clip: `*.npz` (ideal
and noisy depth + masks, camera poses) and a contact sheet.

## Not verified / assumptions

- Which Tianji arm: the M6 CCS V4.0 is assumed (the M3-S-CCS-615 has other link lengths: manual table 4-3).
- Vega: flange bolt pattern and end face (above); F/T wrist assumed (`vega_1p_no_ft` is the other).
- The exo camera model (D435) and the table's extent (1.2 × 1.0 m slab) are choices, not facts about the lab.
- Arm actuator gains in the models are placeholders; the models have no collision geometry for the arm (visual
  meshes only), so they are kinematic models, not yet simulation assets.
- MuJoCo's `mj_geomDistance` returns 0 for some of the detailed vendor meshes; all clearances here are computed from
  surface points instead (`scene.py`).
- `ik_pyroki.py --mode fingertips` changes the hand's pose and finger angles and has not been re-simulated; its
  self-collision cost is unverified on exact geometry (see its section).
