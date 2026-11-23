# FlashDexRetarget on our clips (website tab 5 · Retargeted motions) and §1.4 policies

[FlashDexRetarget](https://github.com/Holiday-Robot/FlashDexRetarget) (FDR, MIT) turns human hand-object clips
into physically plausible robot motions by training one reference-conditioned RL policy (FlashSAC) over all clips in
simulation (Isaac Sim 5.1 / Isaac Lab 2.3 with mjlab managers): the policy drives a floating-wrist Sharpa Wave hand
so the simulated object follows the reconstructed one. Its rollouts are the "retargeted motions".

Everything large lives on /data: FDR's checkout and env (`$PROJECT_STORAGE_ROOT/build/FlashDexRetarget`,
`$PROJECT_STORAGE_ROOT/conda_envs/flashdex`), datasets (`$PROJECT_STORAGE_ROOT/runs/flashdex/<dataset>`), runs
(`$FDR/logs/flashsac/<run>`).

## Pipeline

```sh
bash scripts/flashdex/setup.sh            # FDR + its conda env (Isaac Sim, Isaac Lab, rsl-rl-flashsac), Sharpa USD
bash scripts/flashdex/setup_cfhoi.sh      # link our import source into FDR, write $FDR/scripts/env.sh, MANO pkls
D=$PROJECT_STORAGE_ROOT/runs/flashdex/data_r5
cd $FDR && bash scripts/retarget/import_human_demo.sh source=cfhoi raw=$PROJECT_STORAGE_ROOT dataset=$D sides=right \
    workers=10 [source.refine=refine_r6]
python -I scripts/flashdex/add_table.py $D                     # the table as one support disk per clip
cd $FDR && bash scripts/retarget/kinematic_retargeting.sh dataset=$D robot=sharpa desk.mode=keep
python -I scripts/flashdex/pad_bimanual.py $D                    # -> $D/motion_bimanual.pt (the robot's layout)
bash scripts/flashdex/train.sh $D/motion_bimanual.pt cfhoi_r5 [hydra overrides]  # -> $FDR/logs/flashsac/cfhoi_r5-<stamp>/
bash scripts/flashdex/export_viser.sh --dataset $D --archive <run>/evals/step<N>   # -> outputs/flashdex_viser/
python scripts/flashdex/build_flashdex_html.py                  # -> outputs/flashdex_viser/index.html
```

`prepare_dataset.sh REFINE [OBJECTS_FROM]` runs the dataset steps for one refine round, for both the TrackEnv
dataset and the GenEnv one (`data_<rN>`, `data_<rN>_gen`).

| file | what |
|---|---|
| `cfhoi_source.py` / `.yaml` | FDR import source: one demo per clip, right hand as MANO params + object poses, z-up table world |
| `mano_nochumpy.py` | MANO pkls without chumpy objects (smplx needs them; chumpy does not import on current numpy) |
| `pad_bimanual.py` | right-hand pack (28 joints) -> the bimanual robot's 56, left hand parked |
| `prepare_dataset.sh` | import -> tables -> retarget -> GenEnv clips -> retarget -> pad, for one refine round |
| `add_table.py` | the table as one support disk per clip (instead of FDR's still-segment disk fit) |
| `train.sh`, `train_cfhoi.py` | FDR's train.py + every clip's eval rollout archived (`<run>/evals/step<N>/`) + `metrics.jsonl` |
| `metrics.py` | object position / rotation error, AdaDex progress (5 cm / 20°) and success, wrist / fingertip error |
| `export_viser.py` / `.sh` | rollouts -> self-contained viser HTMLs + metrics JSON |
| `make_gen_clips.py` | GenEnv clips: each clip with a new object start and goal (planar rigid warp) |
| `policy/cfhoi_terms.py`, `policy/train_policy.sh` | §1.4 policy: actor observations, hybrid reward, launcher |
| `policy/reweight_gen.py` | sets a pack's GenEnv share by repeating its TrackEnv clips (FDR draws uniformly per object) |
| `rest_check.py` | MuJoCo drop test of each clip's first / last reference pose on its table disk (no hand) |
| `flat_base.py` | flat-bottom collision meshes (coacd) for objects whose rest pose is unstable; visual mesh unchanged |
| `eval_curves.py` | per-eval scores of a run -> `<run>/curves.json`, `curves.png` |
| `auto_refresh.sh` | background job: keeps every run's curves current and re-exports tab 5 (+50M steps / +10 points) |
| `build_flashdex_html.py` | the tab-5 page: refined reconstruction (left) vs FDR Sharpa rollout (right), pp / prp tabs |

## Decision log

- **Hand references (round 5).** The round-5 refined fingers are vertex edits, not MANO parameters, so they cannot
  drive MANO retargeting. The round-5 dataset (`data_r5`) uses the tracker's MANO hand
  (`scripts/refine/mano.tracker_params_camera`: Dyn-HaMR + fprecon's depth correction) with the round-5 object poses.
  Round 6 exports MANO parameters (`runs/refine_r6/VID/export/mano_params.npz`); the source picks them up
  automatically, and switching is one flag: `source.refine=refine_r6`.
- **World frame.** z = the table normal (`export/table.json`), x = the camera's right projected onto the table, origin
  under the object's first-frame centre, table plane at z = 0.8 m. Clips are right-hand only (FDR supports one
  object per clip only with a single side). FDR resamples to 60 Hz.
- **MANO models.** smplx loads Dyn-HaMR's MANO pkls, which hold chumpy arrays; `mano_nochumpy.py` writes numpy-only
  copies into `$FDR/assets/mano` (same data).
- **Table.** FDR fits the table as up to 4 thin disks where an object stands still. On our clips FoundationPose's
  frame-to-frame jitter breaks the still segments: 6/10 clips lost the disk under the placed object (rejected as
  `obj_offdisk_untouched`) and the other 4 got no disk at all (objects 0.8 m above the floor). `add_table.py` gives
  each clip one disk centred under the object's path (radius 0.3 -> the env's 0.4 m bucket; all object paths fit
  inside), its top at the object's resting height over the first 10 frames (0.4-3.7 cm below the reconstructed
  table plane: the object bottoms sit slightly inside the plane). With `desk.mode=keep` all 10 clips pass FDR's
  filters.
- **Training wrapper.** `train_cfhoi.py` is FDR's `train.py` plus two hooks. Every evaluation (every 20M env steps,
  deterministic, from each clip's first frame) archives every clip's rollout to `<run>/evals/step<N>/`; FDR's own
  `success/` only keeps the first rollout that passes its SR@1.0. Every logged metric also goes to
  `<run>/metrics.jsonl`, because wandb runs offline here. Runs use `eval.stop_when_all_solved=false`, since FDR's
  success (time-mean error within 3 cm / 30°) is looser than AdaDex progress, and a replay buffer of 20M rows
  (35 GB, on the GPU) so each run leaves most of an H200 free.
- **Metrics** (`metrics.py`, on every exported rollout):
  - object position / rotation error per frame against the reconstruction;
  - **progress** (AdaDex): the share of the clip before the object first leaves 5 cm / 20°;
  - **success**: progress = 1, the whole clip within 5 cm / 20°;
  - FDR's success;
  - wrist error (cm) against the kinematic retarget, and fingertip error (cm) against the MANO fingertips.

  The Do-as-I-Do baseline on the page reads `outputs/retarget_viser/VID.json`. Its progress is position-only,
  because that json has no per-frame rotation.

- **Right hand on the bimanual robot.** FDR retargets single-hand clips with `right.xml` (28 joints) but trains
  only on the bimanual robot. That robot's joint ids (MJCF order, which is also the .pt column order) index the
  reference arrays, so a 28-column pack read out of bounds at the first reset (a CUDA device assert).
  `pad_bimanual.py` writes `motion_bimanual.pt` with the left hand parked 2 m above the table and 1.2 m to the
  side. Training uses `per_hand_actor_critic=false`: FDR's per-hand reward split builds left-hand twins of every
  term, which need a left reference. One actor-critic on the right hand's reward is the same objective here.
- **Refine rounds.** The first runs use round 5. The final references come from round 7 (the main session's
  MANO-only refinement with the fixed `fit_mano`; round 6 was rated bad and is skipped). A watcher runs
  `prepare_dataset.sh refine_r7 data_r5` once all 10 `runs/refine_r7/*/report.md` exist. It copies r5's objects,
  which have the same meshes, so coacd is not rerun: the mug meshes alone took 29 min.

- **60M steps** (cfhoi_r7). On the 5 pick-and-place clips the simulated object follows the reconstruction within
  1.0-1.7 cm on average, with 63-97% of frames within 2 cm (Do-as-I-Do: 39-79%). That is FDR's success on all
  five, and position-only progress is 88-100%. AdaDex progress (5 cm / 20°) is still only 2-61%: the object's
  orientation error is 12-22° on average and crosses 20° early, partly from the unstable rest poses. The
  in-hand-rotation clips are not learned yet (position-only progress 19-89%).
  - Mean AdaDex / position-only progress: 16 / 34% at 30M, 24 / 69% at 60M.
  - cfhoi_r7_flat at 30M: 25 / 40% (pp_1 2 → 41%, prp_5 3 → 14%).
- **§1.4 policy** (p14_adadex_r7) at 60M: TrackEnv progress 19% (position-only 33%), GenEnv 19% (32%), final-goal
  success 0/19. It learns much slower than FDR's privileged actor at equal steps.

- **Open option: distil the §1.4 actor** (not implemented). The §1.4 actor sees no velocities, contacts or point
  clouds and learns far slower with RL from scratch than FDR's privileged actor. Once a privileged FDR policy tracks
  well, a teacher-student distillation (DAgger: roll out the student, label with the teacher's actions) into the
  §1.4 observation space is likely the better design: RL then only fine-tunes, for example with the GenEnv goals.

## §1.4 policy (`policy/`)

A first, state-based version of the policy in `antzhang-notes/project.md` §1.4. It runs on FDR's env and FlashSAC
learner (`policy/train_policy.sh`), with `envs/obs@obs=cfhoi_policy envs/rewards@rewards=cfhoi_policy`.

- **Embodiment.** There is no Tianji arm model yet, so the stand-in is FDR's floating 6-DoF wrist under the 22-DoF
  Sharpa Wave: N = 28 instead of 29. The wrist pose the policy commands is what the arm would have to realise, by
  IK or by swapping in an arm model, which is the next step. The rate stays at FDR's 60 Hz (the references are
  60 Hz), not §1.4's 50 Hz.
- **Actor observations** (`policy/cfhoi_terms.py`): §1.4 rows 1-4.
  - the last 3 frames of measured q and of the PD targets q*, each mapped to [-1, 1] by the joint limits;
  - the object pose in the base frame: position + 6D rotation;
  - 5 future object waypoints 0.1-0.5 s ahead, as deltas from the current pose;
  - 5 future reference joint configurations, as deltas from q.

  The wrist D405 and exo depth encoders (rows 5-6) are not in yet. The critic keeps FDR's privileged
  observations (contacts, SDF, point clouds, velocities), plus the actor's object and future terms.
  The future-trajectory Conv1D encoder is off, because the actor only sees §1.4 inputs.
- **Reward** (AdaDex hybrid):
  - one actor trained on two env populations, split by clip;
  - **TrackEnv** (the 10 originals): FDR's dense object tracking (object point-cloud match) plus hand tracking
    (wrist position / rotation, fingertips, joint-space BC);
  - **GenEnv** (`make_gen_clips.py`: 2 warped copies per clip, with a new object start of ±30° / ±5 cm and a new goal
    of ±30° / ±8 cm, as a planar rigid motion that blends over the carry): the hand-tracking terms are off, and a
    sparse `gen_obj_success` pays 1 when the object is within 5 cm / 20° of its goal waypoint;
  - the dense object term stays on in both populations, because FDR's early reset and its object-assist
    curriculum are keyed on it. This is a deviation from AdaDex, which uses the sparse reward only.
  - Both populations' clips are retargeted by FDR's kinematic retargeting, so each has a feasible Sharpa reference.

## Runs (2026-11-23)

| run | data | GPU | notes |
|---|---|---|---|
| `cfhoi_r5-1123-133857` | `data_r5/motion_bimanual.pt` (10 clips) | 1 | FDR defaults; shakedown on round 5; stopped at 17:03 (~85M steps), superseded by the r7 runs |
| `p14_adadex_r7_g25-*` | `data_r7_gen/motion_bimanual_g25.pt` (76 clips: each TrackEnv clip repeated, 25% GenEnv) | 1 | A/B against p14_adadex_r7 (66% GenEnv), everything else equal; started 17:05 |
| `p14_adadex_r5-1123-135358` | `data_r5_gen/motion_bimanual.pt` (10 Track + 19 Gen) | 2 | §1.4 policy shakedown; stopped after 8 min (3.3M steps) once it ran cleanly |
| `p14_adadex_r7-*` | `data_r7_gen/motion_bimanual.pt` (10 Track + 19 Gen) | 2 | §1.4 policy on round 7; eval every 10M steps |
| `cfhoi_r7-1123-135524` | `data_r7/motion_bimanual.pt` (10 clips) | 3 | FDR on the final references (round 7); eval every 10M steps |
| `cfhoi_r7_flat-*` | `data_r7_flat/motion_bimanual.pt` | 0 | as cfhoi_r7, with flat-base collision meshes for pp_1, pp_3, prp_3, prp_5 (started 15:12) |

All runs: `runner.agent.buffer_max_length=20000000` (about 40 GB of GPU memory per run),
`eval.stop_when_all_solved=false` and `num_env_steps=300000000`. Throughput is 3.3 it/s × 2048 envs, about
6.8k env steps/s or 25M env steps/h, so the 300M budget would take 12 h. The page is exported from whichever eval
is latest when it is built.

## Findings

- **Unstable rest poses** (`rest_check.py`: a MuJoCo drop test of each clip's first and last reference pose on its
  table disk, with no hand). Several round-7 rest poses settle by more than 15°:
  - start: pp_1 22.6°, prp_3 20.2°, pp_3 15.5°;
  - end: pp_1 17.0°, prp_3 17.3°, prp_5 20.3°;
  - the rest turn ≤ 7° at the start.

  The simulated object then leaves 20° of the reference within a few frames, before the hand acts. That caps
  AdaDex progress for these clips (pp_1 and prp_3 sit at ~2%). The fix belongs upstream: a flush, stable rest pose
  in the refinement. A constant tilt correction here would also rotate the object in the hand during the grasp.
- **Flat-base variant** (`flat_base.py` -> `data_r7_flat`, run `cfhoi_r7_flat` on GPU 0). SAM3D invents the
  unseen underside, so for the clips whose drop test turned > 10° the collision mesh gets a flat base: at frame 0
  on the disk, vertices within 1 cm of the table top are projected onto it. Only those clips' coacd and retarget
  are redone; `visual.obj` (looks, mass, SDF) is unchanged. Drop test, start / end turn, before → after:
  - pp_1: 22.6°/17.0° → 0.0°/5.5°;
  - prp_5: 5.8°/20.3° → 0.1°/88.0°. The placed pose is rotated 13° from the start, so the flattened base doesn't fit
    it and the tall bottle topples; only its start is meant to be fixed;
  - pp_3: 15.5°/12.9° → 18.3°/15.4°, and prp_3: 20.2°/17.3° → 20.4°/15.8°. The spatula is not fixed: its
    reconstructed pose is tilted about 7°, with the head on the table and the handle end 2.2-2.7 cm above it, so
    a 1 cm band does not reach the handle.
- **First evals** (cfhoi_r7, mean AdaDex progress over the 10 clips): 22% at 10M env steps, 20% at 20M and 16%
  at 30M, with 0/10 clips tracked end to end. This is not a plateau.
  - FDR's object assist (a PD force pulling the object to its reference, `xfrc_kp`) is still decaying: 80 → 28 by
    30M.
  - Training-time object error is 2-3 cm, against 15-18 cm in the evals, which run without assist.
  - The unassisted evals should only improve once the assist is near zero (about 60-70M steps at this rate).
