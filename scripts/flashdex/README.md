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
| `compare_runs.py` | runs side by side at equal env steps: assist gain, task reward, completion (`metrics.jsonl`) and eval progress (`curves.json`) |
| `profile_step.py` | where a training iteration spends its time (stages, GPU kernel launches and synchronizations), with and without `fast.py` |
| `fast.py` | opt-in speed patches (`CFHOI_FAST=1`) for the scene build and the training loop: applied at run time, FDR's checkout stays untouched |
| `sgs.py` | opt-in Success-Guided Sampling of the episode resets (`CFHOI_SGS=1`) |
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
- **Flat vs not, at equal steps** (mean AdaDex progress, cfhoi_r7 vs cfhoi_r7_flat): 22/22 at 10M, 20/23 at 20M,
  16/25 at 30M, 19/24 at 40M, 18/23 at 50M and 24/28 at 60M.
  - The flat run leads from 20M on. At 60M: pp_1 2 → 85%, prp_5 15 → 30%.
  - The other clips swing both ways between runs, which is run-to-run variance.
  - Position-only progress at 60M is 69% for cfhoi_r7 and 61% for flat.
  - Tab 5 switched to cfhoi_r7_flat at its 60M eval (18:12), via `auto_refresh.conf`.
- **§1.4 policy** (p14_adadex_r7) at 60M: TrackEnv progress 19% (position-only 33%), GenEnv 19% (32%), final-goal
  success 0/19. It learns much slower than FDR's privileged actor at equal steps.

- **Open option: distil the §1.4 actor** (not implemented). The §1.4 actor sees no velocities, contacts or point
  clouds and learns far slower with RL from scratch than FDR's privileged actor. Once a privileged FDR policy tracks
  well, a teacher-student distillation (DAgger: roll out the student, label with the teacher's actions) into the
  §1.4 observation space is likely the better design: RL then only fine-tunes, for example with the GenEnv goals.

- **100M steps** (cfhoi_r7_flat, tab 5 since its 19:46 auto-export).
  - Mean AdaDex progress 46%, position-only 80%.
  - pp_1 and pp_5 are tracked end to end within 5 cm / 20°; FDR's success on 6/10.
  - cfhoi_r7 at the same steps: 34% / 79%, with 0 end to end.
  - The spatula clips (pp_3, prp_3) stay low on AdaDex progress. That is the tilted rest pose; it needs fixing in
    the refinement.
- **§1.4 policy learning curve** (p14_adadex_r7, 66% GenEnv). Learning starts after the assist decays:

  | | 60M | 80M | 90M | 100M |
  |---|---|---|---|---|
  | TrackEnv position-only progress | 33% | 44% | 60% | 63% |
  | TrackEnv FDR success | 1/10 | 3/10 | 4/10 | 3/10 |
  | GenEnv final-goal success (object within 5 cm / 20° of its new goal at the end) | 0% | 21% | 26% | 26% |

- **GenEnv share A/B at 30M** (g25 vs g66):

  | | g25 | g66 |
  |---|---|---|
  | TrackEnv position-only progress | 39% | 26% |
  | TrackEnv FDR success | 2/10 | 0/10 |
  | TrackEnv object error | 17.0 cm | 20.4 cm |
  | GenEnv position-only progress | 28% | 26% |

  AdaDex progress is the same in both (19%).
  - At 40-60M the gap closes: position-only progress (TrackEnv / GenEnv) is 34 / 33% for g25 and 33 / 32% for g66
    at 60M.
  - Inconclusive so far; compare again at 90-130M.
  - g66 at 130M: TrackEnv AdaDex progress 29%, GenEnv 43%, GenEnv final-goal success 37%.

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

## Round 10 (2026-11-24)

`prepare_dataset.sh refine_r10` with nothing copied: round 10 rescales every object mesh, so convex parts, USD and
SDF are rebuilt from `export/object_mesh.obj` (coacd: 2-9 min per object, 31 min for the two handled objects, 20-22
parts). 10 clips, 0 retargeting errors (`data_r10`, `data_r10_gen`).

- **Drop test** (`rest_check.py`, turned at the start / at the end of the clip):

  | clip | round 7 | round 10 | `data_r10_flat` |
  |---|---|---|---|
  | pp_1 | 22.6° / 17.0° | 9.8° / 3.8° | same |
  | pp_2 | 0.2° / 4.3° | 0.9° / 2.3° | same |
  | pp_3 (spatula) | 15.5° / 12.9° | 16.9° / 10.9° | same |
  | pp_4 | 5.0° / 3.6° | 7.9° / 1.7° | same |
  | pp_5 (bottle) | 2.7° / 4.0° | 8.8° / 26.4° | 0.0° / 1.4° |
  | prp_1 | 3.6° / 12.0° | 5.4° / 15.2° | same |
  | prp_2 | 1.8° / 2.1° | 1.3° / 1.0° | same |
  | prp_3 (spatula) | 20.2° / 17.3° | 20.0° / 17.6° | same |
  | prp_4 | 7.4° / 0.3° | 1.4° / 3.4° | same |
  | prp_5 (bottle) | 5.8° / 20.3° | 52.8° / 49.3° | 0.1° / 5.2° |

  Round 10 fixes the placed poses' height (the object centre ends within 0.5 cm of its start height; 0.7-2.6 cm
  above it in round 7) and pp_1. The bottle of prp_5 now topples at frame 0: its long axis is 12.9° from vertical
  with the centre 11.8 cm up and a 4 x 2 cm patch on the table. The spatula still rests on its head with the handle
  in the air.
- **`data_r10_flat`**: `flat_base.py --band 0.02` for pp_5 and prp_5. Rule: a clip whose drop test turns more than
  10° at either end, and whose object rests on the same face at both ends, gets the flat base if that lowers the
  turn; tried on pp_3, pp_5, prp_3, prp_5. A 1 cm band leaves prp_5's placed pose at 22.7° (the base of a bottle
  leaning 13° rises 1.2 cm across its width), 2 cm gives 5.2°. The spatula does not improve (pp_3 20.6° / 15.7°,
  prp_3 19.8° / 17.1° with a 1 cm band) and keeps its mesh; prp_1 is turned over during the clip, so a base cut at
  its first frame does not apply to its last. `data_r10_flat4` and `data_r10_flat_band1` are the variants not used.
- **Smoothness** (asked for 2026-11-24: the retargeted hand jitters). FDR's own regulariser is `action_penalty`
  -0.01 on the mean squared action; `eval/action_rate_l2` (the squared change of the 56 actions between steps) grew
  from 0.04 at 10M to 2.0-3.7 from 70M on in `cfhoi_r7_flat`. `train.sh` now gives FDR's `action_rate` term the
  weight -0.02 by default (`CFHOI_ACTION_RATE=<weight>` changes it, `=0` leaves it out). Its size, measured in the
  simulator next to the other terms (weight x value per step): 0.04-0.07 at that jitter (3-5% of the ~1.4 the
  tracking terms give), 0.06 for exploration noise of std 0.15 per action, 0.73 for uniformly random actions (the
  object-tracking term is 0.7-0.8). Since 07:13 `train.sh` also weights FDR's object-velocity terms by default
  (`CFHOI_OBJ_SMOOTH`); the runs below started before that and do not have them.
- **Runs** (the round-10 pair with `CFHOI_FAST=1,contacts`; all four 4096 envs, 4 updates per iteration,
  `num_steps_per_env=16`, 20M-row buffer, 300M env steps planned, evaluation every 10M):

  | run | GPU | |
  |---|---|---|
  | `cfhoi_r10_flat-1124-060850` | 0 | `data_r10_flat`, `action_rate` -0.02 (the default); started 06:08, build 5-8 min |
  | `cfhoi_r10_flat_nosmooth-1124-064040` | 1 | the same with `CFHOI_ACTION_RATE=0`: the control for the smoothness term; started 06:40 |
  | `cfhoi_r7_flat_fast4k` | 2 | round 7, uniform resets: the 4096-env recipe against `cfhoi_r7_flat` (started 03:44) |
  | `cfhoi_r7_flat_sgs4k` | 3 | round 7, `CFHOI_SGS=1`: SGS against `cfhoi_r7_flat_fast4k` (started 04:32) |

- **Paused at 10:01** (the user asked for the GPUs; `kill -USR1`, FDR's graceful stop). Each run saved `stop<N>/`
  (networks, replay buffer, `resume_state.json`; 29 GB) in its run dir:

  | run | stopped at | evaluations |
  |---|---|---|
  | `cfhoi_r10_flat-1124-060850` | 116.5M env steps | to 110M |
  | `cfhoi_r10_flat_nosmooth-1124-064040` | 102.9M | to 100M |
  | `cfhoi_r7_flat_fast4k-1124-034417` | 178.4M | to 170M |
  | `cfhoi_r7_flat_sgs4k-1124-043223` | 139.0M | to 130M |

  To resume one: `agent_load_path=<stop dir> buffer_load_path=<stop dir> step_offset=<N> logger.wandb.resume=true
  logger.wandb.resumeid=<wandb_id in resume_state.json>` on top of the original overrides and environment
  variables, **plus the switches for the rewards `train.sh` has added since these runs started**, or the resumed
  run trains on a different reward: `CFHOI_OBJ_SMOOTH=0` for all four (the object-velocity terms, default since
  07:13), and `CFHOI_ACTION_RATE=0` for the two round-7 runs (launched before the action-rate default). SGS's
  per-cell history is not in the checkpoint; it refills within about 100 episodes per cell.
- **Where they stood** (mean AdaDex progress / clips tracked end to end, averaged over the evaluations from 80M
  to 110M env steps; the object assist is at 4 -> 1 of 80 there, so these are early numbers):

  | | round 10, smoothness | round 10, no smoothness | round 7, 4096 envs | round 7, 4096 envs + SGS | round 7, 2048 envs |
  |---|---|---|---|---|---|
  | progress | 39% | 40% | 32% | 29% | 41% |
  | end to end | 20% | 23% | 5% | 8% | 8% |
  | `eval/action_rate_l2` at 100M | 0.24 | 1.48 | 2.21 | - | 2.67 |

  - Smoothness: the action-rate term cuts the action jitter 6-10x (0.16-0.24 against 1.5-2.4 from 70M on) with the
    same tracking (39% against 40%; task reward 0.81 against 0.82). Per clip the two differ by more than that in
    both directions (pp_1 97% against 74%, pp_5 28% against 72%), which is the noise of 3-4 single rollouts.
  - Round 10 against round 7 with the same recipe: 39-40% against 32%, and 20-23% of clips end to end against 5%.
    The gain is in pp_2 (83% against 38%) and the spatula pick-and-place pp_3 (51% against 20%).
  - The pick-rotate-place clips are still mostly not tracked in any run (prp_3 1%, prp_1 and prp_4 10-17%, prp_5
    14-24%, prp_2 18-37%).

## Speed (2026-11-24)

Measured with `profile_step.py` (the runner's own loop body, a 180M-step policy loaded). All of it was measured on
GPUs shared with a training, so the absolute rates are low; a run alone does 3.4 it/s at 2048 envs (6.9k env steps/s).

- **A run alone keeps an H200 25-50% busy.** One iteration at 2048 envs issues about 10,700 GPU kernel launches,
  41,000 torch operator calls and 1,400 GPU synchronizations: the loop is thousands of small operations that each
  wait for the GPU. Shares of an iteration: physics (4 substeps) 23%, resets 22% (about 15 envs reset on every step;
  Isaac Lab's scene reset is 13%, the command's reset 8%), observations 19%, the command's logging metrics 12%,
  learner (2 updates) 8%, the rest 16%. The numpy round trip of observations and actions is under 2%.
- **Two FDR processes on one GPU are slower than one.** Sharing GPU 3, a 2048-env profile and the `cfhoi_r7` training
  each ran at 1.45 it/s (3.1-3.4 alone): 2.9 it/s together. Every synchronization waits for the other process's time
  slice. One run per GPU, or MPS.
- **More envs is the lever.** The iteration time depends little on the env count (the same ~11,000 kernel launches,
  larger tensors). Each measured next to one training on its GPU:

  | envs | it/s | env steps/s | vs 2048 | GPU memory (2M-row buffer) | scene build |
  |---|---|---|---|---|---|
  | 2048 | 1.45 | 2,970 | 1x | 13.0 GB | 19 min (12 alone) |
  | 4096 | 1.30 | 5,310 | 1.79x | 15.5 GB | 40 min |
  | 8192 | 1.04 | 8,530 | 2.88x | 21.1 GB | 92 min |

  Physics is what grows (1.8x from 2048 to 8192); resets, observations and the command stay within 20%. The envs
  need little memory (5.7 / 7.5 / 12.2 GB of simulation at 2048 / 4096 / 8192, the last with 4x the PhysX patch
  buffer); the replay buffer fills an H200, at 1.76 GB per million rows. The cost was the scene build (FDR's own,
  in the table; those builds ran while other scenes were building): next bullet.
- **Why the scene build is slow, and what `CFHOI_FAST=1` does about it.** With one object per env FDR cannot build
  one env and let PhysX replicate it (`replicate_physics=False`), so every prim of every env is created by a
  single-threaded Python loop. cProfile of a 512-env build (356 s): 69% created the table "support disks"
  (`support_disks.count` x `radii` = 16 kinematic cylinders per env through Isaac Sim's `create_prim`, ~30 ms each;
  our clips use one disk, of one radius), 18% added each env's object (four fifths of that is an asynchronous "does
  this file exist" query that sleeps at least 0.1 s per call), 6% was PhysX loading the stage. Past 512 envs a
  fourth cost shows: after each of its three clone calls Isaac Sim's cloner switches the Fabric USD notice handler
  back on, which re-synchronises Fabric with the stage and grows faster than the env count (5.6 s at 512 envs, 72 s
  at 2048). `fast.py`'s build patches build only the disk slots and radii the motion file uses (`disks`), check
  local files with `os.path.isfile` (`usdcheck`) and switch the handler on once, just before physics is loaded
  (`fabric`):

  | envs | FDR's build | `disks` + `usdcheck` | + `fabric` |
  |---|---|---|---|
  | 512 (under cProfile) | 356 s | 61 s | 49 s |
  | 2048 (under cProfile) | - | 288 s | 186 s |
  | 4096 | 27.5 min (`cfhoi_r7_flat_fast4k`) | 10.9 min | 6.1 min |

  Checked at 512, 2048 and 4096 envs: every env's disk sits at its clip's pose (0.001 mm at most, same radius), and
  the physics state after a reset and two steps agrees with the unpatched Fabric handling within the difference
  between two identical builds. What is left at 2048 envs: PhysX loading the stage 36%, one disk per env 22%, the
  objects 14%, cloning the robot 11%. The one-disk scene also makes resets cheaper (16 of the 18 assets a scene
  reset touched were disks): 126 -> 83 ms per iteration at 2048 envs. Not done: spawning the disk into env_0 before
  the clone (the clone would copy it; a few lines in FDR's `build_isaac_scene`), and a `create_prim` without Isaac
  Sim's per-prim `XFormPrim` view (10 ms per prim).
- **`CFHOI_FAST=1`: 1.23x** at 2048 envs (1.40 -> 1.71 it/s in one process; 10,700 -> 8,450 kernel launches, 1,400
  -> 1,040 synchronizations per iteration). Four training-loop patches, none changes what is learned: the critic
  observation group reuses the actor's (they are identical with `share_actor_critic_obs`, and FDR computed every
  term twice); the command's ~40 logging metrics are computed every 4th step in training; a reset reads its logging
  scalars back from the GPU once, not one by one (checked against FDR's own code: `CFHOI_FAST_CHECK=30`); a scene
  reset resets each asset once, not twice.
- **`contacts`: the contact observation was 40% of the observation time.** Per-term timing (`profile_step.py
  --obs-terms`, 2048 envs): `robot_body_contact_wrench_wrist` takes 29 of the 77 ms the 21 actor terms need, the
  next term 6.5 ms. FDR reads the link contact forces from one Isaac Lab ContactSensor per link (21 per hand), each
  with its own GPU round trip. `CFHOI_FAST=1,contacts` reads them from one PhysX contact view per hand (the first
  `CFHOI_FAST_CHECK` steps are compared with the sensors: equal up to float rounding, 1.2e-4 N next to forces of
  8,900 N under a random policy). Nothing else reads the 42 sensors, but the scene still updates each of them on
  every physics substep and resets them (about 1,050 kernel launches per iteration, and most of what the scene
  reset cost), so the patch then switches that off. With it the patches give 1.56x in one process at 2048 envs
  (1.47 -> 2.28 it/s; 10,590 -> 6,970 kernel launches, 1,290 -> 590 synchronizations; the reset stage 119 -> 51
  ms), against 1.15x without `contacts` in the same setting. It is not part of `CFHOI_FAST=1` until a full run has
  used it: the round-10 runs do.
- **4096 envs: the same curve while the object assist is on, about 10 points behind after it.**
  `cfhoi_r7_flat_fast4k` (`CFHOI_FAST=1`, 4096 envs, 4 updates per iteration, `num_steps_per_env=16`) against
  `cfhoi_r7_flat` (2048 envs), at equal env steps (`compare_runs.py`; never compare at equal wall time). The assist
  gain is identical at every mark, and through 90M env steps the task reward is within 0.02 and the evaluations
  within 2 points. From 100M to 170M (assist below 2 of 80) the 4096-env run's task reward is 0.01-0.05 lower and
  its evaluations average 37.5% progress against 47.0% (8 evaluations each, lower in 7 of them; 9% against 15% of
  clips end to end). One run each and one rollout per clip per evaluation, so part of this can be noise, but it is
  not the equivalence the first 70M suggested. The patches themselves do not change what is learned (same
  observations, rewards, resets), so this is the recipe: twice the data between updates. It collected 11.4k env
  steps/s alone on a GPU (2.79 it/s) against 6.9k, which about cancels: at equal wall time the two are level. The
  safer use of the patches is FDR's own 2048 envs with `CFHOI_FAST=1,contacts` (1.56x per iteration, not yet run
  as a training).
- More envs lowers the updates per env step: scale `updates_per_interaction_step` (or the batch size) with
  `num_envs` to keep the replay ratio (2 updates of 4096 samples per 2048 env steps).
- **The object-assist curriculum paces a run, in iterations.** FDR lowers the assist gain by 0.9x at most once per
  `decay_cooldown_epochs` (40) epochs of `num_steps_per_env` (32) iterations, i.e. every 1,280 iterations, and 80 ->
  0 takes 70 decays. In `cfhoi_r7_flat` every gate passed with margin (task reward 0.84 against 0.6, completion 0.94
  against 0.65), so the assist fell at exactly that pace: below 1 at 114M env steps, 0 at 189M = 92,400 iterations =
  7.6 h. Evaluations run without the assist, so their scores only mean something near the end of that. More env
  steps per second therefore do not shorten a run by themselves: with k times the envs, set
  `curriculum.xfrc_curriculum_adaptive.params.num_steps_per_env=32/k` (the epoch length, used only there) so the
  schedule stays the same in env steps. Faster iterations (`CFHOI_FAST`) shorten it directly.
- Not done, in order of expected gain: resets every k steps instead of every step (22% of an iteration is reset
  overhead that does not depend on how many envs reset), one contact sensor for all 13 links per hand instead of 13
  (the contact observation alone is 4%), a right-hand-only robot (the parked left hand is simulated and observed),
  `torch.compile` of the observation / reward terms. The learner's networks are already compiled; its CUDA-graph
  modes are off because they crash inside Isaac Sim (rsl_rl_flashsac's `resolve_compile_mode`).

## Success-Guided Sampling (`sgs.py`)

[SGS](https://sgs-rl.github.io/) (Zhang et al., CoRL 2026) changes only which task configuration an env resets to:
a fixed set of configurations, a ring buffer of the last 100 episode outcomes per configuration, and a draw weighted
towards the configurations the policy solves about half the time (weight ∝ ((p + ε)(1 - p + ε))^(1/4) with the
paper's manipulation settings: a never-solved one keeps 0.14x the peak, a 92%-solved one 0.73x). The paper's
experiments are PPO with up to 2^20 envs; nothing in the sampler depends on the learner.

Here a configuration is (clip, 0.5 s start-frame bin): 114 cells for the 10 TrackEnv clips. An episode succeeds
when it ends by reaching its clip's end, and fails on any other termination. FDR pins each env to one object, so an
env draws among its own object's cells (its clips' start bins); the share of envs per object stays fixed. A cell with
no outcome yet gets the peak weight. `CFHOI_SGS=1` installs it; `<run>/sgs.jsonl` logs every cell's success rate and
draw probability every 2 minutes. Checked in the simulator (bookkeeping, draws inside the env's clip). Whether it
learns faster than FDR's uniform start frames is being tested at equal env steps: `cfhoi_r7_flat_sgs4k` (started
2026-11-24 04:32) against `cfhoi_r7_flat_fast4k` (03:44), same data and settings (`compare_runs.py`). At 30M env
steps (assist gain 30 of 80) the sampler already separates the cells: in pp_1 the eight start bins succeed 95, 91,
45, 29, 78, 93, 100, 100% of the time and are drawn with probability 0.13, 0.14, 0.19, 0.18, 0.17, 0.14, 0.03,
0.03 (the bins around the grasp and lift get 6-7x the always-solved end of the clip); pp_4, solved from every bin,
stays uniform. One thing to read with care: the assist curriculum's completion gate (0.65; dial-back below 0.5)
sees the episodes SGS chose, which are harder than a uniform draw (completion 0.86 against 0.88 at 25M; the
assist gain stayed the same in both runs).

Result so far (both runs were paused at 10:01, the SGS run at 139M env steps): no gain. Over the evaluations from
100M to 130M SGS averages 31% progress against 35% for uniform resets (4 evaluations each, 5% of clips end to end in
both), and its training task reward is 0.00-0.03 lower. That is within the noise of single rollouts, and the run
stopped before the phase without the assist had lasted long, so it is not a verdict; but nothing here says SGS helps
on these 10 clips. With the paper's settings most of the draw goes to cells that are 85-97% solved (a cell at 95%
gets 4.7x the weight of one at 100%, and as much as one at 27%), so it mostly removes the fully solved bins.

