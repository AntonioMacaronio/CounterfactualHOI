# MuJoCo Warp backend, tasks and trials, wrist hardware (§1.3)

FlashDexRetarget's tracking env on MuJoCo Warp (vanilla mjlab 1.3.0) next to FDR's Isaac Sim backend, the
domain randomization of the retargeting policy as **tasks and trials**, and the hand model with the real robot's wrist
hardware. Built 2026-11-27. Everything here was smoke-tested only, on GPUs shared with other jobs. No full training
run has been made on it yet.

| file | what |
|---|---|
| `setup.sh` | venv `venvs/fdr-mjwarp`: mjlab 1.3.0, MuJoCo Warp 3.8.1, warp 1.12.1, torch 2.7 cu128, FDR's packages without Isaac; links the configs; builds the hand variant |
| `env.py` | `MjwarpTrackingEnv`: mjlab's env + FDR's `capture_final_obs` step; action delay and noise; trials at reset |
| `create.py` | builds train and eval envs from FDR's hydra config with FDR's own MuJoCo scene builder; `python create.py` builds, steps and profiles |
| `train.py`, `train.sh` | FDR's FlashSAC runner on it, with `train_cfhoi.py`'s patches (eval archives, `metrics.jsonl`, rollout fields for §1.4) |
| `trials.py` | `make DATASET`: N stored trials per task -> `DATASET/trials.json` |
| `randomize.py` | applies the trials and the per-episode robot randomization in the env |
| `rest_check.py` | drop test of every clip's rest poses, nominal + every trial, in parallel worlds; `--fix-trials` validates trials |
| `hand_variant.py` | the Sharpa hand with the arm link, mount and D405 as collision geoms -> robot `sharpa_<arm>` |
| `hardware_clearance.py` | how deep that hardware goes into the table along a retargeted dataset |
| `config/` | `simulator_mjwarp.yaml` (-> FDR `envs/simulator/mjwarp`), the generated robot config |
| `../physprops/` | object mass, friction and restitution from Qwen3.8-27B (below) |

## Run

```sh
bash scripts/mjwarp/setup.sh
D=$PROJECT_STORAGE_ROOT/runs/flashdex/data_r10_vega          # round 10, VLM masses, wrist hardware, 0.7 m table
python3 scripts/mjwarp/trials.py make $D --n 5                # 5 trials per task
$PROJECT_STORAGE_ROOT/venvs/fdr-mjwarp/bin/python scripts/mjwarp/rest_check.py $D --fix-trials
bash scripts/mjwarp/train.sh $D/motion_bimanual.pt cfhoi_r10_vega --trials $D/trials.json \
    envs/robot@robot=sharpa_vega_1p_bimanual "commands.object.support_disks.radii=[0.08,0.15,0.25,0.8]" \
    num_envs=4096 runner.agent.buffer_max_length=20000000 eval.stop_when_all_solved=false eval.interval=10000000 \
    num_env_steps=300000000
```

How `data_r10_vega` was made (from `data_r10_flat`, the round-10 dataset):

```sh
python scripts/physprops/estimate.py pp_anthony_real_{1..5} prp_anthony_real_{1..5}   # Qwen server up
cp -a data_r10_flat data_r10_flat_vlm && python -I scripts/physprops/apply_mass.py data_r10_flat_vlm
mkdir data_r10_vega && cp -a data_r10_flat_vlm/{human_demo,objects,flat_base.json} data_r10_vega/
python -I scripts/flashdex/add_table.py data_r10_vega --radius 0.7
cd $FDR && bash scripts/retarget/kinematic_retargeting.sh dataset=$D robot=sharpa_vega_1p desk.mode=keep \
    steps.convex=false steps.usd=false workers=8
python -I scripts/flashdex/pad_bimanual.py $D
```

## Tasks and trials

A task is one video's agent-refined reconstruction, one clip of a dataset. Each task gets N trials, physics variants
of its object. Trials are fixed and stored, so the §1.4 policy trains on the same ones. Robot parameters are drawn
again every episode.

| | parameter | range |
|---|---|---|
| per trial (object) | mass | VLM estimate × U(0.7, 1.3); inertia scales with it |
| | centre of mass | offset within ±10% of the extent per axis, at most 1.5 cm |
| | sliding friction (object–table) | U(VLM low, VLM high), default 0.3–0.8 |
| | torsional / rolling friction | U(0.001, 0.02) m / log-U(1e-5, 1e-3) m; object geoms get condim 6 |
| | restitution | U(0, VLM + 0.15, at most 0.5), as the contact's damping ratio |
| per episode (robot) | PD gains kp, kd | × U(0.8, 1.2) (mjlab `dr.pd_gains`) |
| | joint damping, armature, joint friction | × U(0.7, 1.3), × U(0.8, 1.2), × U(0.5, 1.5) |
| | hand friction | U(0.8, 1.5) on the hand's collision geoms |
| | action delay, action noise | 0, 1 or 2 control steps; N(0, 0.02) per action dimension |

How the trials are applied:
- Each env draws one of its clip's valid trials at every reset, after the command picks the clip, and writes it into
  its own world's model rows: body mass, inertia and CoM, and the object geoms' friction and solref.
- Friction pairing follows MuJoCo's rule of taking the larger coefficient:
  - the support disks get friction (0.1, 0, 0), so object–table contacts use the object's friction;
  - hand–object contacts use the hand's.
- The object geoms get solmix 1000, so their solref, and with it the restitution, decides their contacts.
- Restitution e maps to the damping ratio ζ = −ln e / √(π² + ln² e), with the time constant fixed at 0.02 s.

**Validation.** `rest_check.py --fix-trials` drops every clip's first and last pose under each trial. A trial whose
object turns more than 5° beyond the nominal drop has its CoM offset resampled within half the bound, up to 4 rounds.
On `data_r10_vega`, the first draw tipped the mug by up to 95°, the metal case by up to 95° and the bottle by 23°,
from offsets of 0.8–1.4 cm. After resampling, 49 of 50 trials are valid. One bottle trial still tips at its end pose
even with a near-zero offset, so its friction or restitution is the cause. It is marked invalid and never drawn.

Checked (16–64 worlds):
- each world gets its clip's trial values;
- robot gains, damping, armature, joint friction, fingertip friction and delay all differ across worlds within their
  ranges;
- the MuJoCo Warp drop test matches the CPU one at nominal physics (e.g. 17.0 vs 16.9°, 19.9 vs 20.0°).

## The MuJoCo Warp backend

FDR's env code was written against mjlab, and its release ships only an Isaac Sim backend built on a trimmed copy of
mjlab 1.3.0. This backend uses the real mjlab 1.3.0:
- **Import order.** It is imported before FDR's `envs`, which would otherwise put the trimmed copy first.
- **Scene.** FDR's own MuJoCo scene builder (`envs.scene.build_scene`).
- **Step.** The step body of FDR's Isaac twin, for `capture_final_obs`.

Multi-object mode puts every clip's object in every world and parks the unused ones in the air. FDR's one-object
"swap" mode needs `simulator.mujoco.swap_collision`, which the release does not include. With many tasks the
multi-object scene grows, so that is the next thing to port.

Settings (`config/simulator_mjwarp.yaml`):
- dt 1/240 s with decimation 4, so 60 Hz control as on Isaac;
- implicitfast integrator, Newton solver with 6 iterations;
- pyramidal cones, which do condim 6;
- nconmax 128, njmax 2048.
- Elliptic cones with 20 iterations cost 6× more for the same scene: 424 against 69 ms per 4 substeps at 1024 envs.

**Speed** (measured on a GPU shared with another job at 70–95% utilization, so these are upper bounds):

| envs | physics, 4 substeps | full env step | env steps/s |
|---|---|---|---|
| 64 | 76 ms | 475 ms | 135 |
| 1024 | 75 ms | 690 ms | 1,485 |

At 1024 envs, observations take 312 ms of the step, commands 120 ms, resets 107 ms and physics 75 ms. FDR's
observation terms are thousands of small kernel launches, which a busy GPU serialises. They need re-measuring on a
free GPU before comparing with Isaac (6.9k env steps/s alone at 2048 envs). `fast.py`'s `obs` dedup applies here too
(the actor and critic groups are identical) but is not wired in yet.

Smoke tests:
- 64 envs, 40k and 8k env steps, with trials: training and the evaluation run.
- The archive has `joint_pos_target`, `contact_force` (MuJoCo Warp's contact sensors, up to 47 N) and `link_obj_dist`.
- Open: 3 of 10 clips (pp_4, pp_5, prp_3) were not scored in that evaluation, meaning they logged no in-window steps.
  Check this on a trained policy.

## Wrist hardware

`hand_variant.py` adds the arm's last link (convex hull), the mount plate and the D405 box to the right palm, as
massless collision geoms (`build/embodiment/vega_1p_sharpa_r/wrist_stub/`).

- **Kinematic retargeting:** with `robot=sharpa_vega_1p` the IK keeps them off the table.
- **The table was too small.** On FDR's 0.3 m support disk most of the remaining penetration was past the disk's
  edge, where the simulated table ends but the real one does not. With a 0.7 m disk (`add_table.py --radius 0.7`,
  bucket 0.8 in the env) the hardware stays out of the table.

Deepest point of the hardware into the table (`hardware_clearance.py`):

| clip | retarget without the hardware | with the hardware, 0.7 m table |
|---|---|---|
| pp_2 | 3.6 cm | 0.0 cm |
| pp_3 | 1.4 cm | 0.0 cm |
| prp_2 | 4.0 cm | 0.0 cm |
| prp_3 | 2.0 cm | 0.0 cm |
| prp_4 | 4.5 cm | 0.1 cm |

The other five clips were clear in both. Fingertip error is unchanged (median 9.8–13.2 mm per clip; pp_3 9.1 -> 11.3 mm).

## Not done

- FDR's kinematic-retarget IK stays on CPU MuJoCo. It is per-frame Python around MuJoCo calls and needs a batched
  rewrite to run on MuJoCo Warp. It takes 18–123 s per clip, with clips in parallel.
- No full training run on MuJoCo Warp (only short smoke tests were allowed on the shared GPUs), so learning parity with
  Isaac is unverified.
- The Isaac backend gets neither the trials nor the robot randomization. Only the VLM mass reaches it, through
  `right_object_masses`.
