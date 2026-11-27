# Generating the robot data on SkyPilot (everything before §1.4)

From generated videos on S3 to stored robot trajectories: per task and trial, the FlashDexRetarget policy's rollouts
of the Sharpa hand. Each rollout has joint states, the joint targets the controller received, the object pose,
hand–object contacts, the object distance at the fingertips, and the pick-done / place-done steps. §1.4's IK step and
depth rendering then turn these into the policy's training data.

This file is the plan and the commands. **Section 3 lists what has to be built before it runs on generated videos**:
the per-stage scripts work today on the 10 real clips, but several of them still assume those clips.

## 1. Pipeline and job layout

| stage | per video | GPU | wall time per video (real clips, one H200) |
|---|---|---|---|
| A1 import + gravity | `prepare_clip.sh` | 1 | < 1 min |
| A2 first-frame mask | `first_mask.sh` (SAM3, the object's name) | 1 | ~1 min |
| A3 CHOIR stage 1 | `run_choir.sh` with `CHOIR_STAGE1_ONLY=1`: masks, SAM3D object, Dyn-HaMR + VIPE hands | 1 | 5 + 2 + 10–30 min |
| A4 reconstruction | `run_fprecon.sh`: SAM3 masks, textured SAM3D, Depth Pro 2, hand scale, FoundationPose | 1 | 5–8 min |
| A5 agentic refinement | `run_refine.sh` (Claude Opus 5.5 on Bedrock) + `review.py` (3 votes) | light | 10–40 min, mostly waiting on the LLM |
| A6 physical properties | `physprops/estimate.py` (Qwen3.8-27B, 7 samples) | shares 1 | ~50 s |
| B1 dataset | FDR import, 0.7 m table, convex parts (CPU), kinematic retargeting with the wrist hardware (CPU), bimanual pack, VLM masses | CPU | coacd 2–9 min per object; IK 20–120 s per clip |
| B2 trials | `mjwarp/trials.py make` + `mjwarp/rest_check.py --fix-trials` | 1 | seconds per task |
| B3 training | FlashDexRetarget (`mjwarp/train.sh`, or Isaac `flashdex/train.sh`) | 1 per run | ~12 h per 300M env steps on Isaac |
| B4 robot data | final evaluation: every task × trial rolled out, archived with the new fields, `done_checks.py` | 1 | minutes |

**Recommendation: two kinds of jobs, not one.**

- **Stage A, per video: 10–15 parallel jobs with 1 H200 each**, every job with its own list of videos, uploading its
  reconstructions to S3 as each video finishes.
  - Stage A is independent per video and about 1–1.3 h of wall time per video.
  - On this node 2–3 clips ran per GPU at once (A3–A5 fit side by side in 141 GB).
  - 500 videos are roughly **170–200 H200-hours** whatever the layout. The layout only changes the wall time:

| layout | videos in flight | wall time for stage A |
|---|---|---|
| one 4×H200 job | 12 | ~2 days |
| 12 jobs × 1 H200 | 36 | ~15 h |
| 25 jobs × 1 H200 | 75 | ~8 h |

  - The limit is Bedrock, not GPUs. Each video in flight is one Claude agent sending images, and so far at most 10
    have run at once. Check the account's Bedrock tokens-per-minute quota and the cost before going past ~30
    concurrent agents; `run_refine.sh` retries, but throttled agents spend their time budget waiting.
  - Small jobs also isolate failures and preemptions: a lost pod costs its own videos only, and a recovered job skips
    what is already on S3.
- **Stage B: one 4×H200 job.**
  - It downloads every reconstruction (about 5 MB per video), builds one dataset, makes and validates the trials,
    and trains.
  - Its CPU-heavy steps (convex decomposition of 500 objects, the IK) want a pod with many CPUs (64+).
- **One 4×H200 job for everything** is simpler: one environment restore and no S3 hand-off. It costs the same
  H200-hours, but stage A then takes ~2 days, and the GPUs idle while the refinement waits on Claude. It is a good
  choice for a first run of ~20–50 videos to shake out failures, before the fan-out.

**Training at 500 tasks uses FDR's one-object swap mode on MuJoCo Warp (ported 2026-11-27 and the backend's default;
`scripts/mjwarp/README.md`).**
- Each world holds one object entity, which takes that world's task's object at every reset. The default
  multi-object mode puts every task's object in every world, which is fine for 10 tasks and impossible for 500.
- Tested with 500 objects in the scene: the build takes 2.5 min. Memory is the open item (section 3, item 7).

## 2. S3 layout

Access from this dev node: `AWS_PROFILE=node-identity aws s3 ... --region us-west-2` (the bucket is add-only: no
deletes, so check every key before uploading). The pods use their role, as the H3 jobs did.

```
$CFHOI_S3 = s3://far-research-internal/antzhan/counterfactualhoi
  videogen/real_20261119_b500/<clip>/<mode>/<variant>/seed0.mp4        the 500 generated videos (exist; + seed0.prompt.txt,
                                                                       seed0_compare.mp4), 1344x768 at 24 fps
  videogen/real_20261119_b500/manifests/<clip>[_p<k>].jsonl            one line per video: clip, task, mode, variant, cls (the
                                                                       object), status, output path and size/fps (exist)
  env/robotdata-<date>.tar (+ .sha256)                                 the pods' environment (to build, section 4)
  code/<bundle>.tar.gz                                                 this repo at a commit (as for the H3 jobs)
  robotdata/<DATA_RUN>/manifest.json                                   video id -> S3 path, object name, task
  robotdata/<DATA_RUN>/recon/<VID>/                                    stage A, per video, uploaded when it finishes:
      refine/export/{object_poses.npy, mano_params.npz, table.json, object_mesh.obj, material*}, summary.json,
      critic_final*.json, report.md, phases.json                       ~5 MB: what stage B needs
      fprecon/{object_mesh.obj, material*}                             the import's fallback when the refinement kept the mesh
      physprops.json, gravity.json, status.json (done / failed at which step, with its log tail)
      full.tar.zst                                                     optional: CHOIR stage 1 + fprecon + refine runs, ~1.7 GB
  robotdata/<DATA_RUN>/dataset/                                        stage B: the FDR dataset (motion packs, objects, trials.json)
  robotdata/<DATA_RUN>/runs/<run>/                                     stage B: training checkpoints, evals, robot data
```

## 3. To build before launching on generated videos

1. **Manifest** (`scripts/sky/make_manifest.py`). It lists the 500 videos with a short VID each, the object's text,
   the task, and the shard each video belongs to. The video generation's manifests already have most of it per
   video (`cls`, `task`, `status`, the S3-relative output path), so this script merges them and adds the VIDs and
   shards.
2. **Clip-specific lookups → manifest.** Each place that assumes the real clips needs a `--object` / `--task`
   override fed from the manifest:
   - `refine/agent.py` takes the task from the VID's prefix and the object from its trailing number;
   - `fprecon/run_fprecon.sh` and `physprops/estimate.py` use built-in names keyed by that number;
   - `flashdex/cfhoi_source.yaml` lists the 10 real clips.
3. **Video format.**
   - `choir/prepare_clip.sh` rescales to 1920×1080 at 30 fps. Generated videos are 1344×768 at 24 fps, so that would
     stretch them by 1.6% and duplicate frames. It needs to keep the native size and frame rate.
   - The import (`cfhoi_source.yaml`, `fps: 30`) must then take each clip's own frame rate.
4. **Failure handling.**
   - Text prompts will sometimes find nothing (both real spatula clips needed click points). A failed step should
     write `status.json` and move on, not stop the job.
   - Videos that fail, or that the reviewer rates bad, are dropped.
5. **Stage A job** (`scripts/sky/robot_recon.yaml` + `robot_recon_run.sh`), following `h3_videogen.yaml`:
   - restore the environment tarball;
   - pull its shard of videos;
   - keep 3 videos in flight per GPU through A1–A5, with one Qwen server for A6 at the end;
   - upload each video when it finishes, as an explicit file list: the refinement's `export/` links to fprecon's
     files, and a plain `aws s3 sync` follows those links, which would upload the depth maps too;
   - on recovery, skip videos whose `status.json` is on S3.
6. **Stage B job** (`scripts/sky/robot_train.yaml` + `robot_train_run.sh`): download, then build the dataset with
   the steps of `scripts/mjwarp/README.md` "How `data_r10_vega` was made", then trials, a quality filter, training,
   the final evaluation, and the upload.
7. **Training at scale: memory of the per-object grids.** The swap mode is ported (done). What grows with the task
   count now is FDR's per-object signed-distance grid: 128³ × 4 channels of float32, about 34 MB per object, kept on
   the GPU. With 500 objects that is about 17 GB, plus a second copy while `torch.stack` builds it; the 500-object test
   used 33 GB at 20 envs. One fix is `commands.object.sdf.grid_n=64` (4 MB per object). Check first what the coarser
   grid does to the fingertip-distance observations and rewards.
8. **Credentials on the pods.** S3 works through the pod role, as for the H3 jobs. Bedrock access from ll-smo pods is
   untested. MANO's license-gated files must travel in the private tarball only.
9. **Speed check.** On a free H200, MuJoCo Warp in swap mode ran 12.3k env steps/s at 4096 envs (with trials). Most
   of each step is FDR's observation and command code, not physics. Isaac's 6.9k was measured at 2048 envs in
   another setting, so the two still need comparing on one task set.

## 4. Environment on the pods

Pack the dev node's working environment into a tarball and restore it at the same absolute `/data` paths, as
`scripts/sky/h3_videogen.yaml` does, so nothing is rebuilt:
- `conda_envs/{sam3,sam3d,choir,dynhamr,vipe,fpose,flashdex}`;
- `venvs/{depthpro2,vllm,fdr-mjwarp,hf-dl,tools}`;
- `models/daid` (SAM3D, HaWoR detector, DROID);
- `build/{CHOIR (without output/), FoundationPose, ml-depth-pro-2, FlashDexRetarget, mip-splatting, utils3d_legacy,
  embodiment}`;
- the HF cache entries for SAM3, Depth Pro 2 and Qwen3.8-27B-FP8;
- the MANO files.

That is about 150–170 GB, or 10–20 minutes to restore per pod. Leave out MiniMax H3 (135 GB), not needed here.

## 5. Stage A on one video (what the job runs; the commands exist)

```sh
source scripts/env.sh; V=<VID>; G=<gpu>; OBJ="<object text>"
bash scripts/choir/prepare_clip.sh <video>.mp4 $V $G                     # item 3: native size and fps
GPU=$G bash scripts/choir/first_mask.sh $V "$OBJ"
CHOIR_STAGE1_ONLY=1 bash scripts/choir/run_choir.sh $V $G
OBJECT_PROMPT="$OBJ" bash scripts/fprecon/run_fprecon.sh $V $G
REFINE_TAG=gen bash scripts/refine/run_refine.sh $V $G --effort high --max-turns 60 --max-minutes 45   # item 2
REFINE_TAG=gen CUDA_VISIBLE_DEVICES=$G $CONDA_ENVS_PATH/fpose/bin/python scripts/refine/review.py --samples 3 $V
REFINE_TAG=gen CUDA_VISIBLE_DEVICES=$G $CONDA_ENVS_PATH/fpose/bin/python scripts/refine/summarize.py $V
# once per job, after the last video: Qwen on one GPU, then every video's estimate
bash scripts/physprops/serve_qwen.sh $G &  ...  python3 scripts/physprops/estimate.py --refine refine_gen --name "$OBJ" $V
```

## 6. Stage B (what the job runs)

```sh
D=$PROJECT_STORAGE_ROOT/runs/flashdex/data_<DATA_RUN>
cd $FDR && bash scripts/retarget/import_human_demo.sh source=cfhoi raw=$PROJECT_STORAGE_ROOT dataset=$D sides=right \
    workers=64 source.refine=refine_gen "source.clips=[...]"            # item 2: clips from the manifest
python -I scripts/physprops/apply_mass.py $D
python -I scripts/flashdex/add_table.py $D --radius 0.7
cd $FDR && bash scripts/retarget/kinematic_retargeting.sh dataset=$D robot=sharpa_vega_1p desk.mode=keep workers=64
python -I scripts/flashdex/pad_bimanual.py $D
python3 scripts/mjwarp/trials.py make $D --n 5
venvs/fdr-mjwarp/bin/python scripts/mjwarp/rest_check.py $D --fix-trials
# quality filter (to build): drop tasks the reviewer rated bad, the retargeting rejected, or with no valid trial
# swap object mode, the backend's default (FDR_OBJ_SWAP=0 for multi-object)
bash scripts/mjwarp/train.sh $D/motion_bimanual.pt <run> --trials $D/trials.json \
    envs/robot@robot=sharpa_vega_1p_bimanual "commands.object.support_disks.radii=[0.08,0.15,0.25,0.8]" \
    num_envs=4096 runner.agent.buffer_max_length=20000000 eval.stop_when_all_solved=false num_env_steps=300000000
python -I scripts/flashdex/done_checks.py --dataset $D --archive <run>/evals/step<N>
```

## 7. After this (§1.4)

- `scripts/embodiment/ik.py` turns each rollout's wrist targets into Vega arm joints.
- `scripts/embodiment/render_depth.py` renders the wrist and exo depth images from the rollouts.
- The text command switches at each rollout's pick-done step.
- These give the flow policy's supervised data (`scripts/policy14/model.py`).
