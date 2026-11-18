# CounterfactualHOI
Repository for the CounterfactualHOI project: synthetic (counterfactual) video augmentation
for dexterous robot manipulation, combining PRISM-style video generation with
Do as I Do's reconstruction + retargeting.

## Layout

```
configs/          experiment configs (videogen/sources.json: clips + object variants)
scripts/          env.sh (storage/caches on /data), serve_h3_ref2va.sh
  daid/           Do as I Do on this node: setup.sh (envs + weights), run_recon.sh / finish_recon.sh
                  (headless reconstruction + table + QC), view_recon.py / build_recon_html.py (viser)
videogen/         MiniMax H3 Ref2VA video-to-video augmentation  -> videogen/README.md
antzhang-notes/   project notes (pipeline plan, prompts, results)
third_party/      external code as pinned git submodules (do not edit in place)
  PRISM/          amazon-far/PRISM (reference only; its release is humanoid training code)
  do-as-i-do/     malik-group/do-as-i-do (reconstruction + retargeting)
```

Clone with submodules:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/AntonioMacaronio/CounterfactualHOI.git
# or, in an existing clone:
git submodule update --init            # add --recursive when setting up Do as I Do's own submodules
```

Our code lives in this repo and calls into `third_party/`. If we need to change Do as I Do,
fork it, point the submodule at the fork (`git submodule set-url third_party/do-as-i-do <fork>`),
and commit the new pinned SHA here.

## Do as I Do reconstruction (step 1.1)

```bash
source scripts/env.sh
bash scripts/daid/setup.sh weights && for e in sam3 tapnet sam3d hawor; do bash scripts/daid/setup.sh env $e; done
bash scripts/daid/setup.sh check
# one clip: object text prompt, reference frame (object unoccluded, hand visible); ~2 GPU-h on one H200
RECON_GPU=0 HAND_FRAME=60 bash scripts/daid/run_recon.sh CLIP.mp4 "wooden block" 16 right RUN_NAME
bash scripts/daid/finish_recon.sh $PROJECT_STORAGE_ROOT/runs/recon/RUN_NAME wooden_block 16
$CONDA_ENVS_PATH/sam3d/bin/python scripts/daid/view_recon.py --video-dir $PROJECT_STORAGE_ROOT/runs/recon/RUN_NAME \
    --object-id wooden_block --port 8081
```

`setup.sh` applies the one-line lietorch fix from Do as I Do's `env/README.md` inside the vendored HaWoR
submodule; that edit is not committed here (it would need a fork of the submodule chain).

Large data (weights, venvs, generated videos, checkpoints) goes under
`/data/sky/projects/CounterfactualHOI`, never in the repo — `source scripts/env.sh` first.
