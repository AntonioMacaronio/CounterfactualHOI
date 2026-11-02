# CounterfactualHOI
Repository for the CounterfactualHOI project: synthetic (counterfactual) video augmentation
for dexterous robot manipulation, combining PRISM-style video generation with
Do as I Do's reconstruction + retargeting.

## Layout

```
configs/          experiment configs (videogen/sources.json: clips + object variants)
scripts/          env.sh (storage/caches on /data), serve_h3_ref2va.sh
videogen/         MiniMax H3 Ref2VA video-to-video augmentation  -> videogen/README.md
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

Large data (weights, venvs, generated videos, checkpoints) goes under
`/data/sky/projects/CounterfactualHOI`, never in the repo — `source scripts/env.sh` first.
