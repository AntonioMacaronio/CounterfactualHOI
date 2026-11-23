# CounterfactualHOI

Counterfactual video augmentation for dexterous manipulation: real hand-object clips are edited with MiniMax H3,
reconstructed in 4D (hand + object), refined by an agent, and retargeted to a Sharpa Wave hand in simulation.
What each stage does and why is in `antzhang-notes/project.md`; this file is how to set it up and run it.

The reference machine is one node with 4× H200 (141 GB), 112 cores and 2 TB RAM, driver 580. Other hardware works
with the settings in [Other hardware](#other-hardware).

## Layout

```
configs/videogen/    real clips + their H3 edit variants
videogen/            MiniMax H3 Ref2VA video-to-video (client, prompts, pages)      -> videogen/README.md
scripts/
  env.sh             storage, caches and pinned paths; source it in every shell
  serve_h3_ref2va.sh, sky/   H3 server (SGLang) and the SkyPilot managed job
  daid/              Do-as-I-Do: env + weight setup, its reconstruction (baseline), viewers
  choir/             Open-CHOIR: setup, clip import + gravity, first mask, stage 1 runner, viewers / pages
  fprecon/           our reconstruction: SAM3 masks, textured SAM3D mesh, Depth Pro 2, hand scale, FoundationPose
  refine/            agentic refinement (Claude on Bedrock), reviewer, blind A/B, summaries
  retarget/          Do-as-I-Do retargeting of the refined clips (baseline)
  flashdex/          FlashDexRetarget (Isaac Lab RL) retargeting and the §1.4 policy   -> scripts/flashdex/README.md
  site/              the project website
third_party/         pinned submodules: CHOIR, do-as-i-do, PRISM (reference only); not edited in place
antzhang-notes/      project notes, prompts, results
```

Outside the repo:
- `$PROJECT_STORAGE_ROOT` (default `/data/sky/projects/CounterfactualHOI`): envs, weights, builds, runs.
- The repo's parent directory: `outputs/` (viewers and generated videos, small), `mano_v1_2/models/` (MANO),
  `realvideos/` (recorded clips) and `CounterfactualHOI-site/` (the site repo).

## Requirements

- **GPU**: NVIDIA sm_80 or newer (A100, L40S, RTX 4090, H100 / H200). The flash-attn wheels and SAM3D's
  attention need Ampere or later.
- **Driver**: 580 or newer (CUDA 13.0). The `retargeting` env and the Depth Pro 2 venv use CUDA 13 torch builds;
  every other env is CUDA 12.6 / 12.8. Each conda env brings its own nvcc, so no system CUDA toolkit is needed.
  Extension builds use the system gcc (11 here; nvcc 12.x accepts up to gcc 13).
- **Software**: `~/miniconda3` (the scripts only run `conda create -p`; the base env is never modified), `uv`,
  git, curl, wget, unzip.
- **Disk**: about 200 GB for envs, weights and builds, plus 135 GB for the H3 weights. The 10 real clips take
  about 50 GB of runs and outputs.
- **Accounts**:
  - **Hugging Face** token with access to the gated `facebook/sam3` and `facebook/sam-3d-objects`. Accept both
    licenses first. Then either `export HF_TOKEN=...` or put the token in `~/.cache/huggingface/token`
    (`env.sh` points `HF_TOKEN_PATH` there).
  - **MANO** (mano.is.tue.mpg.de, license-gated): `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` in `$MANO_DIR`
    (default `../mano_v1_2/models`).
  - **AWS** credentials with Bedrock access to Claude Opus 5.5 (`global.anthropic.claude-opus-5-5`, us-west-2).
    These are used by the refinement agent, reviewer and A/B judge. Set `AWS_PROFILE` / `AWS_REGION`;
    `REFINE_MODEL` overrides the model.
  - Optional: S3 (`s3://far-research-internal/antzhan/counterfactualhoi`) and SkyPilot, for the managed H3 jobs.

## 0. Clone

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone -b dev/antzhang https://github.com/AntonioMacaronio/CounterfactualHOI.git
cd CounterfactualHOI
git submodule update --init third_party/CHOIR
git submodule update --init --recursive third_party/do-as-i-do      # sam3, sam-3d-objects, HaWoR (+ DROID-SLAM), ...
# optional, reference only: git submodule update --init third_party/PRISM
source scripts/env.sh
```

On another machine, point the storage at a big disk before sourcing, in every shell:
`export CFHOI_STORAGE_ROOT=/big/disk/cfhoi CFHOI_CACHE_ROOT=/big/disk/cache`.

Notes:
- `env.sh` also sets `UV_NO_CONFIG=1`, because `~/.config/uv/uv.toml` on this node points at an index we can't use.
- `scripts/daid/setup.sh` patches lietorch inside the HaWoR submodule (the one-line fix from Do-as-I-Do's
  `env/README.md`) and links weights into it. That is why `git status` shows `third_party/do-as-i-do` as modified.

## 1. Setup (once, in this order)

Each step skips what already exists, so it can be re-run.

```bash
source scripts/env.sh

# helpers: hf CLI + gdown (downloads), ffmpeg (tools env, put on PATH by env.sh)
uv venv -q --python 3.12 "$PROJECT_STORAGE_ROOT/venvs/hf-dl"
uv pip install -q --python "$PROJECT_STORAGE_ROOT/venvs/hf-dl/bin/python" "huggingface_hub[hf_xet]==1.28.0"
~/miniconda3/bin/conda create -y -p "$TOOLS_ENV" -c conda-forge --override-channels python=3.12 ffmpeg
source scripts/env.sh

# Do-as-I-Do weights (SAM3D ~13 GB, SAM3, HaWoR + detector + DROID, BootsTAPIR) and envs
bash scripts/daid/setup.sh weights
for e in sam3 sam3d retargeting tapnet hawor; do bash scripts/daid/setup.sh env $e; done
bash scripts/daid/setup.sh check

# Open-CHOIR (working copy at the submodule's commit in $CHOIR_ROOT; links the DAID weights and MANO)
bash scripts/choir/setup.sh workdir
bash scripts/choir/setup.sh assets
bash scripts/choir/setup.sh bmc
for e in choir dynhamr vipe; do bash scripts/choir/setup.sh env $e; done
bash scripts/choir/setup.sh check

# FoundationPose (+ the refinement agent's packages) and Depth Pro 2
bash scripts/fprecon/setup.sh fpose
bash scripts/fprecon/setup.sh depthpro2
bash scripts/fprecon/setup.sh check

# FlashDexRetarget (Isaac Sim 5.1 / Isaac Lab 2.3), then our import source and configs linked into it
bash scripts/flashdex/setup.sh
bash scripts/flashdex/setup_cfhoi.sh
```

The H3 weights and the SGLang venv (only needed to generate videos) are set up in `videogen/README.md`.

The order matters:
- The CHOIR assets link Do-as-I-Do's weights.
- `fprecon/setup.sh fpose` reads MANO from the CHOIR working copy.
- FlashDexRetarget also takes its MANO models from there.

| env (`$CONDA_ENVS_PATH/…`, or `venvs/…`) | python / torch | used by |
|---|---|---|
| `sam3` | 3.12 / 2.8.0 cu128 | SAM3 masks (first_mask, fprecon) |
| `sam3d` | 3.11 / 2.8.0 cu128 | SAM3D mesh, GeoCalib gravity, hand scale, viser viewers |
| `retargeting` | 3.12 / 2.12.0 (CUDA 13) | Do-as-I-Do retargeting |
| `tapnet`, `hawor` | 3.10 / 2.7.0, 2.9.0 cu128 | Do-as-I-Do reconstruction only (baseline) |
| `choir`, `dynhamr` | 3.10 / 2.6.0 cu126 | CHOIR stage 1 (masks + object; hands) |
| `vipe` | 3.10 / 2.7.0 cu128 | CHOIR stage 1 camera (VIPE) |
| `fpose` | 3.11 / 2.6.0 cu126 | FoundationPose, refinement agent, reviewer, A/B, retarget input prep |
| `flashdex` | 3.11 / 2.7.0 cu128 | FlashDexRetarget import, retargeting, training, export |
| `venvs/depthpro2` | 3.12 / 2.15.0 | Depth Pro 2 |
| `venvs/sglang-h3` | 3.12 / 2.13.0 | H3 server |

Pinned upstream code:
- Submodules: CHOIR `dda0d72`, do-as-i-do `824591b`.
- Builds: FoundationPose `a1b694b`, ml-depth-pro-2 `aa389cb`, FlashDexRetarget `e84da37`.
- H3: SGLang `$H3_SGLANG_COMMIT`, weights `$H3_REVISION`.
- Python packages are pinned in the setup scripts.

## 2. Run, per clip

The real clips are `pp_anthony_real_{1..5}` (pick-place) and `prp_anthony_real_{1..5}` (pick-rotate-place). The
same five objects appear in both sets: small metal case, blue tape roll, black spatula, gray mug, dish soap bottle.
Each step runs on one GPU and skips outputs that already exist. Logs go to `$PROJECT_STORAGE_ROOT/runs/logs/`.

```bash
source scripts/env.sh
V=pp_anthony_real_1; G=0

# a. import: 30 fps 1080p mp4 in $CHOIR_ROOT/data + GeoCalib gravity (runs/choir_gravity/$V/gravity.json)
bash scripts/choir/prepare_clip.sh ../realvideos/pickplace/$V.MOV $V $G
# b. first-frame object mask (SAM3, text prompt) -> $CHOIR_ROOT/output/$V/inputs/annotations/; check first_mask_overlay.jpg
GPU=$G bash scripts/choir/first_mask.sh $V "silver case"
# c. CHOIR stage 1: masks, SAM3D object, Dyn-HaMR hands, VIPE camera (all fprecon needs)
CHOIR_STAGE1_ONLY=1 bash scripts/choir/run_choir.sh $V $G
# d. reconstruction: SAM3 masks, textured SAM3D mesh, Depth Pro 2 + hand scale, FoundationPose -> runs/fprecon/$V
bash scripts/fprecon/run_fprecon.sh $V $G          # new objects: OBJECT_PROMPT="mug,cup"
# e. agentic refinement (Claude on Bedrock, ~45 min per clip) -> runs/refine_r8/$V, outputs/refine_viser_r8/$V.html
REFINE_TAG=r8 bash scripts/refine/run_refine.sh $V $G --effort high --max-turns 60 --max-minutes 45
```

First-mask prompts used for the real clips:

| clip suffix | `first_mask.sh` prompt |
|---|---|
| `_1` | `"silver case"` |
| `_2` | `"tape roll"` |
| `_3` | points instead of text: `POINTS="775,830;980,950"` (pp), `POINTS="710,770;960,900"` (prp) |
| `_4` | `"mug"` |
| `_5` | `"dish soap bottle"` |

`run_fprecon.sh` has its own built-in prompts for these clips (looked up by the clip-name suffix); new clips need
`OBJECT_PROMPT`.

Steps a–e for several clips can run side by side, one queue per GPU. `run_choir.sh` locks per clip, and CHOIR's
torchrun gets a port per GPU. The site's round 8 ran all ten refinements at once over 4 GPUs (`GPU = i % 4`).

`bash scripts/choir/run_choir.sh $V $G` without `CHOIR_STAGE1_ONLY` also runs CHOIR's stages 2–3. That is the
CHOIR baseline; it is not needed downstream.

## 3. After a refinement round

```bash
bash scripts/refine/post_round.sh r8 r7      # TAG [PREV_TAG]; uses all visible GPUs (NGPU=...)
```

This runs, in order:
1. The automatic-fit baseline `runs/refine_r8_auto` (each clip's v001: the search tools without the agent).
2. The 3-vote reviews.
3. `summary.json` per clip.
4. Blind A/B comparisons of r8 against the initial reconstruction, the automatic fit and r7, in
   `runs/compare/*_final8`.
5. The `outputs/refine_viser_r8/` page.

`REFINE_TAG=r8 python scripts/refine/remerge.py VID ...` (fpose env) redoes only the final merge of a finished run
after a fix to it, with no LLM calls. `scripts/refine/baseline.py` is the non-agentic scripted baseline.

## 4. Retargeting

**FlashDexRetarget** (one RL policy over all clips; Sharpa Wave with a floating wrist). The full pipeline, decision
log and runs are in `scripts/flashdex/README.md`; in short:

```bash
bash scripts/flashdex/prepare_dataset.sh refine_r8 data_r7     # -> runs/flashdex/data_r8{,_gen}/motion_bimanual.pt
D=$PROJECT_STORAGE_ROOT/runs/flashdex/data_r8
bash scripts/flashdex/train.sh $D/motion_bimanual.pt cfhoi_r8 runner.agent.buffer_max_length=20000000 \
    eval.stop_when_all_solved=false num_env_steps=300000000      # -> $FDR/logs/flashsac/cfhoi_r8-<stamp>/
bash scripts/flashdex/export_viser.sh --dataset $D --archive $FDR/logs/flashsac/<run>/evals/step<N> --refine refine_r8
python3 scripts/flashdex/build_flashdex_html.py --refined-dir ../refine_viser_r8   # -> outputs/flashdex_viser/
# §1.4 policy (TrackEnv + GenEnv, AdaDex reward), optionally with a lower GenEnv share:
$CONDA_ENVS_PATH/flashdex/bin/python -I scripts/flashdex/policy/reweight_gen.py ${D}_gen/motion_bimanual.pt \
    ${D}_gen/motion_bimanual_g25.pt --gen-share 0.25
bash scripts/flashdex/policy/train_policy.sh ${D}_gen/motion_bimanual_g25.pt p14_adadex_r8_g25
```

How `prepare_dataset.sh` takes its arguments:
- The second argument copies the object meshes (convex parts, USD, SDF) from an earlier dataset. They are the same
  in every round, so this skips the slow coacd step. Leave it out the first time.
- The import picks up MANO parameters from rounds ≥ 6.

**Do-as-I-Do retargeting** (baseline; IK + MuJoCo Warp sampling, per clip):

```bash
R=$PROJECT_STORAGE_ROOT/runs
$CONDA_ENVS_PATH/fpose/bin/python scripts/retarget/prepare_refined_input.py --refine-dir $R/refine_r8/$V \
    --video-dir $CHOIR_ROOT/output/$V --out $R/retarget_in/${V}_r8
CUDA_VISIBLE_DEVICES=$G bash scripts/retarget/run_retarget.sh $R/retarget_in/${V}_r8 ${V}_r8 right
CUDA_VISIBLE_DEVICES=$G $CONDA_ENVS_PATH/retargeting/bin/python scripts/retarget/export_replay_html.py \
    --run-dir $R/retarget/sharpa/right/${V}_r8/0 --export-html ../outputs/retarget_viser/$V.html
python3 scripts/retarget/build_retarget_html.py --refined-dir ../refine_viser_r8
```

## 5. Pages and the website

Per-method pages (all in the sam3d env) go to `../outputs/*_viser/index.html`:

```bash
$CONDA_ENVS_PATH/sam3d/bin/python scripts/choir/build_choir_html.py --method fprecon
REFINE_TAG=r8 $CONDA_ENVS_PATH/sam3d/bin/python scripts/choir/build_choir_html.py --method refine
```

`--method choir` builds the page for the CHOIR baseline.

The site:

```bash
python3 scripts/site/build_site.py --out ../CounterfactualHOI-site
```

`build_site.py` writes into a clone of github.com/AntonioMacaronio/CounterfactualHOI-site (GitHub Pages, main /
root); commit and push there. Its defaults name the current rounds: `--refine refine_viser_r8`, the
`runs/compare/*_final8` comparisons, and `--retarget flashdex_viser`.

## Video generation and Do-as-I-Do reconstruction

- **MiniMax H3 edits**: `bash scripts/serve_h3_ref2va.sh` (4 GPUs, ~94 GB each), then
  `python -m videogen.generate ...`. Managed jobs go through `scripts/sky/h3_videogen.yaml`. See `videogen/README.md`.
- **Do-as-I-Do reconstruction** (the first baseline, on the IMG_5710 clips; ~2 GPU-h per clip on an H200):

```bash
RECON_GPU=0 HAND_FRAME=60 bash scripts/daid/run_recon.sh CLIP.mp4 "wooden block" 16 right RUN_NAME
bash scripts/daid/finish_recon.sh $PROJECT_STORAGE_ROOT/runs/recon/RUN_NAME wooden_block 16
$CONDA_ENVS_PATH/sam3d/bin/python scripts/daid/view_recon.py --video-dir $PROJECT_STORAGE_ROOT/runs/recon/RUN_NAME \
    --object-id wooden_block --port 8081
```

## Other hardware

- **GPU architecture.** The source builds (pytorch3d, nvdiffrast, the Gaussian rasterizer, mesh_intersection,
  DROID-SLAM, VIPE) compile for `TORCH_CUDA_ARCH_LIST`. The default is `9.0`
  (H100 / H200). Set it before section 1, e.g. `export TORCH_CUDA_ARCH_LIST=8.0` (A100) or `8.9`
  (L40S / RTX 4090). `MAX_JOBS` (default 32) caps the compile parallelism; lower it if RAM is short. The `check`
  steps print the GPU's `sm_XX`.
- **GPU memory.**
  - The H3 server needs 4 × ~94 GB. On 80 GB cards use `TP_SIZE=2 ULYSSES=2` (~63 GB/GPU; see
    `scripts/serve_h3_ref2va.sh`).
  - FlashDexRetarget's replay buffer dominates its memory. FDR's default of 50M rows needs ~110 GB; our runs use
    20M rows, ~40 GB.
  - On smaller GPUs, lower `runner.agent.buffer_max_length` or set `runner.agent.buffer_device_type=cpu`
    (~16 GB GPU + ~100 GB RAM).
  - The reconstruction and refinement steps take one GPU per clip. They have only been run on H200s, and their
    peak memory has not been measured.
- **Fewer GPUs.** Every per-clip script takes a GPU index. `post_round.sh` splits its reviews over `NGPU`.
- **Disk safety.** Long jobs here were launched through `scripts/choir/guarded.sh LOG CMD...`, which kills the
  command's own process group if `/` drops below `MIN_ROOT_FREE_GB` (default 352). That threshold is specific to
  this pod's shared root filesystem.
- **Paths specific to this node.**
  - `scripts/flashdex/auto_refresh.sh` (background re-export of tab 5) still hard-codes the `/data/sky` and
    `/home/sky` paths.
  - Python scripts fall back to `/data/sky/...` when `env.sh` has not been sourced.
