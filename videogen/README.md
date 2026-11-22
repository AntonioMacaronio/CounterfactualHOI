# videogen: counterfactual video-to-video augmentation with MiniMax H3

Edits real pick-and-place clips (swapping the object's size/shape) with a self-hosted
**MiniMax H3-Base Ref2VA** server, producing candidate demos for Do as I Do reconstruction.

H3 has no separate "v2v" task: video-to-video is Ref2VA with the source clip as `<Video 1>`
and a structured editing prompt (`summary` / `retention_analysis` / `detailed_description`,
the same layout H3-Context-IR emits). See `prompts.py`.

| mode | what changes | role in the study |
|---|---|---|
| `identity` | nothing | fidelity control: how far does Ref2VA drift from the real clip? |
| `object` | object only; hand motion must be identical | ablation arm 1 |
| `object_hand` | object + grasp adapts to the new geometry (arm path/timing kept) | ablation arm 2 |

Variants (2 size + 2 shape per clip) live in `configs/videogen/sources.json`.

## One-time setup (already done on this node)

```bash
source scripts/env.sh        # all weights/venvs/caches/outputs go to /data/sky/projects/CounterfactualHOI
# tools env: python 3.12 + ffmpeg (the client is stdlib-only)
~/miniconda3/bin/conda create -y -p "$TOOLS_ENV" -c conda-forge --override-channels python=3.12 ffmpeg
# weights: MiniMaxAI/MiniMax-H3 @ $H3_REVISION, Ref2VA partition only (~135 GiB)
hf download MiniMaxAI/MiniMax-H3 --revision "$H3_REVISION" \
  --include "model_index.json" --include "Ref2VA/*" --local-dir "$H3_MODEL_DIR"
# SGLang at the commit pinned by MiniMax's self-hosting guide
git clone https://github.com/sgl-project/sglang.git "$H3_SGLANG_SRC" && git -C "$H3_SGLANG_SRC" checkout "$H3_SGLANG_COMMIT"
uv venv --python 3.12 "$H3_SGLANG_VENV"
cd "$H3_SGLANG_SRC" && SGLANG_BUILD_RUST_EXTS=none VIRTUAL_ENV="$H3_SGLANG_VENV" \
  uv pip install -e "python[diffusion]" --prerelease=allow
```

`env.sh` sets `UV_NO_CONFIG=1` because `~/.config/uv/uv.toml` points at a CodeArtifact
index we can't authenticate to.

## Run

```bash
source scripts/env.sh
bash scripts/serve_h3_ref2va.sh          # 4 GPUs, ~94 GB each; ready when `curl $H3_URL/health` says ok

python -m videogen.generate --video-dir ../videos --dry-run              # print all prompts
python -m videogen.generate --video-dir ../videos --clips IMG_5690 \
    --variants tall_can --modes identity object object_hand --seeds 0
```

Outputs go to `$VIDEOGEN_GEN_DIR/<clip>/<mode>/<variant>/seed<k>.mp4` (default
`../outputs/minimax_h3`, next to the repo — small enough for the root fs) with the prompt, a
`_compare.mp4` (source | generated) for QA, and one line per attempt (failures included) in
`manifest.jsonl` — `gpu_seconds` there is what feeds "GPU time per valid demonstration".

## 381V group prompts (`type_a`, `type_b`)

`type_a` (geometry edit, A01-A05) and `type_b` (swap to one of ten rigid objects, B01-B10) use the group's
prompts from `antzhang-notes/prompt_prp_type_{a,b}.md`, implemented in `group_prompts.py`. The clip's `task` picks the
template: `pick_rotate_place` uses the notes verbatim. `pick_place` uses the same text with the in-hand rotation steps
replaced by a carry (applied as exact-match replacements, so an edit to the verbatim text fails loudly). Variants in
the config carry a `mode`, so each A/B id only pairs with its own mode:

```bash
python -m videogen.generate --config configs/videogen/real_20261119.json --modes type_a type_b --dry-run   # 100 prompts
```

For more varied swaps, `catalog.py` holds about 150 everyday rigid objects. Each variant gets a concrete object with
its own appearance (colour, material, pattern or label), which goes in `[TARGET_OBJECT_DESCRIPTION]`. The plain
B01-B10 shapes come out in default colours, and H3 sometimes reuses an object already in the scene.

```bash
python -m videogen.catalog --base configs/videogen/real_20261119.json --per-clip 50 --out configs/videogen/real_20261119_b500.json
python -m videogen.build_html outputs/minimax_h3/real_20261119_b500    # browse a run (open RUN/index.html)
```

## Managed jobs on ll-smo H200s

The shared dev node can't launch managed jobs, so launch them from the laptop. `scripts/sky/h3_videogen.yaml` runs
one H3 server on an `H200:4` pod and generates `CLIPS`. Its header lists the S3 inputs: the dev node's environment
tarball (restored at the same `/data` paths), the pinned weights, the reference clips and a code tarball. Outputs
go to `s3://far-research-internal/antzhan/counterfactualhoi/videogen/$RUN_NAME/` every two minutes. A recovered or
retried job pulls what is already there and only generates the missing clips.

```bash
eval "$(far-set-env-helper SKY 2>/dev/null)" && sky jobs launch scripts/sky/h3_videogen.yaml -y -d -n cfhoi-h3-pp-real1 \
  --env CLIPS=pp_anthony_real_1 --env RUN_NAME=real_20261119 --env CODE_BUNDLE=s3://... --env CODE_SHA256=...
```

## Known limits

- Output is 768p-short-edge at 24 fps (sources are 720p/30 fps), 4–15 s; Ref2VA re-generates
  rather than pixel-edits, so check the `identity` runs before trusting hand fidelity.
- The open release lacks H3-Context-IR and 2K regeneration (API-only).
- A true masked edit (object-only vs object+hand mask from Do as I Do's SAM3 masks) would need
  the ComfyUI H3 latent-noise-mask path; not wired up yet.
