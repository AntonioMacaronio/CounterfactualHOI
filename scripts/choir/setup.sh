#!/usr/bin/env bash
# Set up Open-CHOIR (third_party/CHOIR, pinned) on this H200 node, with everything large on /data.
#
#   bash scripts/choir/setup.sh workdir        # working copy at the submodule's commit on /data
#   bash scripts/choir/setup.sh assets         # CHOIR release weights + public checkpoints + MANO links
#   bash scripts/choir/setup.sh bmc            # Dyn-HaMR biomechanical-constraint arrays (Hand-BMC-pytorch)
#   bash scripts/choir/setup.sh env choir|dynhamr|vipe
#   bash scripts/choir/setup.sh check          # CHOIR's asset checker + import smoke tests
#
# CHOIR downloads assets into, and writes outputs under, its own source tree, so we run it from a working
# copy on /data ($CHOIR_ROOT) pinned to the submodule commit; the submodule only records the version.
# Weights shared with Do-as-I-Do (SAM3D, WiLoR detector, DROID) are linked, not re-downloaded.
# Upstream pins torch 1.13/cu117 for Dyn-HaMR, which predates sm_90; we build it on torch 2.6/cu126.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

C="$CHOIR_ROOT"
CONDA="$HOME/miniconda3/bin/conda"
HFBIN="$PROJECT_STORAGE_ROOT/venvs/hf-dl/bin"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}" MAX_JOBS="${MAX_JOBS:-32}" FORCE_CUDA=1   # 9.0 = H100/H200
CU126=(cuda-version=12.6 cuda-nvcc cuda-cudart-dev cuda-libraries-dev cuda-cccl)

link() {  # link SRC DST unless DST is a real file
  mkdir -p "$(dirname "$2")"
  if [ -e "$2" ] && [ ! -L "$2" ]; then echo "  keep $2"; return; fi
  ln -sfn "$1" "$2"
}

workdir() {
  local pin; pin="$(git -C "$PROJECT_ROOT/third_party/CHOIR" rev-parse HEAD)"
  [ -d "$C/.git" ] || git clone -q https://github.com/hxwork/CHOIR.git "$C"
  git -C "$C" fetch -q origin && git -C "$C" checkout -q "$pin"
  # Dyn-HaMR forces PyOpenGL onto OSMesa, which this node lacks; let run_choir.sh pick EGL (NVIDIA) instead.
  sed -i 's/^os.environ\["PYOPENGL_PLATFORM"\] = "osmesa"$/os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")/' \
    "$C/stage1_preprocess/Dyn_HaMR_new/dyn-hamr/vis/viewer.py"
  # Stage 3's GraspFlowMatching torchrun uses port 29500 + local device, i.e. 29500 for every one-GPU run;
  # let run_choir.sh give each GPU its own port so clips on different GPUs can run at the same time.
  sed -i 's/^            _master_port = 29500 + int(_local_dev)$/            _master_port = int(os.environ.get("CHOIR_GFM_MASTER_PORT", 29500 + int(_local_dev)))/' \
    "$C/stage3_temporal_optimization/diffusion-vas/choir_stage3/entrypoints/demo_fitting_5stages_sam3d_reset.py"
  # Its run_temporal_optimization.py wrapper runpy's the entrypoint as __main__ without alter_sys, so the
  # entrypoint's mp.spawn cannot pickle worker_main_gpu (lookup on the wrapper's __main__ fails).
  sed -i 's/^        run_name="__main__",$/        run_name="__main__", alter_sys=True,/' \
    "$C/stage3_temporal_optimization/diffusion-vas/run_temporal_optimization.py"
  # fit_and_visualize_pose reads a module-global `args` that only exists when the entrypoint runs in-process;
  # in mp.spawn workers it is undefined (NameError just before saving). Treat it as absent there.
  sed -i 's/getattr(args, "export_ho3d_eval_data", False)/getattr(globals().get("args"), "export_ho3d_eval_data", False)/' \
    "$C/stage3_temporal_optimization/diffusion-vas/choir_stage3/entrypoints/demo_fitting_5stages_sam3d_reset.py"
  echo "CHOIR working copy $C @ $(git -C "$C" log --oneline -1)"
}

assets() {
  local zip="$TMPDIR/choir_release_assets.zip"
  [ -f "$C/stage2_grasp_correction/GraspFlowMatching/results/050-Linear-velocity-None/checkpoints/0040000.pt" ] || {
    curl -L --fail -o "$zip" https://github.com/hxwork/CHOIR/releases/download/demo-data-v1/choir_release_assets.zip
    (cd "$C" && bash tools/install_choir_assets.sh "$zip") && rm -f "$zip"; }
  # Shared with Do-as-I-Do: SAM3D checkpoints (13 GB), WiLoR detector, DROID weights.
  link "$DAID_WEIGHTS/sam3d-download/checkpoints" "$C/stage1_preprocess/sam-3d-objects/checkpoints/hf"
  for d in stage1_preprocess/Yolov8/models/wilor_hand_detector.pt \
           stage1_preprocess/Dyn_HaMR_new/third-party/hamer/pretrained_models/detector.pt; do
    [ -f "$C/$d" ] || { mkdir -p "$(dirname "$C/$d")"; cp "$DAID_WEIGHTS/hawor/external/detector.pt" "$C/$d"; }
  done
  link "$DAID_WEIGHTS/hawor/external/droid.pth" "$C/stage1_preprocess/Dyn_HaMR_new/_DATA/droid.pth"
  # MANO (license-gated; user-provided copy).
  for d in stage1_preprocess/Dyn_HaMR_new/_DATA/data/mano \
           stage2_grasp_correction/DexGraspNet_table/grasp_generation/mano \
           stage2_grasp_correction/DexGraspNet_table/grasp_generation/mano/models; do
    link "$MANO_DIR/MANO_RIGHT.pkl" "$C/$d/MANO_RIGHT.pkl"
    link "$MANO_DIR/MANO_LEFT.pkl" "$C/$d/MANO_LEFT.pkl"
  done
  # Public checkpoints via CHOIR's own script (hf + gdown from the hf-dl venv; Dyn-HaMR's prepare.sh).
  UV_NO_CONFIG=1 uv pip install -q --python "$HFBIN/python" gdown   # hf-dl is a uv venv (no pip)
  (cd "$C" && PATH="$HFBIN:$PATH" bash tools/download_external_assets.sh)
}

bmc() {
  local b="$PROJECT_STORAGE_ROOT/build/Hand-BMC-pytorch" v="$PROJECT_STORAGE_ROOT/venvs/bmc"
  [ -x "$v/bin/python" ] || UV_NO_CONFIG=1 uv venv -q --python 3.11 "$v"
  local py="$v/bin/python"
  # numpy<1.24: calculate_convex_hull.py builds ragged (object) arrays implicitly, an error since 1.24
  UV_NO_CONFIG=1 uv pip install -q --python "$py" "numpy<1.24" "matplotlib<3.9" tqdm
  [ -d "$b" ] || git clone -q https://github.com/MengHao666/Hand-BMC-pytorch.git "$b"
  if [ ! -d "$b/joints" ]; then
    (cd "$b" && "$HFBIN/gdown" -q 1_wV8QjmsVCMBEBhm56gFA2XTyU8VEHzk -O joints.zip && unzip -q joints.zip && rm joints.zip)
  fi
  UV_NO_CONFIG=1 uv pip install -q --python "$py" rdp
  # calculate_convex_hull.py always maximizes a plot window (--visualize can't be turned off); stub it.
  (cd "$b" && MPLBACKEND=Agg "$py" calculate_bmc.py && MPLBACKEND=Agg "$py" -c "
import runpy, sys, types
from matplotlib.backend_bases import FigureManagerBase
FigureManagerBase.window = types.SimpleNamespace(showMaximized=lambda: None)
sys.argv = ['calculate_convex_hull.py']
runpy.run_path('calculate_convex_hull.py', run_name='__main__')")
  mkdir -p "$C/stage1_preprocess/Dyn_HaMR_new/_DATA/BMC"
  cp "$b"/BMC/*.npy "$C/stage1_preprocess/Dyn_HaMR_new/_DATA/BMC/"
  ls -la "$C/stage1_preprocess/Dyn_HaMR_new/_DATA/BMC/"
}

mkenv() {  # mkenv NAME PYTHON [extra conda pkgs...]
  E="$CONDA_ENVS_PATH/$1"; local py="$2"; shift 2
  [ -x "$E/bin/python" ] || "$CONDA" create -y -q -p "$E" -c conda-forge --override-channels \
    "python=$py" "ffmpeg=7.1" "$@"
  PIP="$E/bin/python -m pip"
  $PIP install -q --upgrade pip
  export CUDA_HOME="$E" PATH="$E/bin:$PATH"
  export CPATH="$E/targets/x86_64-linux/include${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$E/targets/x86_64-linux/lib:$E/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
}

env_choir() {
  mkenv choir 3.10 "${CU126[@]}"
  local reqs="$TMPDIR/choir_reqs.txt"
  # chumpy needs --no-build-isolation; mesh_intersection is the vendored torch_mesh_intersection (built below).
  # mmcv 1.3.9's setup.py imports pkg_resources, which setuptools>=81 (incl. the pinned 82.0.1) dropped:
  # keep setuptools<81 and build mmcv without isolation.
  # The pins are not jointly installable: gradio==5.48.0 (imported by SAM3D's notebook/inference.py) needs
  # pillow<12 but pillow==12.1.1 is pinned. Use pillow 11.3.0, the newest gradio accepts.
  # Likewise trimesh==3.23.5 calls ndarray.ptp, removed in the pinned numpy==2.2.5: use trimesh 4.9.0, plus
  # rtree, which stage 3's contact-map step needs for trimesh proximity queries but is not listed.
  grep -vE "^(chumpy|mesh_intersection|mmcv|setuptools|pillow|trimesh)" "$C/requirements-choir.txt" > "$reqs"
  printf "pillow==11.3.0\ntrimesh==4.9.0\nrtree\n" >> "$reqs"
  $PIP install -q torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126
  $PIP install -q "setuptools<81" wheel
  $PIP install -q --no-build-isolation mmcv==1.3.9
  $PIP install -q -r "$reqs"
  $PIP install -q --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da"
  $PIP install -q -e "$C/stage2_grasp_correction/GraspFlowMatching/manotorch"
  $PIP install -q -e "$C/stage1_preprocess/sam-3d-objects/MoGe"
  $PIP install -q torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
  $PIP install -q kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu126.html
  $PIP install -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post1/flash_attn-2.7.1.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
  echo "[choir] building pytorch3d 0.7.8 and mesh_intersection for sm_$TORCH_CUDA_ARCH_LIST"
  $PIP install -q --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"  # sic: capital V
  $PIP install -q --no-build-isolation "$C/stage3_temporal_optimization/diffusion-vas/torch_mesh_intersection"
  # SAM3D (Stage 1 object) needs nvdiffrast and the Gaussian rasterizer for rendering/texture baking, which
  # requirements-choir.txt omits (run_reconstruction.py then logs the error and still exits 0).
  $PIP install -q --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0"
  local ms="$PROJECT_STORAGE_ROOT/build/mip-splatting"
  [ -d "$ms" ] || git clone -q --recursive https://github.com/autonomousvision/mip-splatting.git "$ms"
  $PIP install -q --no-build-isolation "$ms/submodules/diff-gaussian-rasterization"
  # SAM3D's mesh post-processing needs six utils3d.torch functions that MoGe-2's pinned utils3d (v1.3)
  # removed: install the old utils3d (@3913c65) as utils3d_legacy and a hook that fills in just those.
  local u="$PROJECT_STORAGE_ROOT/build/utils3d_legacy"
  if [ ! -d "$u/utils3d_legacy" ]; then
    mkdir -p "$u/src"
    curl -sL https://github.com/EasternJournalist/utils3d/archive/3913c65d81e05e47b9f367250cf8c0f7462a0900.tar.gz \
      | tar -xz -C "$u/src" --strip-components=1
    cp -r "$u/src/utils3d" "$u/utils3d_legacy"
    grep -rlE "(from|import) utils3d" "$u/utils3d_legacy" --include=*.py | xargs -r sed -i -E 's/\b(from|import) utils3d\b/\1 utils3d_legacy/g'
  fi
  local sp; sp="$("$E/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
  sed "s|@UTILS3D_LEGACY_DIR@|$u|" "$PROJECT_ROOT/scripts/choir/utils3d_compat.py" > "$sp/utils3d_compat.py"
  echo "import utils3d_compat" > "$sp/utils3d_compat.pth"
  # One of the installs above removes OpenCV (it gets swapped between variants); restore the pinned one.
  $PIP install -q opencv-contrib-python==4.11.0.86
}

env_dynhamr() {
  mkenv dynhamr 3.10 "${CU126[@]}"
  local D="$C/stage1_preprocess/Dyn_HaMR_new" reqs="$TMPDIR/dynhamr_reqs.txt"
  $PIP install -q torch==2.6.0+cu126 torchvision==0.21.0+cu126 --index-url https://download.pytorch.org/whl/cu126
  $PIP install -q torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
  # setuptools==59.5.0 predates python-3.10-era builds; numpy<2 for pandas 1.4 / chumpy / torchgeometry.
  grep -vE "^(setuptools|git\+https://github.com/mattloper/chumpy)" "$D/requirements.txt" > "$reqs"
  $PIP install -q "numpy<2" "setuptools<70" wheel
  $PIP install -q --no-build-isolation mmcv==1.3.9      # setup.py imports pkg_resources (see env_choir)
  $PIP install -q "numpy<2" "setuptools<70" -r "$reqs"
  $PIP install -q --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da"
  $PIP install -q -e "$D"
  $PIP install -q -e "$D/third-party/hamer[all]"
  $PIP install -q -e "$D/third-party/hamer/third-party/ViTPose"
  # Imported by run_opt.py / HMP / HaMeR but missing from every requirements list (hamer comments out lightning).
  # Dyn-HaMR's bundled human_body_prior, not PyPI's 0.8.5; plyfile>=1.1 would pull numpy 2.
  $PIP install -q --no-deps -e "$D/human_body_prior"
  $PIP install -q "numpy<2" "plyfile<1.1" OpenEXR loguru dotmap omegaconf pytorch-lightning==2.5.1
  # DROID-SLAM is only used by Dyn-HaMR's legacy SLAM path (preproc/run_slam.py); CHOIR uses VIPE.
}

env_vipe() {
  local V="$C/stage1_preprocess/Dyn_HaMR_new/third-party/vipe"
  E="$CONDA_ENVS_PATH/vipe"
  [ -x "$E/bin/python" ] || "$CONDA" env create -q -p "$E" -f "$V/envs/base.yml"
  PIP="$E/bin/python -m pip"
  export CUDA_HOME="$E" PATH="$E/bin:$PATH"
  export CPATH="$E/targets/x86_64-linux/include${CPATH:+:$CPATH}"   # conda CUDA headers (cuda.h, ...)
  export LIBRARY_PATH="$E/targets/x86_64-linux/lib:$E/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  $PIP install -q -r "$V/envs/requirements.txt"
  (cd "$V" && $PIP install -q --no-build-isolation -e .)
  # VIPE fetches torch.hub checkpoints from hf-mirror.com (308 redirect, not followed): rewrite to huggingface.co.
  local sp; sp="$("$E/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
  cp "$PROJECT_ROOT/scripts/choir/vipe_compat.py" "$sp/vipe_compat.py"
  echo "import vipe_compat" > "$sp/vipe_compat.pth"
}

check() {
  (cd "$C" && "$CONDA_ENVS_PATH/choir/bin/python" tools/check_assets.py 2>&1 | tail -5) || true
  for spec in "choir:torch pytorch3d kaolin flash_attn torch_scatter mesh_intersection manotorch moge spconv.pytorch gsplat sapien" \
              "dynhamr:torch torch_scatter hamer mmpose ultralytics smplx" "vipe:torch vipe"; do
    name="${spec%%:*}"; mods="${spec#*:}"; E="$CONDA_ENVS_PATH/$name"
    [ -x "$E/bin/python" ] || { echo "$name: not built"; continue; }
    LIDRA_SKIP_INIT=true "$E/bin/python" - $mods <<'EOF' 2>&1 | tail -3 | sed "s/^/$name: /"
import importlib, sys, torch
bad = []
for m in sys.argv[1:]:
    try: importlib.import_module(m)
    except Exception as e: bad.append(f"{m}: {type(e).__name__}: {e}"[:200])
ok = torch.cuda.is_available() and float((torch.ones(1, device="cuda") * 2).item()) == 2.0
cap = "sm_%d%d" % torch.cuda.get_device_capability() if torch.cuda.is_available() else "-"
print(f"torch {torch.__version__} cuda_ok={ok} ({cap}) ->", "OK" if not bad else "FAIL\n  " + "\n  ".join(bad))
EOF
  done
}

case "${1:-}" in
  workdir) workdir ;;
  assets) assets ;;
  bmc) bmc ;;
  env) "env_${2:?env: choir|dynhamr|vipe}" ;;
  check) check ;;
  *) sed -n '2,13p' "$0"; exit 1 ;;
esac
