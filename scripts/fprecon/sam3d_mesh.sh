#!/usr/bin/env bash
# Textured SAM3D object mesh for one clip from frame 0 and its SAM3 object mask (FoundationPose's model).
#
#   bash scripts/fprecon/sam3d_mesh.sh VID [GPU=0]   -> $PROJECT_STORAGE_ROOT/runs/fprecon/VID/sam3d_tex/masks/object/object.obj
#
# CHOIR's SAM3D fork exports untextured meshes with 95% of the triangles removed (with_texture_baking=False,
# use_vertex_color=False, simplify=0.95); Do-as-I-Do's generate_mesh_sam3d.py bakes a 1024x1024 texture, which
# FoundationPose's refiner/scorer and the viewer both use. Its masks/layout.json has SAM3D's local->scene scale and
# translation (in SAM3D's own depth units), which run_fpose.py turns into meters with the metric depth.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
VID="$1"; GPU="${2:-0}"
VD="$CHOIR_ROOT/output/$VID"; O="$PROJECT_STORAGE_ROOT/runs/fprecon/$VID/sam3d_tex"
mkdir -p "$O/masks"
ln -sfn "$VD/inputs/frames/rgb/0.png" "$O/frame0.png"
cp "$PROJECT_STORAGE_ROOT/runs/fprecon/$VID/masks_sam3/object/0.png" "$O/masks/object.png"
cd "$DAID_ROOT/reconstruction/modules/sam-3d-objects"
CUDA_VISIBLE_DEVICES="$GPU" "$CONDA_ENVS_PATH/sam3d/bin/python" generate_mesh_sam3d.py --image_path "$O/frame0.png" \
  --masks_dir "$O/masks" --with_texture_baking True --use_vertex_color False
ls "$O/masks/object/object.obj" "$O/masks/layout.json" > /dev/null
