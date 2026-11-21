"""Where the FoundationPose pipeline reads masks and the object mesh from (shared by all its steps).

Masks: RUN/masks_sam3/{object,hand}/<i>.png from sam3_masks.py when present (per-frame SAM3, text prompts), else
CHOIR's stage-1 SAM2 tracks (VD/inputs/masks/{object,hand_<side>}/<i>.png, RGBA cut-outs). Mesh: the textured
SAM3D mesh from sam3d_mesh.sh when present, else CHOIR's untextured glb.
"""

import os

import cv2


def _read(path):
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    return (m[..., 3] if m.ndim == 3 and m.shape[2] == 4 else m if m.ndim == 2 else m.max(-1)) > 127


def object_mask(vd, run, i):
    p = os.path.join(run, "masks_sam3", "object", f"{i}.png")
    return _read(p if os.path.exists(p) else os.path.join(vd, "inputs", "masks", "object", f"{i}.png"))


def hand_mask(vd, run, i, side):
    p = os.path.join(run, "masks_sam3", "hand", f"{i}.png")
    return _read(p if os.path.exists(p) else os.path.join(vd, "inputs", "masks", f"hand_{side}", f"{i}.png"))


def mask_source(run):
    return "sam3" if os.path.isdir(os.path.join(run, "masks_sam3", "object")) else "choir_sam2"


def textured_mesh(run):
    p = os.path.join(run, "sam3d_tex", "masks", "object", "object.obj")
    return p if os.path.exists(p) else None


def tracked_mesh(run):
    """The metric object mesh written by run_fpose.py (OBJ when textured, else PLY)."""
    p = os.path.join(run, "object_mesh.obj")
    return p if os.path.exists(p) else os.path.join(run, "object_mesh.ply")
