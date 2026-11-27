"""FDR's Sharpa hand with the real robot's wrist hardware behind the palm, as collision geometry: the arm's last link,
the camera mount plate and the D405 (scripts/embodiment/build_model.py's wrist_stub/, in the palm frame). Without it,
the floating palm is retargeted 3-8.5 cm above the table, where that hardware does not fit (scripts/embodiment/
README.md), so the data cannot be converted to the arm.

  python scripts/mjwarp/hand_variant.py [--arm vega_1p]          (any env with mujoco + pyyaml)

Writes a robot directory next to FDR's sharpa one, $FDR/assets/robot/sharpa_<arm>/{right.xml, left.xml, bimanual.xml,
meshes/} (meshes/: links to sharpa's plus the stub's STLs), so the kinematic retargeting runs with robot=sharpa_<arm>,
and the training robot config scripts/mjwarp/config/robot_sharpa_<arm>_bimanual.yaml (FDR's sharpa_bimanual.yaml with
that model and the three geoms added to its collision list), linked into FDR as
config/envs/robot/sharpa_<arm>_bimanual.yaml (select with envs/robot@robot=sharpa_<arm>_bimanual).

The geoms are massless (density 0): the floating wrist is position controlled, and the arm's mass is not the hand's.
Their names start with collision_hand_right_ (the retargeter pairs every such geom with the table disks and the
object) and contain no digit (its fingertip self-collision rule picks names with "0" or "1").
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import mujoco
import yaml

HERE = Path(__file__).resolve().parent
STORAGE = Path(os.environ.get("PROJECT_STORAGE_ROOT", "/data/sky/projects/CounterfactualHOI"))
FDR = Path(os.environ.get("FDR", STORAGE / "build" / "FlashDexRetarget"))
PALM = {"right": "R_forearm_rot_y_link"}
NAMES = ("collision_hand_right_wrist_armlink", "collision_hand_right_wrist_mount", "collision_hand_right_wrist_camera")


def add_stub(spec, stub, mesh_prefix="wrist/"):
    body = next(b for b in spec.bodies if b.name == PALM["right"])
    spec.add_mesh(name="wrist_armlink", file=mesh_prefix + stub["arm_last_link_hull"]["file"])
    spec.add_mesh(name="wrist_mount", file=mesh_prefix + stub["d405_mount"]["file"])
    rgba = [0.35, 0.35, 0.38, 1.0]
    body.add_geom(name=NAMES[0], type=mujoco.mjtGeom.mjGEOM_MESH, meshname="wrist_armlink", density=0, rgba=rgba)
    body.add_geom(name=NAMES[1], type=mujoco.mjtGeom.mjGEOM_MESH, meshname="wrist_mount", density=0, rgba=rgba)
    box = stub["d405_housing_box"]
    body.add_geom(name=NAMES[2], type=mujoco.mjtGeom.mjGEOM_BOX, pos=box["centre"], quat=box["quat_wxyz"],
                  size=[s / 2 for s in box["size"]], density=0, rgba=[0.15, 0.15, 0.17, 1.0])
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="vega_1p")
    a = ap.parse_args()
    stub_dir = STORAGE / "build" / "embodiment" / f"{a.arm}_sharpa_r" / "wrist_stub"
    stub = json.load(open(stub_dir / "stub.json"))
    src, dst = FDR / "assets" / "robot" / "sharpa", FDR / "assets" / "robot" / f"sharpa_{a.arm}"
    (dst / "meshes" / "wrist").mkdir(parents=True, exist_ok=True)
    for sub in ("left", "right"):
        link = dst / "meshes" / sub
        if not link.exists():
            link.symlink_to(src / "meshes" / sub)
    for f in (stub["arm_last_link_hull"]["file"], stub["d405_mount"]["file"]):
        shutil.copy(stub_dir / f, dst / "meshes" / "wrist" / f)
    for name in ("right.xml", "bimanual.xml", "left.xml"):
        spec = mujoco.MjSpec.from_file(str(src / name))
        spec.meshdir = str(dst / "meshes") + "/"          # absolute while compiling here, relative in the written file
        if any(b.name == PALM["right"] for b in spec.bodies):
            add_stub(spec, stub)
        xml = spec.to_xml().replace(f'meshdir="{dst / "meshes"}/"', 'meshdir="meshes/"')
        (dst / name).write_text(xml)
        m = mujoco.MjSpec.from_file(str(dst / name)).compile()          # it loads from its new place
        print(f"[variant] {dst / name}: {m.ngeom} geoms, {m.nmesh} meshes, nbody {m.nbody}")
    # training robot config
    rc = yaml.safe_load(open(FDR / "config" / "envs" / "robot" / "sharpa_bimanual.yaml"))
    rc_body = rc["robot"] if "robot" in rc else rc
    rc_body["xml_path"] = f"assets/robot/sharpa_{a.arm}/bimanual.xml"
    rc_body["collisions"][0]["geom_names_expr"] = list(rc_body["collisions"][0]["geom_names_expr"]) + list(NAMES)
    out = HERE / "config" / f"robot_sharpa_{a.arm}_bimanual.yaml"
    head = (f"# @package _global_\n# Written by scripts/mjwarp/hand_variant.py --arm {a.arm}: FDR's sharpa_bimanual.yaml with the "
            f"{a.arm} wrist hardware\n# (arm link, mount, D405) as collision geoms on the right palm. Do not edit by hand.\n")
    out.write_text(head + yaml.safe_dump(rc, sort_keys=False))
    link = FDR / "config" / "envs" / "robot" / f"sharpa_{a.arm}_bimanual.yaml"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(out)
    print(f"[variant] {out} (linked as {link})")


if __name__ == "__main__":
    main()
