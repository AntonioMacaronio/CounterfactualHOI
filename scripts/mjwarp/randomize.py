"""Domain randomization of FlashDexRetarget's tracking env on MuJoCo Warp (scripts/mjwarp/env.py): the stored
object trials of DATASET/trials.json (scripts/mjwarp/trials.py) and the per-episode robot randomization.

Object trials, at every reset after the command has picked the env's clip: the env draws one of the clip's valid
trials (uniformly; the drop test of scripts/mjwarp/rest_check.py --fix-trials marks them; trial_id per env is kept
for logging and for §1.4) and writes into its world's model rows of that clip's
object body: body_mass and body_inertia (x the trial's mass factor, from the base mass), body_ipos (+ the CoM
offset), and for the object's collision geoms geom_friction (sliding, torsional, rolling) and geom_solref
(restitution). Then the derived constants are recomputed (mjlab's set_const level).
Robot, every reset, through mjlab's own randomization terms (mjlab.envs.mdp.dr): PD gains, joint damping, armature,
joint friction, the hand's collision-geom friction. Action delay and noise are applied in the env's step.

Scene changes made at build time (patch_scene): object collision geoms get condim 6 (torsional and rolling friction)
and solmix 1000 (their solref decides the contact's); support disks get friction (0.1, 0, 0), so the object-table
contact takes the object's friction (MuJoCo uses the larger of the two geoms' coefficients) while hand-object
contacts take the hand's.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

TRIAL_FIELDS = ("body_mass", "body_inertia", "body_ipos", "geom_friction", "geom_solref")


@requires_model_fields(*TRIAL_FIELDS, recompute=RecomputeLevel.set_const)
def _expand_trial_fields(env, env_ids, **_):
    """No-op startup event: makes the event manager expand the trial fields per world before CUDA-graph capture."""
    return None


def patch_scene(scene_cfg, disk_friction=(0.1, 0.0, 0.0), solmix=1000.0):
    """Object collision geoms: condim 6, large solmix. Support disks: low friction. Mutates scene_cfg's spec_fns."""
    for name, ent in scene_cfg.entities.items():
        if name.startswith("object_"):
            fn = ent.spec_fn

            def obj_spec(fn=fn):
                spec = fn()
                for g in spec.geoms:
                    if g.contype:
                        g.condim, g.solmix = 6, solmix
                return spec
            ent.spec_fn = obj_spec
        elif "disk" in name:
            fn = ent.spec_fn

            def disk_spec(fn=fn):
                spec = fn()
                for g in spec.geoms:
                    if g.contype or g.conaffinity:
                        g.friction = list(disk_friction)
                return spec
            ent.spec_fn = disk_spec
    return scene_cfg


def robot_events(entity_name: str, hand_geoms: tuple[str, ...], r: dict) -> dict[str, EventTermCfg]:
    """Per-episode robot randomization as mjlab reset events."""
    from mjlab.envs.mdp import dr

    rob = SceneEntityCfg(entity_name)
    return {
        "dr_trial_fields": EventTermCfg(func=_expand_trial_fields, mode="startup"),
        "dr_pd_gains": EventTermCfg(func=dr.pd_gains, mode="reset", params={
            "kp_range": tuple(r["kp_scale"]), "kd_range": tuple(r["kd_scale"]), "asset_cfg": rob}),
        "dr_joint_damping": EventTermCfg(func=dr.joint_damping, mode="reset", params={
            "ranges": tuple(r["damping_scale"]), "operation": "scale", "asset_cfg": rob}),
        "dr_joint_armature": EventTermCfg(func=dr.joint_armature, mode="reset", params={
            "ranges": tuple(r["armature_scale"]), "operation": "scale", "asset_cfg": rob}),
        "dr_joint_friction": EventTermCfg(func=dr.joint_friction, mode="reset", params={
            "ranges": tuple(r["frictionloss_scale"]), "operation": "scale", "asset_cfg": rob}),
        "dr_hand_friction": EventTermCfg(func=dr.geom_friction, mode="reset", params={
            "ranges": tuple(r["hand_friction"]), "operation": "abs", "axes": [0],
            "asset_cfg": SceneEntityCfg(entity_name, geom_names=hand_geoms)}),
    }


class TrialRandomizer:
    """Writes each env's object trial into its world at reset (called by MjwarpTrackingEnv._reset_idx)."""

    def __init__(self, env, trials_path: str | Path, command_name: str = "motion", side: str = "right"):
        self.env, self.side = env, side
        t = json.load(open(trials_path))
        self.cmd = env.command_manager.get_term(command_name)
        lib = self.cmd.motion_lib
        names = [f.rsplit("#", 1)[-1] for f in lib.motion_files]
        ents = self.cmd.obj_entity_names(side)
        m = env.sim.model
        dev = env.device
        self.n_trials = int(t["n_trials"])
        # per clip: the object body (global id), its collision geoms, and N trial rows
        self.body, self.geoms, self.mass, self.inertia, self.ipos, self.fric, self.solref = [], [], [], [], [], [], []
        for k, name in enumerate(names):
            task = t["tasks"].get(name)
            if task is None:
                raise KeyError(f"{name} has no trials in {trials_path}")
            slot = int(lib.traj_obj_slot[side][k]) if hasattr(lib, "traj_obj_slot") else k
            ent = env.scene[ents[slot] if len(ents) > 1 else ents[0]]
            b = int(ent.indexing.body_ids[0])
            gids = [int(g.id) for g in ent.indexing.geoms if g.contype]
            m0 = float(env.sim.get_default_field("body_mass")[b])
            I0 = env.sim.get_default_field("body_inertia")[b].clone()
            p0 = env.sim.get_default_field("body_ipos")[b].clone()
            base = task["mass_kg_base"] if task["mass_kg_base"] else m0
            tr = task["trials"]
            mass = torch.tensor([base * x["mass_scale"] for x in tr], device=dev)
            self.body.append(b)
            self.geoms.append(torch.tensor(gids, device=dev, dtype=torch.long))
            self.mass.append(mass)
            self.inertia.append(I0[None] * (mass / m0)[:, None])
            self.ipos.append(p0[None] + torch.tensor([x["com_offset_m"] for x in tr], device=dev))
            self.fric.append(torch.tensor([x["friction"] for x in tr], device=dev))
            self.solref.append(torch.tensor([x["solref"] for x in tr], device=dev))
        # trials the drop test kept (scripts/mjwarp/rest_check.py --fix-trials); all of them when none is marked
        self.valid = []
        for name in names:
            v = [k for k, x in enumerate(t["tasks"][name]["trials"]) if x.get("valid", True)]
            self.valid.append(torch.tensor(v or list(range(self.n_trials)), device=dev))
        self.trial_id = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
        self.names = names

    def apply(self, env_ids: torch.Tensor) -> None:
        if env_ids is None or len(env_ids) == 0:
            return
        m = self.env.sim.model
        env_ids = env_ids.to(self.env.device).long()
        clip = self.cmd.motion_ids[env_ids].long()
        trial = torch.zeros(len(env_ids), dtype=torch.long, device=self.env.device)
        for c in torch.unique(clip).tolist():
            sel = clip == c
            v = self.valid[c]
            trial[sel] = v[torch.randint(0, len(v), (int(sel.sum()),), device=self.env.device)]
            e, k, b, g = env_ids[sel], trial[sel], self.body[c], self.geoms[c]
            m.body_mass[e, b] = self.mass[c][k]
            m.body_inertia[e, b] = self.inertia[c][k]
            m.body_ipos[e, b] = self.ipos[c][k]
            ee, gg = torch.meshgrid(e, g, indexing="ij")
            m.geom_friction[ee, gg] = self.fric[c][k][:, None, :].expand(-1, len(g), -1)
            m.geom_solref[ee, gg] = self.solref[c][k][:, None, :].expand(-1, len(g), -1)
        self.trial_id[env_ids] = trial
        self.env.sim.recompute_constants(RecomputeLevel.set_const)
