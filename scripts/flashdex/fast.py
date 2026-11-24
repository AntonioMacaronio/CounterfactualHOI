"""Opt-in speed patches for FlashDexRetarget's training loop, applied at run time (FDR's checkout stays untouched).

  CFHOI_FAST=1 bash scripts/flashdex/train.sh MOTION RUN_NAME ...      (scripts/flashdex/train_cfhoi.py installs them)

Why (scripts/flashdex/profile_step.py, 2048 envs): one training iteration issues ~10,700 GPU kernel launches and
~1,400 GPU synchronizations, so the loop waits on the GPU thousands of times per step and a run alone keeps an H200
25-50% busy. Shares of an iteration: physics 23%, resets 22%, observations 19%, the command's logging metrics 12%,
learner 8%. These patches remove work that does not change what is learned:

  obs       with share_actor_critic_obs the critic observation group has the same terms as the actor's, and FDR
            computes every term twice per step: the critic group reuses the actor's tensor.
  metrics   the command recomputes ~50 tracking / contact metrics on every step, but only evaluation and the
            logging at resets read them: in training they are computed every CFHOI_FAST_METRICS_EVERY (4) steps.
            The logged values are then up to 3 steps old; rewards, terminations and the curriculum do not use them.
  resetlog  every reset reads ~120 logging scalars back from the GPU one by one (.item(): one synchronization
            each). They are gathered into one tensor and read once. Same values.
  entities  a scene reset resets every asset twice (FDR's adapters, then Isaac Lab's scene): once is enough.

CFHOI_FAST selects them: "1" = all, or a comma list (e.g. "obs,resetlog"). CFHOI_FAST_CHECK=N verifies the first N
batched reset logs against FDR's own code run on a copy of the command's accumulators.
"""

from __future__ import annotations

import os

import torch

ALL = ("obs", "metrics", "resetlog", "entities")


def _dedup_obs(env, share_obs: bool) -> str:
    om = env.observation_manager
    names = om._group_obs_term_names
    if not share_obs or "actor" not in names or "critic" not in names:
        return "obs: skipped (actor and critic observations are not shared)"
    def same_term(a, c) -> bool:
        try:
            scale = (a.scale is None and c.scale is None) or bool(torch.equal(torch.as_tensor(a.scale), torch.as_tensor(c.scale)))
            return bool(a.func == c.func and a.params == c.params and a.noise is None and c.noise is None
                        and a.clip == c.clip and scale)
        except Exception:  # noqa: BLE001 - anything unusual: do not share
            return False
    same = names["actor"] == names["critic"] and all(
        same_term(a, c) for a, c in zip(om._group_obs_term_cfgs["actor"], om._group_obs_term_cfgs["critic"]))
    hist = any(om._group_obs_term_delay_buffer.get(g) or om._group_obs_term_history_buffer.get(g) for g in ("actor", "critic"))
    if not same or hist:
        return "obs: skipped (the critic group differs from the actor's, or has delay / history terms)"
    orig_group, orig_compute, shared = om.compute_group, om.compute, {}

    def compute(update_history=False):
        shared.clear()                                     # one actor-group tensor per observation pass
        return orig_compute(update_history)

    def compute_group(group_name, update_history=False):
        if group_name not in ("actor", "critic"):
            return orig_group(group_name, update_history)
        if "obs" not in shared:
            shared["obs"] = orig_group("actor", update_history)
        return shared["obs"]
    om.compute, om.compute_group = compute, compute_group
    return f"obs: the critic group reuses the actor's {len(names['actor'])} terms"


def _decimate_metrics(env, cmd, every: int) -> str:
    orig = cmd._update_metrics

    def update_metrics():
        if cmd._eval_mode or env.common_step_counter % every == 0:
            orig()
    cmd._update_metrics = update_metrics
    return f"metrics: the command's logging metrics are computed every {every} steps in training (every step in evaluation)"


def _batched_command_reset(cmd):
    """MetricsMixin.reset + CommandTerm.reset (FDR / mjlab) with one GPU read-back instead of one per value."""
    side_prefixes = {"right": "r", "left": "l"}

    def reset(env_ids):
        keys, vals = [], []
        for name, value in cmd.metrics.items():
            keys.append(name)
            vals.append(torch.mean(value[env_ids]))
            value[env_ids] = 0.0
        cmd.command_counter[env_ids] = 0
        cmd._resample(env_ids)
        if cmd.has_objects:
            for si, side in enumerate(cmd._side_list):
                p = side_prefixes[side]
                for fi, finger in enumerate(cmd.finger_names):
                    k = f"{p}_{finger}"
                    count_ref, count_contact = cmd._contact_count_ref[k], cmd._contact_count_contact[k]
                    total_ref = count_ref[env_ids].sum().clamp(min=1.0)
                    total_contact = count_contact[env_ids].sum().clamp(min=1.0)
                    keys += [f"contact_ref_dist_{k}", f"contact_found_{k}", f"contact_pen_{k}", f"contact_force_{k}",
                             f"contact_miss_max_{k}"]
                    vals += [cmd._contact_sum[f"contact_ref_dist_{k}"][env_ids].sum() / total_ref,
                             count_contact[env_ids].sum() / total_ref,
                             cmd._contact_sum[f"contact_pen_{k}"][env_ids].sum() / total_contact,
                             cmd._contact_sum[f"contact_force_{k}"][env_ids].sum() / total_contact,
                             cmd.contact_miss_max[env_ids, si, fi].mean()]
                    for name in ("ref_dist", "pen", "force"):
                        cmd._contact_sum[f"contact_{name}_{k}"][env_ids] = 0.0
                    count_ref[env_ids] = 0.0
                    count_contact[env_ids] = 0.0
                    cmd.contact_miss_counter[env_ids, si, fi] = 0.0
                    cmd.contact_miss_max[env_ids, si, fi] = 0.0
        return dict(zip(keys, torch.stack(vals).tolist())) if vals else {}
    return reset


def _check_command_reset(cmd, fast_reset, n_checks: int):
    """fast_reset wrapped so its first n_checks results are compared with FDR's own reset run on a copy of the
    accumulators (same class code, _resample disabled on the copy)."""
    left = [n_checks]

    def reset(env_ids):
        if left[0] <= 0:
            return fast_reset(env_ids)
        left[0] -= 1
        shim = object.__new__(type(cmd))
        shim.__dict__.update(cmd.__dict__)
        shim.metrics = {k: v.clone() for k, v in cmd.metrics.items()}
        shim._contact_sum = {k: v.clone() for k, v in cmd._contact_sum.items()}
        shim._contact_count_ref = {k: v.clone() for k, v in cmd._contact_count_ref.items()}
        shim._contact_count_contact = {k: v.clone() for k, v in cmd._contact_count_contact.items()}
        shim.contact_miss_counter, shim.contact_miss_max = cmd.contact_miss_counter.clone(), cmd.contact_miss_max.clone()
        shim.command_counter = cmd.command_counter.clone()
        shim._resample = lambda ids: None
        want = type(cmd).reset(shim, env_ids)
        got = fast_reset(env_ids)
        bad = [k for k in want if k not in got or abs(want[k] - got[k]) > 1e-5 * max(1.0, abs(want[k]))]
        if bad or set(got) != set(want):
            raise RuntimeError(f"[fast] batched reset log differs from FDR's on {bad[:5]} (keys {len(got)} vs {len(want)})")
        if left[0] == 0:
            print(f"[fast] resetlog: {n_checks} batched reset logs match FDR's own ({len(want)} values each)", flush=True)
        return got
    return reset


def _batch_reset_logs(env, cmd, n_checks: int) -> str:
    from envs.commands.motion_tracking.monitor.metrics import MetricsMixin
    from mjlab.managers.command_manager import CommandTerm
    done = []
    # the command term: only when its reset is the code this file mirrors
    # the two reset methods this file mirrors must be the first two in the command's class hierarchy
    owners = [c for c in type(cmd).__mro__ if "reset" in c.__dict__]
    if owners[:2] == [MetricsMixin, CommandTerm] and "reset" not in cmd.__dict__:
        fast = _batched_command_reset(cmd)
        cmd.reset = _check_command_reset(cmd, fast, n_checks) if n_checks > 0 else fast
        done.append("command")
    else:
        print(f"[fast] resetlog: command reset left as is (reset defined by {[c.__name__ for c in owners[:3]]})", flush=True)
    tm = env.termination_manager
    if not tm._class_term_cfgs:
        keys = list(tm._term_dones.keys())

        def term_reset(env_ids=None):
            ids = slice(None) if env_ids is None else env_ids
            counts = torch.stack([tm._term_dones[k] for k in keys])[:, ids].sum(1).tolist()
            return {"Episode_Termination/" + k: int(c) for k, c in zip(keys, counts)}
        tm.reset = term_reset
        done.append("terminations")
    rm = env.reward_manager
    if not rm._class_term_cfgs:
        rkeys = list(rm._episode_sums.keys())

        def reward_reset(env_ids=None):
            ids = slice(None) if env_ids is None else env_ids
            sums = torch.stack([rm._episode_sums[k] for k in rkeys])       # (terms, B)
            avg = (sums[:, ids].mean(1) / env.max_episode_length_s).tolist()
            for k in rkeys:
                rm._episode_sums[k][ids] = 0.0
            return {"Episode_Reward/" + k: v for k, v in zip(rkeys, avg)}
        rm.reset = reward_reset
        done.append("rewards")
    return "resetlog: one GPU read-back per reset for the " + ", ".join(done) + " logs" if done else "resetlog: skipped"


def _single_entity_reset(env) -> str:
    scene = env.scene
    inner = getattr(scene, "_scene", None)
    ents = getattr(scene, "entities", None)
    if inner is None or not ents:
        return "entities: skipped (not FDR's Isaac scene adapter)"
    lab_assets = {id(a) for d in ("_articulations", "_rigid_objects") for a in getattr(inner, d, {}).values()}

    def covered(ent) -> bool:
        """The adapter's reset only resets an asset that Isaac Lab's scene reset resets as well."""
        asset = getattr(ent, "_art", None)
        asset = getattr(ent, "_obj", None) if asset is None else asset
        gravcomp = getattr(ent, "_gravcomp_full", None) is not None and float(getattr(ent, "_gravcomp_scale", 0.0)) != 0.0
        return asset is not None and id(asset) in lab_assets and not gravcomp
    rest = [ent for ent in ents.values() if not covered(ent)]
    if len(rest) == len(ents):
        return "entities: skipped (no adapter is covered by Isaac Lab's scene reset)"

    def reset(env_ids=None):
        for ent in rest:
            ent.reset(env_ids)
        inner.reset(env_ids)                               # resets every registered asset and sensor once
    scene.reset = reset
    return f"entities: {len(ents) - len(rest)} of {len(ents)} assets are reset once per scene reset, not twice"


def install(env, share_obs: bool, command_name: str = "motion") -> list[str]:
    """Apply the patches named in CFHOI_FAST to a FlashDexRetarget env (the raw manager env); returns what was done."""
    sel = os.environ.get("CFHOI_FAST", "1")
    which = ALL if sel in ("1", "all") else tuple(x.strip() for x in sel.split(",") if x.strip())
    unknown = [x for x in which if x not in ALL]
    if unknown:
        raise ValueError(f"CFHOI_FAST: unknown patches {unknown}; choose from {ALL}")
    cmd = env.command_manager.get_term(command_name)
    notes = []
    if "obs" in which:
        notes.append(_dedup_obs(env, share_obs))
    if "metrics" in which:
        notes.append(_decimate_metrics(env, cmd, max(1, int(os.environ.get("CFHOI_FAST_METRICS_EVERY", "4")))))
    if "resetlog" in which:
        notes.append(_batch_reset_logs(env, cmd, int(os.environ.get("CFHOI_FAST_CHECK", "0"))))
    if "entities" in which:
        notes.append(_single_entity_reset(env))
    for n in notes:
        print(f"[fast] {n}", flush=True)
    return notes
