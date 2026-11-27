"""Opt-in speed patches for FlashDexRetarget's scene build and training loop, applied at run time (FDR's checkout
stays untouched).

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
  contacts  the link contact forces (the contact observation, rewards and metrics) come from one Isaac Lab
            ContactSensor per link: 21 links x 2 hands = 42 reads per step, each with its own GPU round trip. One
            PhysX contact view per hand returns the same net forces in one read (checked against the sensors for
            the first CFHOI_FAST_CHECK steps). The sensors, which nothing else reads, are then no longer updated
            (4 times per step each) or reset.

And three for the scene build (prepare(), before the envs are created). With one object per env FDR cannot build
one env and let PhysX replicate it, so the build is a Python loop over the envs (a 512-env build under cProfile:
356 s, of which 246 s spawn support disks and 65 s spawn the objects):

  disks     FDR spawns count x radii = 16 table "support disk" cylinders in every env, one prim at a time (~30 ms
            each), whatever the dataset needs; our clips use one. Only the disk slots and radius buckets the motion
            file uses are built (each clip keeps exactly the disk it had: same radius, same pose). Resets get
            cheaper too (16 of the 18 assets a scene reset touches were disks).
  usdcheck  Isaac Lab checks that every spawned USD file exists with an asynchronous server query that sleeps at
            least 0.1 s, once per env's object; for a local path it becomes os.path.isfile.
  fabric    Isaac Sim's cloner switches the Fabric USD notice handler off while it copies prims and on again after
            each of its three calls; switching it on re-synchronises Fabric with the stage, and that grows faster
            than the env count (5.6 s at 512 envs, 72 s at 2048). It stays off from the first clone until physics
            is about to be loaded (SimulationContext.reset) and is then switched on once (4.4 s at 2048). The
            per-env spawns in between are cheaper as well. Same USD stage, same physics state.

CFHOI_FAST selects them: "1" = all but the ones in EXTRA, or a comma list (e.g. "obs,resetlog", "1,contacts").
CFHOI_FAST_CHECK=N verifies the first N batched reset logs against FDR's own code run on a copy of the command's
accumulators, and the first N steps of the one-view contact forces against the sensors.
"""

from __future__ import annotations

import os

import torch

ALL = ("obs", "metrics", "resetlog", "entities", "disks", "usdcheck", "fabric")
EXTRA = ("contacts",)  # selectable by name; not part of "1" until a full run has used them
_pending = []          # deferred work of the build patches that install() must not leave undone


def _selected() -> tuple[str, ...]:
    sel = os.environ.get("CFHOI_FAST", "1")
    which = []
    for x in (x.strip() for x in sel.split(",")):
        which += list(ALL) if x in ("1", "all") else [x] if x else []
    unknown = [x for x in which if x not in ALL + EXTRA]
    if unknown:
        raise ValueError(f"CFHOI_FAST: unknown patches {unknown}; choose from {ALL + EXTRA}")
    return tuple(dict.fromkeys(which))


def _trim_support_disks(cfg) -> str:
    """commands.object.support_disks.count / radii cut down to what the motion file uses. A disk goes to the first
    radius >= its radius + margin (the largest if none), as in FDR's _write_support_disks; dropping unused radii
    keeps that choice for every disk."""
    from omegaconf import OmegaConf
    sd = cfg.commands.object.support_disks
    if not bool(sd.enable):
        return "disks: skipped (support disks are off)"
    pack = torch.load(str(cfg.commands.motion_file), weights_only=False, map_location="cpu")
    disks = pack.get("support_disks") if isinstance(pack, dict) else None
    if disks is None or disks.numel() == 0 or not bool((disks[..., 3] > 0).any()):
        return "disks: skipped (the motion file has no support disks)"
    radii = [float(r) for r in sd.radii]
    if radii != sorted(radii):
        return "disks: skipped (radii are not ascending)"
    r = disks[..., 3].float()                                              # (M, K)
    used = r > 0
    slots = int(used.any(0).nonzero().max()) + 1                           # disk slots in use: 0 .. slots - 1
    r_eff = r[used] + float(sd.radius_margin)
    bucket = (torch.tensor(radii)[None, :] < r_eff[:, None] - 1e-6).sum(-1).clamp(max=len(radii) - 1)
    keep = [radii[j] for j in sorted(set(bucket.tolist()))]
    before = int(sd.count) * len(radii)
    OmegaConf.update(cfg, "commands.object.support_disks.count", slots)
    OmegaConf.update(cfg, "commands.object.support_disks.radii", keep)
    return f"disks: {slots * len(keep)} support disk(s) per env instead of {before} (slots {slots}, radii {keep})"


def _local_usd_check() -> str:
    import importlib
    ff = importlib.import_module("isaaclab.sim.spawners.from_files.from_files")   # the module, not the package's re-exports
    orig = getattr(ff, "check_usd_path_with_timeout", None)
    if orig is None:
        return "usdcheck: skipped (this Isaac Lab has no check_usd_path_with_timeout in its USD spawner)"

    def check(usd_path, *a, **k):
        if "://" not in str(usd_path):
            return os.path.isfile(usd_path)
        return orig(usd_path, *a, **k)
    ff.check_usd_path_with_timeout = check
    return "usdcheck: local USD files are checked with os.path.isfile (no 0.1 s server query per spawn)"


def _defer_fabric_resync() -> str:
    import importlib
    try:
        SM = importlib.import_module("isaacsim.core.simulation_manager").SimulationManager
        SC = importlib.import_module("isaaclab.sim.simulation_context").SimulationContext
        orig_enable, orig_reset = SM.enable_fabric_usd_notice_handler, SC.reset
    except Exception as exc:  # noqa: BLE001 - another Isaac Sim / Isaac Lab layout: leave the build as it is
        return f"fabric: skipped ({exc!r})"
    owed, state = [], {"defer": True}

    def enable(stage_id, flag):
        if flag and state["defer"]:
            if stage_id not in owed:
                owed.append(stage_id)                  # switched on again in flush()
            return None
        return orig_enable(stage_id, flag)

    def flush():
        if not state["defer"]:
            return
        state["defer"] = False
        SM.enable_fabric_usd_notice_handler, SC.reset = orig_enable, orig_reset
        for stage_id in owed:
            orig_enable(stage_id, True)
        if owed:
            print(f"[fast] fabric: notice handler switched back on once for {len(owed)} stage(s)", flush=True)
        owed.clear()

    def reset(self, *a, **k):
        flush()                                        # before physics is loaded from the stage
        return orig_reset(self, *a, **k)
    SM.enable_fabric_usd_notice_handler = staticmethod(enable)
    SC.reset = reset
    _pending.append(flush)
    return "fabric: the cloner's Fabric re-synchronisations are done once, before physics is loaded"


def prepare(cfg) -> list[str]:
    """The build patches named in CFHOI_FAST; call after the Isaac app is up and before the envs are created."""
    which, notes = _selected(), []
    if "disks" in which:
        notes.append(_trim_support_disks(cfg))
    if "usdcheck" in which:
        notes.append(_local_usd_check())
    if "fabric" in which:
        notes.append(_defer_fabric_resync())
    for n in notes:
        print(f"[fast] {n}", flush=True)
    return notes


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


def _noop(*args, **kwargs) -> None:
    return None


def _one_contact_view(env, n_checks: int) -> str:
    banks = []
    for adapter in getattr(env.scene, "sensors", {}).values():
        bank = getattr(adapter, "_bank", None)
        if bank is not None and getattr(bank, "_sensors", None) and all(bank is not b for b in banks):
            banks.append(bank)
    if not banks:
        return "contacts: skipped (no per-link contact sensors)"
    B, done, patched = env.num_envs, [], []

    def silence():
        """Nothing reads the per-link sensors any more: stop updating (4 times per step) and resetting them."""
        for sens, left in patched:
            left[0] = 0
            for x in sens:
                x.update = x.reset = _noop

    for bank in banks:
        sensors = list(bank._sensors)
        try:
            names, parents = [], set()
            for sen in sensors:
                c = sen.cfg
                if sen.num_bodies != 1 or c.filter_prim_paths_expr or c.history_length or c.update_period != 0.0:
                    raise ValueError("a sensor has several bodies, filters, a history or an update period")
                parents.add(c.prim_path.rsplit("/", 1)[0])
                names.append(sen.body_names[0])
            if len(parents) != 1 or len(set(names)) != len(names):
                raise ValueError("the sensors are not distinct links of one asset")
            pattern = f"{parents.pop().replace('.*', '*')}/({'|'.join(names)})"
            view = sensors[0]._physics_sim_view.create_rigid_contact_view(pattern, filter_patterns=[], max_contact_data_count=0)
            n = len(names)
            if view.sensor_count != n * B:
                raise ValueError(f"the view has {view.sensor_count} bodies, expected {n} x {B}")
            got = [path.rsplit("/", 1)[-1] for path in view.sensor_paths[:n]]
            order = None if got == names else torch.tensor([got.index(nm) for nm in names], device=bank.device)
            dt = float(sensors[0]._sim_physics_dt)
        except Exception as exc:  # noqa: BLE001 - keep FDR's sensors for this hand
            print(f"[fast] contacts: {getattr(bank, 'side', '?')} hand left as is ({exc!r})", flush=True)
            continue

        left = [n_checks]

        def force(bank=bank, view=view, n=n, order=order, dt=dt, sensors=sensors, left=left, worst=[0.0, 0.0]):
            store = bank._fresh()
            if "force" not in store:
                f = view.get_net_contact_forces(dt=dt).view(B, n, 3)
                f = (f if order is None else f[:, order]).to(bank.device).clone()
                if left[0] > 0:
                    left[0] -= 1
                    want = torch.stack([x.data.net_forces_w[:, 0] for x in sensors], dim=1).to(bank.device)
                    # the two views sum the same contact impulses in a different order: float32 rounding, relative
                    # to the largest force (1.2e-4 N seen next to forces of thousands of N); a wrong link or env
                    # order would be off by the forces themselves
                    diff, top = float((f - want).abs().max()), float(want.abs().max())
                    worst[0], worst[1] = max(worst[0], diff), max(worst[1], top)
                    if diff > 1e-3 + 1e-5 * top:
                        raise RuntimeError(f"[fast] contacts: the {bank.side} hand's one-view forces differ from the "
                                           f"sensors' by up to {diff:.3e} N (largest force {top:.1f} N)")
                    if left[0] == 0:
                        print(f"[fast] contacts: {bank.side} hand, {n_checks} steps match the {n} sensors (largest "
                              f"difference {worst[0]:.1e} N, largest force {worst[1]:.0f} N); the per-link sensors "
                              f"are no longer updated or reset", flush=True)
                        silence()                          # also ends the check of a hand nothing reads
                store["force"] = f
            return store["force"]
        bank.force = force
        patched.append((sensors, left))
        done.append(f"{getattr(bank, 'side', '?')} ({n} links)")
    if n_checks <= 0:
        silence()
    return ("contacts: one PhysX contact view per hand instead of one sensor per link: " + ", ".join(done)) if done \
        else "contacts: skipped (see above)"


def install(env, share_obs: bool, command_name: str = "motion") -> list[str]:
    """Apply the training-loop patches named in CFHOI_FAST to a FlashDexRetarget env (the raw manager env); returns
    what was done."""
    which = _selected()
    while _pending:                                    # e.g. a build that never reached SimulationContext.reset
        _pending.pop()()
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
    if "contacts" in which:
        notes.append(_one_contact_view(env, int(os.environ.get("CFHOI_FAST_CHECK", "0"))))
    for n in notes:
        print(f"[fast] {n}", flush=True)
    return notes
