"""FlashDexRetarget training (its train.py) with two additions for our clips.

  bash scripts/flashdex/train.sh MOTION RUN_NAME [hydra overrides ...]       (sources FDR's env.sh; Sharpa)

1. Every evaluation also archives the rollout of every clip, not only the passing ones (FDR's success/ keeps the
   first rollout that passes SR@1.0): <run>/evals/step<N>/ (manifest.json + rollouts/m<id>.npz, FDR's format). The
   viewers (export_viser.py) and the AdaDex-style progress curves (eval_curves.py) read them.
2. Every logged metric is also appended to <run>/metrics.jsonl (wandb runs offline here).

Opt-in, by environment variable (both off by default, so a plain launch is FDR's own behaviour):
  CFHOI_FAST=1   speed patches that do not change what is learned (scripts/flashdex/fast.py): the scene build
                 (only the table disks the clips use) and the training loop
  CFHOI_SGS=1    Success-Guided Sampling of the episode resets (scripts/flashdex/sgs.py; log in <run>/sgs.jsonl)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

FDR = Path(os.environ.get("FDR", "/data/sky/projects/CounterfactualHOI/build/FlashDexRetarget"))
sys.path.insert(0, str(FDR / "src"))

import hydra  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


def patch(R, E):
    import torch
    from evaluation.callbacks.success_export import export_success

    run_dir = {}
    orig_archive = E._archive_success

    def archive_all(raw_eval, eval_cfg, tp_cb, rec_cb, states, fail, fail_obj, scored, success_dir, env_step):
        out = orig_archive(raw_eval, eval_cfg, tp_cb, rec_cb, states, fail, fail_obj, scored, success_dir, env_step)
        try:
            lib = raw_eval.command_manager.get_term(eval_cfg.command_name).motion_lib
            d = Path(success_dir).parent / "evals" / f"step{int(env_step)}"
            run_dir["dir"] = Path(success_dir).parent
            export_success(d, rec_cb, states[id(rec_cb)], ok=scored.clone(),
                           motion_ids=torch.cat([s["motion_ids"] for s in states[id(rec_cb)]]), env_step=env_step,
                           criterion="all clips (cfhoi)", motion_names=E._motion_names(lib),
                           motion_file=getattr(lib, "motion_file", None))
        except Exception as exc:  # noqa: BLE001 - a side artefact must not stop training
            print(f"[cfhoi] eval archive failed: {exc!r}", flush=True)
        return out

    E._archive_success = archive_all
    orig_log = R.log_metrics

    def log_metrics(metrics, step):
        orig_log(metrics, step)
        base = run_dir.get("dir") or getattr(R, "_cfhoi_base", None)
        if base is not None and metrics:
            base.mkdir(parents=True, exist_ok=True)
            with open(base / "metrics.jsonl", "a") as f:
                f.write(json.dumps({"step": int(step), **{k: float(v) for k, v in metrics.items()}}) + "\n")

    R.log_metrics = log_metrics


def patch_record():
    """Two more fields in every archived rollout, for the §1.4 policy's data (antzhang-notes/project.md §1.3, rows 2
    and 4 of "Data this stage has to hand to 1.4"):
      joint_pos_target  (T, n_joints)    the PD target the controller received at that step: FDR's action after the
                                         residual scaling, the wrist's low-pass and the limits. The saved `actions`
                                         are numbers in [-1, 1], not joint angles.
      contact_force     (T, [S,] L, 3)   net force of the object on each of the hand's L collision links (world
                                         frame, zero without contact), per side with an object. The links and their
                                         order are the r_alllink_contact sensor's pattern in the run's config.yaml
                                         (21 on the Sharpa hand: palm, thumb x3, index x4, middle x4, ring x4,
                                         little x5).
      link_obj_dist     (T, S * 6)       the object's signed distance at the palm and the five fingertips (FDR's
                                         robot_links_obj_sdf; metres, negative inside the object).
    scripts/flashdex/done_checks.py reads them. A field that cannot be read at the first step is left out; a later
    failure writes zeros, so the widths stay. With the `contacts` patch of fast.py (CFHOI_FAST=1,contacts) the
    contact sensors are no longer updated, so contact_force is then stale: collect data without that patch."""
    import torch
    from envs.observations.motion_tracking import MotionTrackingObs
    from evaluation.callbacks.rollout_record import RolloutRecord

    start0, step0 = RolloutRecord.on_start, RolloutRecord.on_step
    names = ("joint_pos_target", "contact_force", "link_obj_dist")

    def extras(self, env):
        robot = self.cmd.robot
        out = []
        if self._cfhoi_target:
            out.append(robot.data.joint_pos_target)
        if self._cfhoi_sensors:
            out.append(torch.stack([s.data.force for s in self._cfhoi_sensors], 1).reshape(self.num_envs, -1))
        if self._cfhoi_dist:
            out.append(MotionTrackingObs.robot_links_obj_sdf(env, self.command_name).reshape(self.num_envs, -1))
        return torch.cat(out, -1)

    def on_start(self, env):
        start0(self, env)
        if "contacts" in os.environ.get("CFHOI_FAST", "") and not getattr(RolloutRecord, "_cfhoi_told", False):
            print("[cfhoi] CFHOI_FAST has `contacts`: the archived contact_force is stale in this run", flush=True)
            RolloutRecord._cfhoi_told = True
        self._cfhoi_target, self._cfhoi_sensors, self._cfhoi_dist, self._cfhoi_width = False, [], False, 0
        if not self.enabled:
            return
        try:
            n_j = int(self.cmd.robot.data.joint_pos_target.shape[-1])
            self._cfhoi_target = True
            self._roll_spec.append(("joint_pos_target", n_j, ()))
        except Exception as exc:  # noqa: BLE001
            print(f"[cfhoi] rollouts without joint_pos_target: {exc!r}", flush=True)
        try:
            sensors = [env.scene[f"{side[0]}_alllink_contact"] for side in self.cmd._side_list]
            n_l = int(sensors[0].data.force.shape[1])
            self._cfhoi_sensors = sensors
            tail = (n_l, 3) if len(sensors) == 1 else (len(sensors), n_l, 3)
            self._roll_spec.append(("contact_force", len(sensors) * n_l * 3, tail))
        except Exception as exc:  # noqa: BLE001
            print(f"[cfhoi] rollouts without contact_force: {exc!r}", flush=True)
        try:
            n_d = int(MotionTrackingObs.robot_links_obj_sdf(env, self.command_name).reshape(self.num_envs, -1).shape[-1])
            self._cfhoi_dist = True
            self._roll_spec.append(("link_obj_dist", n_d, ()))
        except Exception as exc:  # noqa: BLE001
            print(f"[cfhoi] rollouts without link_obj_dist: {exc!r}", flush=True)
        self._cfhoi_width = sum(w for name, w, _ in self._roll_spec if name in names)

    def on_step(self, env):
        step0(self, env)
        if not self.enabled or not self._cfhoi_width or not self._frames:
            return
        try:
            x = extras(self, env).detach().to("cpu", torch.float32)
        except Exception as exc:  # noqa: BLE001
            if not getattr(self, "_cfhoi_warned", False):
                print(f"[cfhoi] extra rollout fields failed, writing zeros: {exc!r}", flush=True)
                self._cfhoi_warned = True
            x = torch.zeros(self._frames[-1].shape[0], self._cfhoi_width)
        self._frames[-1] = torch.cat([self._frames[-1], x], -1)

    RolloutRecord.on_start, RolloutRecord.on_step = on_start, on_step


def run(overrides):
    OmegaConf.register_new_resolver("eval", lambda s: eval(s), replace=True)
    with hydra.initialize_config_dir(version_base=None, config_dir=str(FDR / "config")):
        cfg = hydra.compose(config_name="flashdexretarget", overrides=overrides)
    OmegaConf.resolve(cfg)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "1")
    from isaaclab.app import AppLauncher
    AppLauncher({"headless": True})

    from envs import create_envs                  # first: puts FDR's vendored mjlab on sys.path
    import evaluation.evaluate as E
    import runner.off_policy_runner as R
    from common.logger import setup_logger
    from common.utils import configure_torch_backends, set_seed
    from runner import create_runner

    patch(R, E)
    patch_record()
    set_seed(cfg.seed)
    configure_torch_backends()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    fast_on = os.environ.get("CFHOI_FAST", "0") not in ("", "0")
    if fast_on:
        import fast
        fast.prepare(cfg)                         # scene-build patches: before the envs exist
    train_env, eval_env = create_envs(cfg, device=cfg.device)
    setup_logger(cfg.logger, cfg.runner)
    runner = create_runner(cfg.runner, cfg.eval, train_env, eval_env=eval_env, headless=True)
    R._cfhoi_base = runner._save_path_base
    if fast_on:
        fast.install(runner._train_env.unwrapped, runner._train_env.share_obs, cfg.eval.command_name)
    if os.environ.get("CFHOI_SGS", "0") == "1":
        import sgs
        sgs.install(runner._train_env.unwrapped, cfg.eval.command_name).log_path = runner._save_path_base / "sgs.jsonl"
    print(f"[cfhoi] run dir {runner._save_path_base}", flush=True)
    (runner._save_path_base).mkdir(parents=True, exist_ok=True)
    (runner._save_path_base / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    runner.learn(num_learning_iterations=int(cfg.runner.max_iterations), init_at_random_ep_len=True)
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:  # noqa: BLE001
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)                                  # isaacsim 5.1 segfaults in close() (as FDR's train.py)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("overrides", nargs="*")
    run(ap.parse_args().overrides)
