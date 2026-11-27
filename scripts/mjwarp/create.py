"""Build FlashDexRetarget's train and eval envs on MuJoCo Warp (mjlab 1.3.0), the MuJoCo twin of FDR's
simulator/isaacsim/create.py: the same manager builders (envs.build_*), FDR's MuJoCo scene builder
(envs.scene.build_scene: robot MJCF, object entities from the packed motion file, support disks, sensors) and
mjlab's Simulation instead of Isaac Sim.

  from create import compose, create_envs_mjwarp
  cfg = compose(["envs/robot@robot=sharpa_bimanual", "commands.motion_file=...", "num_envs=64"])
  train_env, eval_env = create_envs_mjwarp(cfg)

Runs in $PROJECT_STORAGE_ROOT/venvs/fdr-mjwarp (scripts/mjwarp/setup.sh). Isaac Sim is not imported.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FDR = Path(os.environ.get("FDR", "/data/sky/projects/CounterfactualHOI/build/FlashDexRetarget"))

import mjlab  # noqa: E402,F401  (the real package first; envs puts FDR's vendored copy on sys.path)
import mjlab.envs  # noqa: E402,F401
import mjlab.managers  # noqa: E402,F401
import mjlab.scene  # noqa: E402,F401
import mjlab.sim  # noqa: E402,F401

sys.path.insert(0, str(FDR / "src"))
sys.path.insert(1, str(HERE))


def compose(overrides: list[str]):
    """FDR's hydra config with the MuJoCo Warp simulator group (config/envs/simulator/mjwarp.yaml, ours). FDR's
    configs use paths relative to its checkout (assets/robot/...), so this also changes into it."""
    import hydra

    os.chdir(FDR)
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", lambda s: eval(s), replace=True)
    with hydra.initialize_config_dir(version_base=None, config_dir=str(FDR / "config")):
        cfg = hydra.compose(config_name="flashdexretarget", overrides=["envs/simulator@simulator=mjwarp", *overrides])
    OmegaConf.resolve(cfg)
    return cfg


def _sim_cfg(cfg):
    from mjlab.sim import MujocoCfg, SimulationCfg

    m = cfg.sim.mujoco
    return SimulationCfg(
        nconmax=m.get("nconmax", None), njmax=m.get("njmax", None),
        mujoco=MujocoCfg(timestep=float(cfg.sim.dt), integrator=str(m.integrator), cone=str(m.cone),
                         impratio=float(m.impratio), solver=str(m.solver), iterations=int(m.iterations),
                         ls_iterations=int(m.ls_iterations), tolerance=float(m.tolerance)),
    )


def _manager_cfgs(cfg) -> dict:
    from envs import (build_actions, build_commands, build_curriculum, build_events, build_observations,
                      build_rewards, build_terminations)

    entity, command = cfg.robot.entity_name, cfg.commands.name
    return dict(
        observations=build_observations(cfg.obs, command_name=command, entity_name=entity),
        rewards=build_rewards(cfg.rewards, command_name=command, entity_name=entity),
        commands=build_commands(cfg.commands),
        actions=build_actions(cfg.actions),
        terminations=build_terminations(cfg.terminations, command_name=command, entity_name=entity),
        curriculum=build_curriculum(cfg.curriculum, command_name=command, entity_name=entity),
        events=build_events(cfg.events),
    )


def make_env(cfg, num_envs: int, device: str = "cuda:0", extra_events: dict | None = None, trials: str | None = None):
    """One MjwarpTrackingEnv with num_envs worlds. trials: a DATASET/trials.json (scripts/mjwarp/trials.py) turns on
    the domain randomization of scripts/mjwarp/randomize.py: the stored object trials, and per episode the robot's
    PD gains, joint damping / armature / friction, hand friction, action delay and action noise."""
    from mjlab.envs import ManagerBasedRlEnvCfg

    from env import MjwarpTrackingEnv
    from envs.scene import build_scene

    cfg.scene.num_envs = int(num_envs)
    scene = build_scene(cfg)          # first: object discovery writes commands.object.{entity_names, mesh_paths, ...}
    managers = _manager_cfgs(cfg)
    if extra_events:
        managers["events"] = {**managers["events"], **extra_events}
    ranges = None
    if trials:
        import json

        import randomize

        ranges = json.load(open(trials))["ranges"]
        randomize.patch_scene(scene)
        hand = tuple(g for c in cfg.robot.collisions for g in c.geom_names_expr if "_right_" in g)
        managers["events"] = {**managers["events"], **randomize.robot_events(cfg.robot.entity_name, hand, ranges)}
    env_cfg = ManagerBasedRlEnvCfg(
        decimation=int(cfg.get("decimation", 4)), scene=scene, sim=_sim_cfg(cfg),
        episode_length_s=float(cfg.get("episode_length_s", 20.0)), seed=cfg.get("seed", None), **managers)
    env = MjwarpTrackingEnv(env_cfg, device=device)
    if trials:
        env.trial_randomizer = randomize.TrialRandomizer(env, trials, cfg.commands.name)
        lo, hi = ranges["action_delay_steps"]
        env.action_dr = (int(hi), float(ranges["action_noise"]))
    return env


def create_envs_mjwarp(cfg, device: str = "cuda:0", extra_events: dict | None = None, trials: str | None = None):
    """(train_env, eval_env), wrapped for FDR's runner as the Isaac backend's are (RslRlVecEnvWrapper); the eval env
    is a second, smaller env with num_per_motion worlds per clip, the layout FDR's mujoco eval used."""
    from mjlab.rl import RslRlVecEnvWrapper
    from omegaconf import OmegaConf

    train = make_env(cfg, int(cfg.scene.num_envs), device, extra_events, trials)
    ecfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    n_motion = int(train.command_manager.get_term(cfg.commands.name).motion_lib.num_trajectories)
    n_eval = max(1, n_motion * int(cfg.eval.get("num_per_motion", 1)))
    ev = make_env(ecfg, n_eval, device, extra_events)           # evaluation without randomization
    # the eval batching FDR's runner reads (the Isaac backend's SharedEvalEnv sets the same three): all clips in one
    # batch, num_per_motion worlds each
    k = max(1, min(n_motion, ev.num_envs // max(1, int(cfg.eval.get("num_per_motion", 1)))))
    ev.eval_k, ev.eval_num_batches, ev.eval_num_motions = k, (n_motion + k - 1) // k, n_motion
    return RslRlVecEnvWrapper(train), RslRlVecEnvWrapper(ev)


if __name__ == "__main__":
    import argparse
    import time

    import torch

    ap = argparse.ArgumentParser(description="build the env, step it with zero actions, print timings")
    ap.add_argument("--motion", default="/data/sky/projects/CounterfactualHOI/runs/flashdex/data_r10_flat/motion_bimanual.pt")
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--trials", default=None, help="DATASET/trials.json: domain randomization on")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    t0 = time.time()
    cfg = compose(["envs/robot@robot=sharpa_bimanual", f"commands.motion_file={a.motion}", f"num_envs={a.num_envs}",
                   "per_hand_actor_critic=false", *a.overrides])
    env = make_env(cfg, a.num_envs, trials=a.trials)
    print(f"[mjwarp] built {a.num_envs} envs in {time.time() - t0:.1f} s; nq {env.sim.mj_model.nq}, ngeom "
          f"{env.sim.mj_model.ngeom}, nbody {env.sim.mj_model.nbody}", flush=True)
    obs, _ = env.reset()
    print("[mjwarp] obs groups", {k: tuple(v.shape) for k, v in obs.items()}, flush=True)
    act = torch.zeros(a.num_envs, sum(env.action_manager.action_term_dim), device=env.device)
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(a.steps):
        obs, rew, term, trunc, ext = env.step(act)
    torch.cuda.synchronize()
    dt = time.time() - t1
    print(f"[mjwarp] {a.steps} steps: {dt / a.steps * 1000:.1f} ms/step, {a.steps * a.num_envs / dt:.0f} env steps/s; "
          f"reward mean {float(rew.mean()):.3f}; terminated {int(term.sum())}, timed out {int(trunc.sum())}", flush=True)
    if env.trial_randomizer is not None:          # what the randomization wrote: per env clip, trial, object mass
        tr = env.trial_randomizer
        b = torch.tensor(tr.body, device=env.device)[tr.cmd.motion_ids.long()]
        mass = env.sim.model.body_mass[torch.arange(env.num_envs, device=env.device), b]
        g0 = torch.stack([gg[0] for gg in tr.geoms])[tr.cmd.motion_ids.long()]
        fr = env.sim.model.geom_friction[torch.arange(env.num_envs, device=env.device), g0]
        for i in range(min(6, env.num_envs)):
            c = int(tr.cmd.motion_ids[i])
            print(f"[mjwarp] env {i}: {tr.names[c]} trial {int(tr.trial_id[i])} mass {1000 * float(mass[i]):.0f} g, friction "
                  f"{[round(float(x), 4) for x in fr[i]]}, delay {int(env.action_delay[i])}", flush=True)
    if os.environ.get("MJW_PROFILE"):            # time each manager inside env.step (synchronised)
        import collections
        tm = collections.defaultdict(float)

        def timed(obj, name, key):
            f = getattr(obj, name)

            def w(*x, **k):
                torch.cuda.synchronize()
                t = time.time()
                r = f(*x, **k)
                torch.cuda.synchronize()
                tm[key] += time.time() - t
                return r
            setattr(obj, name, w)
        for obj, name, key in ((env.observation_manager, "compute", "observations"), (env.reward_manager, "compute", "rewards"),
                               (env.command_manager, "compute", "commands"), (env.termination_manager, "compute", "terminations"),
                               (env.action_manager, "process_action", "actions"), (env.action_manager, "apply_action", "actions"),
                               (env.scene, "update", "scene.update"), (env.scene, "write_data_to_sim", "scene.write"),
                               (env.sim, "step", "physics"), (env.sim, "forward", "forward"), (env.sim, "sense", "sense"),
                               (env, "_reset_idx", "resets"), (env.metrics_manager, "compute", "metrics")):
            timed(obj, name, key)
        torch.cuda.synchronize()
        t3 = time.time()
        for _ in range(a.steps):
            env.step(act)
        torch.cuda.synchronize()
        tot = time.time() - t3
        print("[mjwarp] per step: " + ", ".join(f"{k} {v / a.steps * 1000:.1f} ms" for k, v in sorted(tm.items(), key=lambda kv: -kv[1]))
              + f" (total {tot / a.steps * 1000:.1f} ms)", flush=True)
        if os.environ.get("MJW_PROFILE") == "terms":
            om = env.observation_manager
            tt = collections.defaultdict(float)
            for g, terms in om._group_obs_term_cfgs.items():
                for cfg_t, nm in zip(terms, om._group_obs_term_names[g]):
                    fn = cfg_t.func
                    torch.cuda.synchronize(); t = time.time()
                    for _ in range(5):
                        fn(env, **cfg_t.params)
                    torch.cuda.synchronize(); tt[nm] += (time.time() - t) / 5
            print("[mjwarp] observation terms: " + ", ".join(f"{k} {v * 1000:.1f}" for k, v in sorted(tt.items(), key=lambda kv: -kv[1])[:12]), flush=True)
    t2 = time.time()                              # physics alone: the decimation loop without the managers
    for _ in range(a.steps):
        for _ in range(env.cfg.decimation):
            env.sim.step()
    torch.cuda.synchronize()
    dp = time.time() - t2
    print(f"[mjwarp] physics alone ({env.cfg.decimation} substeps): {dp / a.steps * 1000:.1f} ms/step "
          f"({100 * dp / dt:.0f}% of a full step); CUDA graphs: {env.sim.use_cuda_graph}", flush=True)
