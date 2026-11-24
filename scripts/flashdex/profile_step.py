"""Where a FlashDexRetarget training iteration spends its time, and what the opt-in patches change.

  cd $FDR && source scripts/env.sh && unset HEADLESS FDR_ROBOT_USD && export FDR CUDA_VISIBLE_DEVICES=<gpu>
  python $CFHOI/scripts/flashdex/profile_step.py --out prof.json [--agent CKPT_DIR] [--iters 150] [--fast] [--sgs 400] \
      <the hydra overrides of scripts/flashdex/train.sh>  [num_envs=4096 runner.agent.buffer_max_length=2000000 ...]

Builds the env and the runner as scripts/flashdex/train_cfhoi.py does (no training run, nothing is saved), fills the
replay buffer until the agent updates, then times the runner's own loop body (collect + updates):
  plain   no instrumentation -> iterations/s and env steps/s
  stages  torch.cuda.synchronize() around each stage of the step (physics, reset, observations, ...), ms / iteration
  kernels torch profiler over 4 iterations: GPU kernel launches, synchronizations, operator calls per iteration
With --fast the same passes run again after scripts/flashdex/fast.py is installed (same process, same GPU load, so
the two are comparable). With --sgs N, scripts/flashdex/sgs.py is installed and N more iterations run; its per-clip
success rates and draw probabilities go to the output. --agent loads a checkpoint so the policy behaves like a
trained one (pass runner.agent.load_reward_normalizer=false when num_envs differs from the checkpoint's run).

Measure on a GPU that no other job uses: two FDR processes on one GPU each run at well under half speed.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FDR = Path(os.environ.get("FDR", "/data/sky/projects/CounterfactualHOI/build/FlashDexRetarget"))
sys.path.insert(0, str(FDR / "src"))
sys.path.insert(0, str(HERE))

import hydra  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

STAGES = [("action.process", "action_manager", "process_action"), ("action.apply", "action_manager", "apply_action"),
          ("scene.write", "scene", "write_data_to_sim"), ("sim.step", "sim", "step"), ("scene.update", "scene", "update"),
          ("termination", "termination_manager", "compute"), ("reward", "reward_manager", "compute"),
          ("reset", None, "_reset_idx"), ("sim.forward", "sim", "forward"), ("command", "command_manager", "compute"),
          ("event", "event_manager", "apply"), ("sim.sense", "sim", "sense"), ("observation", "observation_manager", "compute"),
          ("reset/scene.reset", "scene", "reset"), ("reset/command.reset", "command_manager", "reset"),
          ("reset/termination.reset", "termination_manager", "reset"), ("reset/reward.reset", "reward_manager", "reset")]


class Timers:
    def __init__(self, torch):
        self.torch, self.t, self.on = torch, collections.defaultdict(float), False

    def wrap(self, name, fn):
        def w(*a, **k):
            if not self.on:
                return fn(*a, **k)
            self.torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*a, **k)
            self.torch.cuda.synchronize()
            self.t[name] += time.perf_counter() - t0
            return out
        return w

    def take(self, n):
        out = {k: 1000 * v / n for k, v in self.t.items()}
        self.t.clear()
        return out


def gpu_mem_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return max((int(ln.split(",")[1]) for ln in out.splitlines() if int(ln.split(",")[0]) == os.getpid()), default=None)
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--agent", default=None)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--warm", type=int, default=40, help="iterations with updates before timing (torch.compile)")
    ap.add_argument("--fast", action="store_true", help="time again with scripts/flashdex/fast.py installed")
    ap.add_argument("--sgs", type=int, default=0, help="then run this many iterations with scripts/flashdex/sgs.py")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    t_start = time.time()
    OmegaConf.register_new_resolver("eval", lambda s: eval(s), replace=True)
    with hydra.initialize_config_dir(version_base=None, config_dir=str(FDR / "config")):
        cfg = hydra.compose(config_name="flashdexretarget", overrides=args.overrides)
    OmegaConf.resolve(cfg)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "1")
    from isaaclab.app import AppLauncher
    AppLauncher({"headless": True})
    import torch
    from envs import create_envs
    from common.average_meters import AverageMeter
    from common.logger import setup_logger
    from common.utils import configure_torch_backends, set_seed
    from runner import create_runner

    set_seed(cfg.seed)
    configure_torch_backends()
    train_env, eval_env = create_envs(cfg, device=cfg.device)
    setup_logger(cfg.logger, cfg.runner)
    runner = create_runner(cfg.runner, cfg.eval, train_env, eval_env=eval_env, headless=True)
    if args.agent:
        runner._agent.load(args.agent)
    env, agent, meter = runner._train_env, runner._agent, AverageMeter()
    raw, n_envs = env.unwrapped, runner._n_envs
    res = {"overrides": args.overrides, "num_envs": n_envs, "build_s": time.time() - t_start,
           "updates_per_step": float(cfg.updates_per_interaction_step), "batch": int(cfg.runner.agent.sample_batch_size),
           "obs_dim": int(env.single_observation_space.shape[0]), "action_dim": int(env.single_action_space.shape[0]),
           "gpu_mem_after_build_mb": gpu_mem_mb()}
    print(f"[prof] built {n_envs} envs in {res['build_s']:.0f}s; GPU memory {res['gpu_mem_after_build_mb']} MB", flush=True)
    step = [0]

    def iteration():
        step[0] += 1
        runner._collect(step[0], meter)
        if not agent.ready():
            return False
        budget = runner._updates_per_step
        while budget >= 1:
            meter.update(agent.update())
            budget -= 1
        return True

    while not iteration():
        pass
    for _ in range(args.warm):
        iteration()
    T = Timers(torch)

    def instrument():
        """(Re)wrap the stages: the patches replace some of these attributes."""
        for name, owner, attr in STAGES + [("agent.act", "@agent", "act"), ("agent.observe", "@agent", "observe"),
                                           ("agent.update", "@agent", "update"), ("env.step", None, "step")]:
            obj = agent if owner == "@agent" else raw if owner is None else getattr(raw, owner)
            if not getattr(getattr(obj, attr), "_prof", False):
                w = T.wrap(name, getattr(obj, attr))
                w._prof = True
                setattr(obj, attr, w)

    def measure(tag):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            iteration()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / args.iters
        instrument()
        T.on = True
        t0 = time.perf_counter()
        for _ in range(args.iters):
            iteration()
        torch.cuda.synchronize()
        T.on = False
        out = {"it_per_s": 1 / dt, "env_steps_per_s": n_envs / dt, "ms_per_iter": 1000 * dt,
               "stages_ms": T.take(args.iters), "stages_pass_ms_per_iter": 1000 * (time.perf_counter() - t0) / args.iters,
               "gpu_mem_mb": gpu_mem_mb()}
        try:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(4):
                    iteration()
                torch.cuda.synchronize()
            ev = prof.key_averages()
            cnt = lambda *keys: sum(e.count for e in ev if e.key in keys) / 4
            out["kernels"] = {"gpu_kernel_launches": cnt("cudaLaunchKernel", "cuLaunchKernel"),
                              "gpu_synchronizations": cnt("cudaStreamSynchronize", "cudaDeviceSynchronize"),
                              "memcpys": cnt("cudaMemcpyAsync", "cudaMemcpy"),
                              "operator_calls": sum(e.count for e in ev if e.key.startswith("aten::")) / 4}
        except Exception as exc:  # noqa: BLE001
            out["kernels"] = {"error": repr(exc)}
        res[tag] = out
        print(f"[prof] {tag}: {1 / dt:.2f} it/s = {n_envs / dt:.0f} env steps/s; {out['kernels']}", flush=True)

    measure("base")
    if args.fast:
        import fast
        os.environ.setdefault("CFHOI_FAST_CHECK", "30")
        res["fast_notes"] = fast.install(raw, env.share_obs, cfg.eval.command_name)
        for _ in range(40):
            iteration()                                  # runs the reset-log check
        measure("fast")
        print(f"[prof] fast / base: {res['fast']['it_per_s'] / res['base']['it_per_s']:.2f}x", flush=True)
    if args.sgs:
        import sgs
        sampler = sgs.install(raw, cfg.eval.command_name)
        t0 = time.perf_counter()
        for _ in range(args.sgs):
            iteration()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / args.sgs
        p, n = sampler.success_rate()
        w = sampler.cell_weights()
        cmd = raw.command_manager.get_term(cfg.eval.command_name)
        f0, f1 = sampler.cell_f0[sampler.env_cell.clamp(min=0)], sampler.cell_f1[sampler.env_cell.clamp(min=0)]
        clips = {}
        for m, name in enumerate(sampler.names):
            sel = sampler.cell_motion == m
            clips[name] = {"success": [round(float(x), 2) for x in p[sel]], "n": [int(x) for x in n[sel]],
                           "prob": [round(float(x), 3) for x in (w[sel] / w[sel].sum())]}
        res["sgs"] = {"it_per_s": 1 / dt, "outcomes": sampler.n_outcomes, "cells": int(sampler.cell_motion.numel()),
                      "envs_in_their_clip": bool((cmd.motion_ids == sampler.cell_motion[sampler.env_cell.clamp(min=0)])[sampler.env_cell >= 0].all()),
                      "clips": clips}
        print(f"[prof] sgs: {1 / dt:.2f} it/s, {sampler.n_outcomes} episode outcomes over {res['sgs']['cells']} cells", flush=True)
        for name, c in clips.items():
            print(f"[prof]   {name}: success by start bin {c['success']}\n[prof]   {' ' * len(name)}  draw prob {c['prob']}", flush=True)
    res["total_s"] = time.time() - t_start
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[prof] wrote {args.out}", flush=True)
    sys.stdout.flush()
    os._exit(0)                                          # isaacsim 5.1 segfaults in close() (as FDR's train.py)


if __name__ == "__main__":
    main()
