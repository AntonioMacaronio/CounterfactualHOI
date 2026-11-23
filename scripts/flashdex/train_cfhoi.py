"""FlashDexRetarget training (its train.py) with two additions for our clips.

  bash scripts/flashdex/train.sh MOTION RUN_NAME [hydra overrides ...]       (sources FDR's env.sh; Sharpa)

1. Every evaluation also archives the rollout of every clip, not only the passing ones (FDR's success/ keeps the
   first rollout that passes SR@1.0): <run>/evals/step<N>/ (manifest.json + rollouts/m<id>.npz, FDR's format). The
   viewers (export_viser.py) and the AdaDex-style progress curves (eval_curves.py) read them.
2. Every logged metric is also appended to <run>/metrics.jsonl (wandb runs offline here).
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
    set_seed(cfg.seed)
    configure_torch_backends()
    train_env, eval_env = create_envs(cfg, device=cfg.device)
    setup_logger(cfg.logger, cfg.runner)
    runner = create_runner(cfg.runner, cfg.eval, train_env, eval_env=eval_env, headless=True)
    R._cfhoi_base = runner._save_path_base
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
