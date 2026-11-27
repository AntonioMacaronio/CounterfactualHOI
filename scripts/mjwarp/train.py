"""FlashDexRetarget training on MuJoCo Warp: FDR's FlashSAC runner on the MuJoCo Warp env of scripts/mjwarp/create.py,
with the same additions as scripts/flashdex/train_cfhoi.py (every evaluation's rollouts archived, metrics.jsonl, the
rollout fields §1.4 needs) and, with --trials, the domain randomization of scripts/mjwarp/randomize.py.

  bash scripts/mjwarp/train.sh MOTION RUN_NAME [--trials DATASET/trials.json] [hydra overrides ...]

Isaac Sim is not imported; FDR's Isaac backend (scripts/flashdex/train.sh) stays as it is.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import create  # noqa: E402  (imports the real mjlab before FDR's envs)

sys.path.insert(0, str(HERE.parent / "flashdex"))


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--trials", default=None)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = create.compose(a.overrides)
    import evaluation.evaluate as E
    import runner.off_policy_runner as R
    from common.logger import setup_logger
    from common.utils import configure_torch_backends, set_seed
    from omegaconf import OmegaConf
    from runner import create_runner

    import train_cfhoi                            # our Isaac launcher's patches, backend independent

    train_cfhoi.patch(R, E)
    train_cfhoi.patch_record()
    set_seed(cfg.seed)
    configure_torch_backends()
    train_env, eval_env = create.create_envs_mjwarp(cfg, device=cfg.device, trials=a.trials)
    setup_logger(cfg.logger, cfg.runner)
    runner = create_runner(cfg.runner, cfg.eval, train_env, eval_env=eval_env, headless=True)
    R._cfhoi_base = runner._save_path_base
    if os.environ.get("CFHOI_SGS", "0") == "1":
        import sgs
        sgs.install(runner._train_env.unwrapped, cfg.eval.command_name).log_path = runner._save_path_base / "sgs.jsonl"
    runner._save_path_base.mkdir(parents=True, exist_ok=True)
    (runner._save_path_base / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    if a.trials:
        import shutil
        shutil.copy(a.trials, runner._save_path_base / "trials.json")
    print(f"[cfhoi] run dir {runner._save_path_base} (MuJoCo Warp, {'swap' if os.environ.get('FDR_OBJ_SWAP') else 'multi-object'} object mode{', trials ' + a.trials if a.trials else ''})", flush=True)
    runner.learn(num_learning_iterations=int(cfg.runner.max_iterations), init_at_random_ep_len=True)
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
