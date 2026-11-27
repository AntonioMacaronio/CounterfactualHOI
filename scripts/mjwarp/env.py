"""FlashDexRetarget's tracking env on MuJoCo Warp: vanilla mjlab 1.3.0's ManagerBasedRlEnv (the API FDR's env code
was written against; FDR vendors a trimmed copy of it for its Isaac Sim backend) plus the step semantics FDR's Isaac
twin adds (simulator/isaacsim/env.py, "mjlab 1.3.0 + mjlab_ext"):
  - capture_final_obs: with auto_reset, the pre-reset observation of envs that time out is put in
    extras["final_obs"] (FlatObsVecEnv bootstraps truncated episodes from it);
  - ObservationManager.compute_snapshot: an observation pass that bypasses the cache and the history buffers.
Domain randomization hooks (scripts/mjwarp/randomize.py): `trial_randomizer` writes each env's object trial at reset,
after the command has chosen the env's clip; `action_dr` = (max delay in control steps, noise std) delays each env's
action by 0..max steps (drawn per episode) and adds Gaussian noise to it.

Import order matters: `import mjlab` (the real package) must happen before `import envs`, which puts FDR's vendored
copy first on sys.path. Once the real package is in sys.modules, its submodules resolve against it.
"""

from __future__ import annotations

import torch

import mjlab  # noqa: F401  (before envs: the real package, not FDR's vendored subset)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationManager


def _install_compute_snapshot() -> None:
    if hasattr(ObservationManager, "compute_snapshot"):
        return

    def compute_snapshot(self):
        self._obs_buffer = None
        return self.compute(update_history=False)

    ObservationManager.compute_snapshot = compute_snapshot


class MjwarpTrackingEnv(ManagerBasedRlEnv):
    """mjlab's env with FDR's capture_final_obs step (the Isaac twin's step body, on mjlab's scene and sim)."""

    def __init__(self, cfg, device: str, **kwargs) -> None:
        _install_compute_snapshot()
        if not hasattr(cfg, "capture_final_obs"):
            cfg.capture_final_obs = False
        self.trial_randomizer = None
        self.action_dr = None                     # (max_delay_steps, noise_std)
        self._act_hist = None
        super().__init__(cfg, device=device, **kwargs)
        self.action_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _delayed(self, action: torch.Tensor) -> torch.Tensor:
        """The action each env executes: its own action from action_delay[env] steps ago, plus noise."""
        max_d, noise = self.action_dr
        if noise > 0:
            action = action + noise * torch.randn_like(action)
        if max_d <= 0:
            return action
        if self._act_hist is None:
            self._act_hist = torch.zeros(max_d + 1, *action.shape, device=action.device, dtype=action.dtype)
        self._act_hist = torch.roll(self._act_hist, 1, 0)
        self._act_hist[0] = action
        return self._act_hist[self.action_delay, torch.arange(action.shape[0], device=action.device)]

    def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        super()._reset_idx(env_ids)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        if self.trial_randomizer is not None:
            self.trial_randomizer.apply(ids)
        if self.action_dr is not None and self.action_dr[0] > 0:
            self.action_delay[ids] = torch.randint(0, self.action_dr[0] + 1, (len(ids),), device=self.device)
            if self._act_hist is not None:
                self._act_hist[:, ids] = 0.0

    def step(self, action: torch.Tensor):
        if not self.cfg.auto_reset and torch.any(self._manual_reset_pending):
            raise RuntimeError("pending manual resets before step()")
        action = action.to(self.device)
        if self.action_dr is not None:
            action = self._delayed(action)
        self.action_manager.process_action(action)
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(dt=self.physics_dt)
            self.metrics_manager.compute_substep()
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        self.metrics_manager.compute()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.extras.pop("final_obs", None)
        if self.cfg.auto_reset and len(reset_env_ids) > 0:
            if getattr(self.cfg, "capture_final_obs", False) and bool(torch.any(self.reset_time_outs)):
                self.sim.forward()
                self.extras["final_obs"] = self.observation_manager.compute_snapshot()
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
        self.sim.forward()
        self.command_manager.compute(dt=self.step_dt)
        if "step" in self.event_manager.available_modes:
            self.event_manager.apply(mode="step", dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.sim.sense()
        self.obs_buf = self.observation_manager.compute(update_history=True)
        if self.cfg.auto_reset and len(reset_env_ids) > 0:
            self.recorder_manager.record_post_reset(reset_env_ids)
        elif len(reset_env_ids) > 0:
            self._manual_reset_pending[reset_env_ids] = True
        self.recorder_manager.record_post_step()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
