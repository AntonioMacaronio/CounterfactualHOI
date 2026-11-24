"""Success-Guided Sampling (SGS; Zhang et al., "A Balanced Data Diet", CoRL 2026, https://sgs-rl.github.io/) for
FlashDexRetarget's episode resets.

  CFHOI_SGS=1 bash scripts/flashdex/train.sh MOTION RUN_NAME ...      (scripts/flashdex/train_cfhoi.py installs it)

FDR resets an env to a clip drawn uniformly among its object's clips and a start frame drawn uniformly over the
clip. SGS keeps a fixed set of task configurations and trains on the ones at the edge of what the policy can do.
Here a configuration is a cell = (clip, start-frame bin of CFHOI_SGS_BIN_S seconds). Each cell has a ring buffer of
its last H episode outcomes: success = the episode that started there ended by a time-out (it reached its clip's end)
and not by a failure termination (object dropped, tracking reward too low, fingertips diverged, ...). An env is
reset to a cell of its own object drawn with probability (paper eq. 1-2 and app. B.2)

    w_i = (p_i + eps)^(kappa t) (1 - p_i + eps)^(kappa (1 - t)),   p(i) ∝ (max(w_i, eps) + eps)^(1 / max(T, 1))

with p_i the cell's success rate, and to a frame drawn uniformly inside the cell. Defaults are the paper's
manipulation settings (H 100, t 0.5, kappa 1, T 2, eps 1e-4): the weight peaks at 50% success, a cell that never
succeeds keeps 0.14x the peak, one at 92% keeps 0.73x. A cell without outcomes yet counts as p = t (peak weight), so
every cell is tried early. The paper uses PPO; nothing here depends on the learner. Evaluation is untouched (it
starts every clip at frame 0).

Environment variables: CFHOI_SGS_BIN_S (0.5), CFHOI_SGS_H (100), CFHOI_SGS_T_SUCCESS (0.5), CFHOI_SGS_KAPPA (1),
CFHOI_SGS_TEMP (2), CFHOI_SGS_EPS (1e-4). Every CFHOI_SGS_LOG_S (120) seconds the per-cell success rates and
sampling probabilities are appended to <run>/sgs.jsonl (set .log_path).
"""

from __future__ import annotations

import json
import math
import os
import time

import torch


class SuccessGuidedSampler:
    def __init__(self, env, command_name: str = "motion", bin_s: float = 0.5, history: int = 100, target: float = 0.5,
                 kappa: float = 1.0, temp: float = 2.0, eps: float = 1e-4, log_s: float = 120.0):
        self.env, self.cmd = env, env.command_manager.get_term(command_name)
        cmd, dev = self.cmd, self.cmd.device
        self.H, self.t, self.kappa, self.temp, self.eps = int(history), float(target), float(kappa), max(float(temp), 1.0), float(eps)
        frames = cmd.motion_lib._motion_num_frames.long()                     # (M,) frames per clip
        M = int(frames.numel())
        bin_f = max(1, int(round(bin_s / float(env.step_dt))))
        # cells: clip m has ceil(0.99 T_m / bin_f) bins over the frames FDR's uniform draw can start at
        last = (0.99 * frames.float()).long().clamp(min=1)                   # start frames are in [0, last)
        n_bins = torch.div(last + bin_f - 1, bin_f, rounding_mode="floor")
        self.cell_motion = torch.repeat_interleave(torch.arange(M, device=dev), n_bins)
        first = torch.cumsum(n_bins, 0) - n_bins                             # (M,) first cell of each clip
        k = torch.arange(int(n_bins.sum()), device=dev) - first[self.cell_motion]
        self.cell_f0 = k * bin_f
        self.cell_f1 = torch.minimum((k + 1) * bin_f, last[self.cell_motion])
        self.first_cell, self.bin_f, C = first, bin_f, int(n_bins.sum())
        # the cells an env may draw: those of its object slot's clips (per-env-object mode), else all
        assigned = getattr(env.scene, "assigned_obj_slot", None)
        if assigned is not None and M > 1:
            traj_slot = cmd.motion_lib.traj_obj_slot[cmd._side_list[0]].long()
            self.env_slot = assigned.long()
            cell_slot = traj_slot[self.cell_motion]
        else:
            self.env_slot = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
            cell_slot = torch.zeros(C, dtype=torch.long, device=dev)
        S = int(max(int(cell_slot.max()), int(self.env_slot.max()))) + 1
        rows = [torch.where(cell_slot == s)[0] for s in range(S)]
        width = max(1, max(int(r.numel()) for r in rows))
        self.slot_cells = torch.zeros(S, width, dtype=torch.long, device=dev)
        self.slot_mask = torch.zeros(S, width, dtype=torch.bool, device=dev)
        for s, r in enumerate(rows):
            self.slot_cells[s, : r.numel()] = r
            self.slot_mask[s, : r.numel()] = True
        self.hist = torch.full((C, self.H), -1, dtype=torch.int8, device=dev)  # -1 = no outcome yet
        self.ptr = torch.zeros(C, dtype=torch.long, device=dev)
        self.env_cell = torch.full((env.num_envs,), -1, dtype=torch.long, device=dev)
        self.n_outcomes = 0
        self._probs = None
        self._next_frames = None
        self.log_path, self.log_s, self._last_log = None, float(log_s), time.time()
        self.names = _clip_names(cmd.motion_lib, M)
        # hooks: the reset draw and the start frame
        self._orig_resample = cmd._resample_command
        self._orig_uniform = cmd._uniform_sampling
        cmd._resample_command = self._resample_command
        cmd._uniform_sampling = self._start_frames
        print(f"[sgs] {C} cells over {M} clips ({bin_f} frames = {bin_f * float(env.step_dt):.2f} s each), {S} object "
              f"slots; H {self.H}, target {self.t}, kappa {self.kappa}, T {self.temp}, eps {self.eps}", flush=True)

    # ── statistics ────────────────────────────────────────────────────────────────────────────
    def success_rate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(p, n): success rate per cell over its recorded outcomes (the target t where it has none), and their count."""
        filled = self.hist >= 0
        n = filled.sum(1)
        p = (self.hist.clamp(min=0).float().sum(1) / n.clamp(min=1)).where(n > 0, torch.full_like(n, self.t, dtype=torch.float))
        return p, n

    def cell_weights(self) -> torch.Tensor:
        p, _ = self.success_rate()
        w = (p + self.eps) ** (self.kappa * self.t) * (1 - p + self.eps) ** (self.kappa * (1 - self.t))
        return (w.clamp(min=self.eps) + self.eps) ** (1.0 / self.temp)

    def probs(self) -> torch.Tensor:
        """(S, width) draw probabilities of each object slot's cells."""
        if self._probs is None:
            w = self.cell_weights()[self.slot_cells] * self.slot_mask
            self._probs = w / w.sum(1, keepdim=True).clamp(min=1e-12)
        return self._probs

    # ── hooks ─────────────────────────────────────────────────────────────────────────────────
    def _record(self, env_ids: torch.Tensor) -> None:
        """Outcomes of the episodes that just ended in env_ids, written to their cells' ring buffers."""
        tm = self.env.termination_manager
        cell = self.env_cell[env_ids]
        ended = tm.terminated[env_ids] | tm.time_outs[env_ids]
        ok = cell >= 0
        ok &= ended
        if not bool(ok.any()):
            return
        cell = cell[ok]
        success = (tm.time_outs[env_ids] & ~tm.terminated[env_ids])[ok]
        # several envs can end in the same cell in one step: give them consecutive ring positions
        order = torch.argsort(cell, stable=True)
        cell, success = cell[order], success[order]
        start = torch.ones_like(cell, dtype=torch.bool)
        start[1:] = cell[1:] != cell[:-1]
        idx = torch.arange(cell.numel(), device=cell.device)
        rank = idx - torch.cummax(torch.where(start, idx, torch.zeros_like(idx)), 0).values
        keep = rank < self.H
        cell, success, rank = cell[keep], success[keep], rank[keep]
        self.hist[cell, (self.ptr[cell] + rank) % self.H] = success.to(torch.int8)
        self.ptr.index_add_(0, cell, torch.ones_like(cell))
        self.ptr %= self.H
        self.n_outcomes += int(cell.numel())
        self._probs = None

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        cmd = self.cmd
        if cmd._eval_mode or cmd.cfg.sampling.mode != "uniform":
            self.env_cell[:] = -1                     # evaluation drives the same envs: their episodes do not count
            return self._orig_resample(env_ids)
        if env_ids.numel() == 0:
            return self._orig_resample(env_ids)
        self._record(env_ids)
        slot = self.env_slot[env_ids]
        pick = torch.multinomial(self.probs()[slot], 1).squeeze(1)
        cell = self.slot_cells[slot, pick]
        self.env_cell[env_ids] = cell
        cmd.motion_ids[env_ids] = self.cell_motion[cell]
        f0, f1 = self.cell_f0[cell], self.cell_f1[cell]
        self._next_frames = torch.minimum(f0 + (torch.rand(cell.numel(), device=cell.device) * (f1 - f0).float()).long(), f1 - 1)
        prev = cmd._eval_mode
        cmd._eval_mode = True                         # FDR's own switch for "keep motion_ids": the draw above stands,
        try:                                          # the rest of the reset (warm start, object seeding) runs as usual
            self._orig_resample(env_ids)
        finally:
            cmd._eval_mode = prev
            self._next_frames = None
        if self.log_path is not None and time.time() - self._last_log > self.log_s:
            self.write_log()

    def _start_frames(self, env_ids: torch.Tensor) -> None:
        if self._next_frames is None:
            return self._orig_uniform(env_ids)
        self.cmd.motion_steps[env_ids] = self._next_frames

    # ── log ───────────────────────────────────────────────────────────────────────────────────
    def write_log(self) -> None:
        self._last_log = time.time()
        p, n = self.success_rate()
        w = self.cell_weights()
        rec = {"time": time.time(), "env_step": int(getattr(self.env, "common_step_counter", 0)) * int(self.env.num_envs),
               "outcomes": self.n_outcomes, "clips": {}}
        for m, name in enumerate(self.names):
            sel = self.cell_motion == m
            pm = w[sel] / w[sel].sum().clamp(min=1e-12)
            rec["clips"][name] = {"success": [round(float(x), 3) for x in p[sel]], "n": [int(x) for x in n[sel]],
                                  "prob": [round(float(x), 4) for x in pm]}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def _clip_names(lib, M: int) -> list[str]:
    try:
        from evaluation.evaluate import _motion_names
        names = [str(n) for n in _motion_names(lib)]
        if len(names) == M:
            return names
    except Exception:  # noqa: BLE001 - names are only for the log
        pass
    return [f"m{i}" for i in range(M)]


def install(env, command_name: str = "motion") -> SuccessGuidedSampler:
    """Install SGS on a FlashDexRetarget env (the raw manager env) with the CFHOI_SGS_* settings."""
    g = lambda k, d: float(os.environ.get(k, d))
    return SuccessGuidedSampler(env, command_name, bin_s=g("CFHOI_SGS_BIN_S", 0.5), history=int(g("CFHOI_SGS_H", 100)),
                                target=g("CFHOI_SGS_T_SUCCESS", 0.5), kappa=g("CFHOI_SGS_KAPPA", 1.0),
                                temp=g("CFHOI_SGS_TEMP", 2.0), eps=g("CFHOI_SGS_EPS", 1e-4), log_s=g("CFHOI_SGS_LOG_S", 120.0))


def _self_test() -> None:
    """The weight curve against the numbers on the paper's project page (weight relative to its peak)."""
    for p, want in ((0.0, 0.14), (9 / 64, 0.83), (27 / 64, 0.99), (51 / 64, 0.90), (57 / 64, 0.79), (59 / 64, 0.73)):
        w = ((p + 1e-4) * (1 - p + 1e-4)) ** 0.25 / ((0.5 + 1e-4) ** 2) ** 0.25
        assert abs(w - want) < 0.011, (p, w, want)
    print("sgs weight curve matches the paper's trace")


if __name__ == "__main__":
    _self_test()
