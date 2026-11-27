"""The §1.4 policy network (antzhang-notes/project.md §1.4): a flow-matching policy that maps proprioception, a text
command and two depth images to a chunk of 8 steps of 29 joint targets, and the value network for RL finetuning.

  python scripts/policy14/model.py        (any env with torch; prints every tensor's shape and the parameter counts)

Tensors (B = batch):
  obs["proprio"]  (B, 174)          last 3 frames of q and q*, each joint in [-1, 1]
  obs["command"]  (B, 2)            one-hot: pick, place
  obs["wrist"]    (B, 2, 60, 106)   depth in [0, 1] over the valid range (0 where invalid), validity mask
  obs["exo"]      (B, 2, 60, 106)   the same for the exo camera
  action chunk    (B, 8, 29)        absolute joint targets, each joint in [-1, 1]

Observation encoder (once per chunk):
  wrist, exo -> a CNN each (4 conv layers, stride 2: 60x106 -> 30x53 -> 15x27 -> 8x14 -> 4x7, then 3584 -> 64)
  proprio -> MLP -> 256;  command -> linear -> 32
  [256 | 32 | 64 | 64] = 416 -> MLP -> c (B, 512)

Velocity network v(a_tau, tau; c) (once per flow step):
  a_tau (B, 8, 29) -> flatten 232 -> linear -> 512
  tau (B,) -> sinusoidal 64 -> MLP -> 128;  cond = [c | tau] (B, 640)
  4 residual blocks: x <- x + W2 GELU(W1 FiLM(LN(x), cond)), 512 -> 1024 -> 512; FiLM = LN(x) * (1 + gamma) + beta with
  [gamma | beta] = linear(cond)
  LN -> linear -> 232 -> (B, 8, 29)

Flow (FPO++'s convention): a_tau = tau * a + (1 - tau) * eps, target velocity a - eps. Sampling integrates
da/dtau = v from tau = 0 (eps) to 1 with Euler steps: 10 steps from eps ~ N(0, I) in training, 5 or 10 steps from
eps = 0 at deployment.
"""

import math

import torch
import torch.nn as nn

N_JOINTS, CHUNK, N_CMD = 29, 8, 2
PROPRIO = 3 * 2 * N_JOINTS                 # 174
IMG = (2, 60, 106)                         # channels (depth, mask), height, width
ACT = CHUNK * N_JOINTS                     # 232


class DepthEncoder(nn.Module):
    """(B, 2, 60, 106) -> (B, 64)."""

    def __init__(self, out_dim=64, widths=(32, 64, 128, 128)):
        super().__init__()
        layers, c_in = [], IMG[0]
        for i, c in enumerate(widths):
            k = 5 if i == 0 else 3
            layers += [nn.Conv2d(c_in, c, k, stride=2, padding=k // 2), nn.GroupNorm(8, c), nn.GELU()]
            c_in = c
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            self.flat = self.conv(torch.zeros(1, *IMG)).flatten(1).shape[1]       # 128 * 4 * 7 = 3584
        self.fc = nn.Linear(self.flat, out_dim)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


class ObsEncoder(nn.Module):
    """obs dict -> c (B, 512)."""

    def __init__(self, dim=512, proprio_dim=256, cmd_dim=32, img_dim=64):
        super().__init__()
        self.proprio = nn.Sequential(nn.Linear(PROPRIO, proprio_dim), nn.GELU(), nn.Linear(proprio_dim, proprio_dim))
        self.command = nn.Linear(N_CMD, cmd_dim)
        self.wrist, self.exo = DepthEncoder(img_dim), DepthEncoder(img_dim)
        self.fuse = nn.Sequential(nn.Linear(proprio_dim + cmd_dim + 2 * img_dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, obs):
        return self.fuse(torch.cat([self.proprio(obs["proprio"]), self.command(obs["command"]),
                                    self.wrist(obs["wrist"]), self.exo(obs["exo"])], -1))


class TimeEmbedding(nn.Module):
    """tau (B,) in [0, 1] -> (B, 128)."""

    def __init__(self, freq_dim=64, dim=128):
        super().__init__()
        self.register_buffer("freqs", torch.exp(-math.log(1000.0) * torch.arange(freq_dim // 2) / (freq_dim // 2)))
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, tau):
        ang = 1000.0 * tau[:, None] * self.freqs
        return self.mlp(torch.cat([ang.sin(), ang.cos()], -1))


class ResBlock(nn.Module):
    """x <- x + W2 GELU(W1 FiLM(LN(x), cond)); 512 -> 1024 -> 512."""

    def __init__(self, dim, cond_dim, mult=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.film = nn.Linear(cond_dim, 2 * dim)
        self.w1, self.w2 = nn.Linear(dim, mult * dim), nn.Linear(mult * dim, dim)

    def forward(self, x, cond):
        gamma, beta = self.film(cond).chunk(2, -1)
        return x + self.w2(nn.functional.gelu(self.w1(self.norm(x) * (1 + gamma) + beta)))


class VelocityNet(nn.Module):
    """(a_tau (B, 8, 29), tau (B,), c (B, 512)) -> v (B, 8, 29)."""

    def __init__(self, dim=512, obs_dim=512, time_dim=128, blocks=4):
        super().__init__()
        self.time = TimeEmbedding(dim=time_dim)
        self.inp = nn.Linear(ACT, dim)
        self.blocks = nn.ModuleList(ResBlock(dim, obs_dim + time_dim) for _ in range(blocks))
        self.norm, self.out = nn.LayerNorm(dim), nn.Linear(dim, ACT)
        nn.init.zeros_(self.out.weight), nn.init.zeros_(self.out.bias)             # a fresh policy's velocity is 0

    def forward(self, a_tau, tau, c):
        cond = torch.cat([c, self.time(tau)], -1)
        x = self.inp(a_tau.flatten(1))
        for b in self.blocks:
            x = b(x, cond)
        return self.out(self.norm(x)).view(-1, CHUNK, N_JOINTS)


class FlowPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.velocity = ObsEncoder(), VelocityNet()

    def cfm_loss(self, c, action, tau=None, eps=None):
        """Conditional flow-matching loss per sample, summed over the chunk's 232 numbers: (B,). c = encoder(obs).
        FPO++'s ratio is exp(loss_old - loss_new) on the same (tau, eps); supervised learning takes the mean."""
        eps = torch.randn_like(action) if eps is None else eps
        tau = torch.rand(action.shape[0], device=action.device) if tau is None else tau
        t = tau[:, None, None]
        v = self.velocity(t * action + (1 - t) * eps, tau, c)
        return ((v - (action - eps)) ** 2).flatten(1).sum(-1)

    @torch.no_grad()
    def sample(self, obs, steps=10, zero_noise=False):
        """An action chunk (B, 8, 29): Euler steps from eps at tau = 0 to tau = 1."""
        c = self.encoder(obs)
        a = torch.zeros(c.shape[0], CHUNK, N_JOINTS, device=c.device)
        if not zero_noise:
            a = torch.randn_like(a)
        for k in range(steps):
            a = a + self.velocity(a, torch.full((c.shape[0],), k / steps, device=c.device), c) / steps
        return a


class Critic(nn.Module):
    """Value of the privileged state at the start of a chunk: (B, state_dim) -> (B,). state_dim is set by the
    environment's critic observation (robot, object, contact and task-state terms; project.md §1.4)."""

    def __init__(self, state_dim, hidden=(1024, 512, 256)):
        super().__init__()
        layers, d = [], state_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
            d = h
        self.net = nn.Sequential(*layers, nn.Linear(d, 1))

    def forward(self, state):
        return self.net(state).squeeze(-1)


def _count(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 4
    obs = {"proprio": torch.rand(B, PROPRIO) * 2 - 1, "command": torch.eye(N_CMD)[torch.randint(0, N_CMD, (B,))],
           "wrist": torch.rand(B, *IMG), "exo": torch.rand(B, *IMG)}
    pi = FlowPolicy()
    enc = pi.encoder
    x = obs["wrist"]
    print("wrist / exo image", tuple(x.shape))
    for layer in enc.wrist.conv:
        x = layer(x)
        if isinstance(layer, nn.GELU):
            print("  conv ->", tuple(x.shape))
    print("  flatten", enc.wrist.flat, "-> latent", tuple(enc.wrist(obs["wrist"]).shape))
    print("proprio", tuple(obs["proprio"].shape), "->", tuple(enc.proprio(obs["proprio"]).shape),
          "| command", tuple(obs["command"].shape), "->", tuple(enc.command(obs["command"]).shape))
    c = enc(obs)
    print("observation vector c", tuple(c.shape))
    a = torch.rand(B, CHUNK, N_JOINTS) * 2 - 1
    tau = torch.rand(B)
    print("tau", tuple(tau.shape), "->", tuple(pi.velocity.time(tau).shape), "| action chunk", tuple(a.shape),
          "-> velocity", tuple(pi.velocity(a, tau, c).shape))
    loss = pi.cfm_loss(c, a)
    print("cfm loss per sample", tuple(loss.shape), "| sample (10 steps, noise)", tuple(pi.sample(obs).shape),
          "| sample (5 steps, zero noise)", tuple(pi.sample(obs, 5, True).shape))
    v = Critic(600)
    print("critic (state_dim 600 as an example)", tuple(v(torch.randn(B, 600)).shape))
    print(f"parameters: depth encoder {_count(enc.wrist) / 1e6:.2f} M each, observation encoder {_count(enc) / 1e6:.2f} M, "
          f"velocity net {_count(pi.velocity) / 1e6:.2f} M (block {_count(pi.velocity.blocks[0]) / 1e6:.2f} M), "
          f"policy {_count(pi) / 1e6:.2f} M")
    # a fresh policy's velocity is 0 (zero-initialised output layer), so from zero noise it samples all-zero targets;
    # after one update the output layer is non-zero and the loss's gradient reaches the encoders
    print("zero-noise sample of a fresh policy: max |a| = %.3f" % pi.sample(obs, 5, True).abs().max())
    opt = torch.optim.Adam(pi.parameters(), 1e-3)
    for step in range(3):
        opt.zero_grad()
        ls = pi.cfm_loss(pi.encoder(obs), a).mean()
        ls.backward()
        g = sum(float(p.grad.norm()) for p in enc.parameters() if p.grad is not None)
        print(f"  update {step}: loss {float(ls):.1f} (mean over the batch of the sum over 232), encoder grad norm {g:.3g}")
        opt.step()
