"""The §1.4 policy network (antzhang-notes/project.md §1.4): a flow-matching transformer that maps proprioception, a
text command and two depth images to a chunk of 8 steps of 29 joint targets, and the value network for RL finetuning.

  python scripts/policy14/model.py        (any env with torch; prints every tensor's shape and the parameter counts)

Tensors (B = batch):
  obs["proprio"]  (B, 174)          last 3 frames of q and q*, each joint in [-1, 1]
  obs["command"]  (B, 2)            one-hot: pick, place
  obs["wrist"]    (B, 2, 60, 106)   depth in [0, 1] over the valid range (0 where invalid), validity mask
  obs["exo"]      (B, 2, 60, 106)   the same for the exo camera
  action chunk    (B, 8, 29)        absolute joint targets, each joint in [-1, 1]

Observation tokens (once per chunk; ObsTokenizer), width D = 256:
  wrist, exo -> a CNN each (4 conv layers, stride 2: 60x106 -> 30x53 -> 15x27 -> 8x14 -> 4x7, 128 channels) -> 28
               patch tokens each (128 -> 256) + a learned position embedding per patch and a camera embedding
  proprio    -> 3 tokens, one per history frame ([q_t-k | q*_t-k], 58 -> 256) + a frame embedding
  command    -> 1 token (a learned embedding per command)
  O = [proprio x3 | command | wrist x28 | exo x28]   (B, 60, 256)

Flow transformer v(a_tau, tau; O) (once per flow step; FlowTransformer): joint self-attention over one sequence
  X = [O | A],  A = the noised chunk's 8 action tokens (29 -> 256) + a learned step embedding     (B, 68, 256)
  6 blocks, DiT-style adaLN-Zero on the flow time: from t = MLP(sinusoid(tau)) (256) each block computes
  (shift, scale, gate) for its attention and its MLP (6 x 256), applied to every token:
    X <- X + gate_a * Attn(LN(X) * (1 + scale_a) + shift_a)      8 heads of 32
    X <- X + gate_m * MLP(LN(X) * (1 + scale_m) + shift_m)       256 -> 1024 -> 256, GELU
  the gates start at 0 (zero-initialised modulation), so every block starts as the identity.
  The 8 action tokens -> LN (adaLN shift / scale) -> linear 256 -> 29, zero-initialised -> v (B, 8, 29).
  Actions attend to each other and to every image patch; the observation tokens are re-modulated by tau in every
  step, so they are not cached across flow steps (the image CNNs are: they run once per chunk).

Flow (FPO++'s convention): a_tau = tau * a + (1 - tau) * eps, target velocity a - eps. Sampling integrates
da/dtau = v from tau = 0 (eps) to 1 with Euler steps: 10 steps from eps ~ N(0, I) in training, 5 or 10 steps from
eps = 0 at deployment.
"""

import math

import torch
import torch.nn as nn

N_JOINTS, CHUNK, N_CMD = 29, 8, 2
HIST = 3
PROPRIO = HIST * 2 * N_JOINTS              # 174
IMG = (2, 60, 106)                         # channels (depth, mask), height, width
ACT = CHUNK * N_JOINTS                     # 232


class DepthEncoder(nn.Module):
    """(B, 2, 60, 106) -> patch features (B, 28, 128): the 4 x 7 map of the last conv layer."""

    def __init__(self, widths=(32, 64, 128, 128)):
        super().__init__()
        layers, c_in = [], IMG[0]
        for i, c in enumerate(widths):
            k = 5 if i == 0 else 3
            layers += [nn.Conv2d(c_in, c, k, stride=2, padding=k // 2), nn.GroupNorm(8, c), nn.GELU()]
            c_in = c
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            f = self.conv(torch.zeros(1, *IMG))
        self.channels, self.grid = f.shape[1], tuple(f.shape[2:])       # 128, (4, 7)

    def forward(self, x):
        return self.conv(x).flatten(2).transpose(1, 2)


class ObsTokenizer(nn.Module):
    """obs dict -> observation tokens O (B, 60, D)."""

    def __init__(self, dim=256):
        super().__init__()
        self.wrist, self.exo = DepthEncoder(), DepthEncoder()
        n_patch = self.wrist.grid[0] * self.wrist.grid[1]
        self.patch = nn.Linear(self.wrist.channels, dim)
        self.patch_pos = nn.Parameter(torch.zeros(1, n_patch, dim))
        self.camera = nn.Parameter(torch.zeros(2, dim))
        self.proprio = nn.Linear(2 * N_JOINTS, dim)
        self.frame = nn.Parameter(torch.zeros(1, HIST, dim))
        self.command = nn.Linear(N_CMD, dim)
        for p in (self.patch_pos, self.camera, self.frame):
            nn.init.normal_(p, std=0.02)

    def forward(self, obs):
        B = obs["proprio"].shape[0]
        # proprio is [q (3 frames) | q* (3 frames)], 29 each: regroup into one [q | q*] per frame
        p = obs["proprio"].view(B, 2, HIST, N_JOINTS).transpose(1, 2).reshape(B, HIST, 2 * N_JOINTS)
        return torch.cat([self.proprio(p) + self.frame,
                          self.command(obs["command"])[:, None],
                          self.patch(self.wrist(obs["wrist"])) + self.patch_pos + self.camera[0],
                          self.patch(self.exo(obs["exo"])) + self.patch_pos + self.camera[1]], 1)


class TimeEmbedding(nn.Module):
    """tau (B,) in [0, 1] -> (B, D)."""

    def __init__(self, dim=256, freq_dim=128):
        super().__init__()
        self.register_buffer("freqs", torch.exp(-math.log(1000.0) * torch.arange(freq_dim // 2) / (freq_dim // 2)))
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, tau):
        ang = 1000.0 * tau[:, None] * self.freqs
        return self.mlp(torch.cat([ang.sin(), ang.cos()], -1))


class Block(nn.Module):
    """Self-attention + MLP over all tokens, both modulated by the flow time (adaLN-Zero)."""

    def __init__(self, dim=256, heads=8, mult=4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim, elementwise_affine=False), nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, mult * dim), nn.GELU(), nn.Linear(mult * dim, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight), nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, t):
        sa, ca, ga, sm, cm, gm = self.ada(t)[:, None].chunk(6, -1)
        h = self.n1(x) * (1 + ca) + sa
        x = x + ga * self.attn(h, h, h, need_weights=False)[0]
        return x + gm * self.mlp(self.n2(x) * (1 + cm) + sm)


class FlowTransformer(nn.Module):
    """(a_tau (B, 8, 29), tau (B,), O (B, 60, D)) -> v (B, 8, 29)."""

    def __init__(self, dim=256, depth=6, heads=8):
        super().__init__()
        self.time = TimeEmbedding(dim)
        self.act_in = nn.Linear(N_JOINTS, dim)
        self.step = nn.Parameter(torch.zeros(1, CHUNK, dim))
        nn.init.normal_(self.step, std=0.02)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, N_JOINTS)
        for m in (self.ada_out[1], self.out):
            nn.init.zeros_(m.weight), nn.init.zeros_(m.bias)          # a fresh policy's velocity is 0

    def forward(self, a_tau, tau, O):
        t = self.time(tau)
        x = torch.cat([O, self.act_in(a_tau) + self.step], 1)
        for b in self.blocks:
            x = b(x, t)
        shift, scale = self.ada_out(t)[:, None].chunk(2, -1)
        return self.out(self.norm(x[:, -CHUNK:]) * (1 + scale) + shift)


class FlowPolicy(nn.Module):
    def __init__(self, dim=256, depth=6, heads=8):
        super().__init__()
        self.encoder, self.velocity = ObsTokenizer(dim), FlowTransformer(dim, depth, heads)

    def cfm_loss(self, O, action, tau=None, eps=None):
        """Conditional flow-matching loss per sample, summed over the chunk's 232 numbers: (B,). O = encoder(obs).
        FPO++'s ratio is exp(loss_old - loss_new) on the same (tau, eps); supervised learning takes the mean."""
        eps = torch.randn_like(action) if eps is None else eps
        tau = torch.rand(action.shape[0], device=action.device) if tau is None else tau
        t = tau[:, None, None]
        v = self.velocity(t * action + (1 - t) * eps, tau, O)
        return ((v - (action - eps)) ** 2).flatten(1).sum(-1)

    @torch.no_grad()
    def sample(self, obs, steps=10, zero_noise=False):
        """An action chunk (B, 8, 29): Euler steps from eps at tau = 0 to tau = 1."""
        O = self.encoder(obs)
        a = torch.zeros(O.shape[0], CHUNK, N_JOINTS, device=O.device)
        if not zero_noise:
            a = torch.randn_like(a)
        for k in range(steps):
            a = a + self.velocity(a, torch.full((O.shape[0],), k / steps, device=O.device), O) / steps
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
    import time

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
    print("  patch tokens", tuple(enc.wrist(obs["wrist"]).shape), "-> projected", tuple(enc.patch(enc.wrist(obs["wrist"])).shape))
    O = enc(obs)
    print("observation tokens O", tuple(O.shape), "= proprio 3 + command 1 + wrist 28 + exo 28")
    a = torch.rand(B, CHUNK, N_JOINTS) * 2 - 1
    tau = torch.rand(B)
    print("tau", tuple(tau.shape), "->", tuple(pi.velocity.time(tau).shape), "| action tokens",
          tuple(pi.velocity.act_in(a).shape), "| sequence", (B, O.shape[1] + CHUNK, O.shape[2]),
          "-> velocity", tuple(pi.velocity(a, tau, O).shape))
    loss = pi.cfm_loss(O, a)
    print("cfm loss per sample", tuple(loss.shape), "| sample (10 steps, noise)", tuple(pi.sample(obs).shape),
          "| sample (5 steps, zero noise)", tuple(pi.sample(obs, 5, True).shape))
    v = Critic(600)
    print("critic (state_dim 600 as an example)", tuple(v(torch.randn(B, 600)).shape))
    print(f"parameters: depth CNN {_count(enc.wrist) / 1e6:.2f} M each, observation tokenizer {_count(enc) / 1e6:.2f} M, "
          f"flow transformer {_count(pi.velocity) / 1e6:.2f} M (block {_count(pi.velocity.blocks[0]) / 1e6:.2f} M), "
          f"policy {_count(pi) / 1e6:.2f} M")
    print("zero-noise sample of a fresh policy: max |a| = %.3f" % pi.sample(obs, 5, True).abs().max())
    opt = torch.optim.Adam(pi.parameters(), 1e-3)
    for step in range(3):
        opt.zero_grad()
        ls = pi.cfm_loss(pi.encoder(obs), a).mean()
        ls.backward()
        g = sum(float(p.grad.norm()) for p in enc.parameters() if p.grad is not None)
        print(f"  update {step}: loss {float(ls):.1f} (mean over the batch of the sum over 232), encoder grad norm {g:.3g}")
        opt.step()
    if torch.cuda.is_available():                 # cost of one chunk at a rollout batch size: encoder once, 10 flow steps
        pi = pi.cuda().eval()
        ob = {k: v.cuda().repeat(256 // B, *[1] * (v.dim() - 1)) for k, v in obs.items()}
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            pi.sample(ob)
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(5):
                pi.sample(ob)
            torch.cuda.synchronize()
        print(f"one chunk for 256 envs (encoder + 10 flow steps, bf16): {(time.time() - t0) / 5 * 1000:.1f} ms")
