"""Draws antzhang-notes/policy_arch.svg, the tensor-flow diagram of scripts/policy14/model.py.

  python3 scripts/policy14/draw_arch.py antzhang-notes/policy_arch.svg      (stdlib only)

The shapes and parameter counts are written in by hand from model.py's printout; redraw after changing the model.
"""
import sys
W, H = 1970, 1560
T, INK, MUTE = "#2f6f66", "#1f3f3a", "#4b6b66"
o = []
def rect(x, y, w, h, fill="#ffffff", stroke=T, sw=2, rx=0, dash=None):
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" rx="{rx}"' + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
def text(x, y, s, size=17, fill=INK, anchor="start", weight="400", mono=False, italic=False):
    fam = "ui-monospace, Menlo, Consolas, monospace" if mono else "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
    o.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}" font-family="{fam}"' + (' font-style="italic"' if italic else "") + f'>{s}</text>')
def arrow(pts, dash=None):
    d = "M" + " L".join(f"{x},{y}" for x, y in pts)
    o.append(f'<path d="{d}" fill="none" stroke="{T}" stroke-width="2.2" marker-end="url(#ah)"' + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
def op(x, y, w, h, label, sub=None, size=17):
    rect(x, y, w, h, rx=9, sw=2.4)
    text(x + w / 2, y + h / 2 + 6, label, size, anchor="middle", weight="700")
    if sub: text(x + w / 2, y + h + 20, sub, 14, MUTE, "middle", mono=True)
def bar(x, y, w, h, shape, fill="#ffffff", stroke=T, name=None, lines=0, above=True):
    rect(x, y, w, h, fill, stroke, 2.2, 2)
    for k in range(1, lines):
        yy = y + k * h / lines
        o.append(f'<line x1="{x}" y1="{yy}" x2="{x + w}" y2="{yy}" stroke="{stroke}" stroke-width="0.7" opacity="0.55"/>')
    text(x + w / 2, y - 9 if above else y + h + 19, shape, 15, MUTE, "middle", mono=True)
    if name: text(x + w / 2, y + h + (20 if above else 38), name, 16, INK, "middle")
def grid(x, y, w, h, cols, rows, fill, stroke, shape, name=None):
    rect(x, y, w, h, fill, stroke, 2.2, 2)
    for k in range(1, cols):
        xx = x + k * w / cols
        o.append(f'<line x1="{xx}" y1="{y}" x2="{xx}" y2="{y + h}" stroke="{stroke}" stroke-width="0.8" opacity="0.6"/>')
    for k in range(1, rows):
        yy = y + k * h / rows
        o.append(f'<line x1="{x}" y1="{yy}" x2="{x + w}" y2="{yy}" stroke="{stroke}" stroke-width="0.5" opacity="0.4"/>')
    text(x + w / 2, y - 9, shape, 15, MUTE, "middle", mono=True)
    if name: text(x + w / 2, y + h + 20, name, 16, INK, "middle")
def trap(x, y, w, h_in, h_out, label, sub):
    c = y
    o.append(f'<path d="M{x},{c - h_in / 2} L{x + w},{c - h_out / 2} L{x + w},{c + h_out / 2} L{x},{c + h_in / 2} Z" fill="#ffffff" stroke="{T}" stroke-width="2.4"/>')
    text(x + w / 2, c + 6, label, 18, anchor="middle", weight="700", italic=True)
    text(x + w / 2, c + max(h_in, h_out) / 2 + 20, sub, 14, MUTE, "middle", mono=True)
def num(x, y, n):
    o.append(f'<circle cx="{x}" cy="{y}" r="13" fill="{T}"/>')
    text(x, y + 6, str(n), 16, "#ffffff", "middle", "700")
def head(x, y, s):
    text(x, y, s, 21, T, weight="700")
def plus(x, y):
    o.append(f'<circle cx="{x}" cy="{y}" r="15" fill="#ffffff" stroke="{T}" stroke-width="2.4"/>')
    text(x, y + 7, "+", 22, T, "middle", "700")

OR, ORS, BL, BLS, YE, YES, GR, GRS = "#fdf1e7", "#c0532c", "#e9eefc", "#3b4fd0", "#fbf6dc", "#a8862a", "#e3f4ee", "#2f6f66"
# ---------------- A: observation encoder
head(20, 38, "Observation encoder, once per action chunk (1.53 M parameters)")
def cam(y, fill, stroke, name):
    grid(20, y, 124, 70, 8, 5, fill, stroke, "2 × 60 × 106", name)
    arrow([(144, y + 35), (196, y + 35)])
    op(198, y + 9, 330, 52, "CNN: 4 × (conv, GroupNorm, GELU)", "32×30×53 → 64×15×27 → 128×8×14 → 128×4×7", 15)
    arrow([(528, y + 35), (580, y + 35)])
    bar(582, y - 5, 26, 80, "3584", fill, stroke, lines=10)
    arrow([(608, y + 35), (654, y + 35)])
    op(656, y + 9, 96, 52, "W", "3584→64")
    arrow([(752, y + 35), (800, y + 35)])
    bar(802, y + 15, 26, 40, "64", fill, stroke, lines=4)
cam(74, OR, ORS, "wrist depth + mask"); num(363, 66, 1)
cam(214, BL, BLS, "exo depth + mask"); num(363, 206, 1)
bar(20, 348, 26, 96, "174", YE, YES, "proprioception", lines=6)
arrow([(46, 396), (196, 396)])
op(198, 370, 330, 52, "MLP: linear, GELU, linear", "174→256→256"); num(363, 362, 2)
arrow([(528, 396), (800, 396)])
bar(802, 356, 26, 80, "256", YE, YES, lines=8)
bar(20, 492, 26, 20, "2", GR, GRS, "command: pick / place")
arrow([(46, 502), (196, 502)])
op(198, 476, 330, 52, "W", "2→32"); num(363, 468, 3)
arrow([(528, 502), (800, 502)])
bar(802, 487, 26, 30, "32", GR, GRS, lines=2)
# concat
cx = 940
text(cx + 13, 148, "416", 15, MUTE, "middle", mono=True)
rect(cx, 158, 26, 100, YE, YES, 2.2, 2); rect(cx, 258, 26, 14, GR, GRS, 2.2, 2); rect(cx, 272, 26, 26, OR, ORS, 2.2, 2); rect(cx, 298, 26, 26, BL, BLS, 2.2, 2)
text(896, 352, "[256 | 32 | 64 | 64]", 15, INK)
for yy, ty in ((109, 285), (249, 311), (396, 208), (502, 265)):
    arrow([(828, yy), (884, yy), (884, ty), (cx - 2, ty)])
num(cx + 13, 120, 4)
arrow([(cx + 26, 241), (1040, 241)])
op(1042, 215, 190, 52, "MLP", "416→512→512")
arrow([(1232, 241), (1290, 241)])
bar(1292, 161, 30, 160, "512", GR, GRS, lines=16)
text(1307, 345, "c", 20, INK, "middle", "700", italic=True)

# ---------------- B: velocity network
head(20, 584, "Velocity network v(a_τ, τ; c), once per flow step (7.09 M parameters)")
grid(20, 640, 124, 76, 8, 6, "#ffffff", T, "8 × 29", "noised chunk a_τ")
arrow([(144, 678), (196, 678)])
op(198, 652, 110, 52, "flatten")
arrow([(308, 678), (356, 678)])
bar(358, 632, 26, 92, "232", lines=8)
arrow([(384, 678), (432, 678)])
op(434, 652, 120, 52, "W_in", "232→512"); num(494, 644, 5)
arrow([(554, 678), (602, 678)])
bar(604, 628, 30, 100, "512", lines=10)
text(619, 752, "x", 20, INK, "middle", "700", italic=True)
# tau and cond
o.append(f'<circle cx="33" cy="838" r="14" fill="#ffffff" stroke="{T}" stroke-width="2.2"/>'); text(33, 844, "τ", 18, INK, "middle", italic=True)
text(33, 876, "flow time", 16, INK, "middle")
arrow([(47, 838), (196, 838)])
op(198, 812, 150, 52, "sin / cos", "→ 64")
arrow([(348, 838), (396, 838)])
op(398, 812, 156, 52, "MLP", "64→128→128")
arrow([(554, 838), (602, 838)])
bar(604, 818, 26, 40, "128", lines=4)
bar(760, 778, 30, 120, "512", GR, GRS, lines=12); text(775, 922, "c", 20, INK, "middle", "700", italic=True)
text(775, 944, "from the encoder", 14, MUTE, "middle")
num(862, 800, 6)
rect(886, 818, 28, 100, GR, GRS, 2.2, 2); rect(886, 918, 28, 26, "#ffffff", T, 2.2, 2)
text(900, 808, "640", 15, MUTE, "middle", mono=True)
text(930, 876, "cond = [c | τ]", 16, INK)
arrow([(790, 838), (884, 838)])
arrow([(630, 838), (700, 838), (700, 966), (850, 966), (850, 931), (884, 931)])
# res block
by = 1000
rect(20, by, 1300, 312, "none", T, 2, 12, "8 6")
text(36, by + 28, "ResBlock × 4 (1.71 M each)", 18, T, weight="700")
cy = by + 196
bar(48, cy - 48, 30, 96, "", lines=10); text(63, cy + 74, "x, 512", 18, INK, "middle", "700", italic=True)
arrow([(78, cy), (120, cy)])
op(122, cy - 26, 70, 52, "LN")
arrow([(192, cy), (232, cy)])
op(234, cy - 26, 214, 52, "FiLM: × (1 + γ) + β"); num(460, cy - 34, 7)
op(246, by + 44, 190, 44, "W_f: cond → [γ | β]", None, 15)
text(446, by + 72, "640→1024", 14, MUTE, mono=True)
arrow([(341, by + 88), (341, cy - 28)])
text(352, by + 104, "γ, β", 14, MUTE, italic=True)
arrow([(448, cy), (488, cy)])
trap(490, cy, 96, 90, 170, "W₁", "512→1024")
arrow([(586, cy), (622, cy)])
op(624, cy - 26, 90, 52, "GELU")
arrow([(714, cy), (750, cy)])
bar(752, cy - 85, 34, 170, "1024", lines=20)
arrow([(786, cy), (822, cy)])
trap(824, cy, 96, 170, 90, "W₂", "1024→512")
arrow([(920, cy), (966, cy)])
plus(983, cy); num(1012, cy - 30, 8)
arrow([(63, cy - 48), (63, by + 132), (983, by + 132), (983, cy - 17)])
text(640, by + 126, "skip: x", 14, MUTE, "middle", italic=True)
arrow([(998, cy), (1046, cy)])
bar(1048, cy - 48, 30, 96, "", lines=10); text(1063, cy + 74, "x, 512", 18, INK, "middle", "700", italic=True)
text(1094, cy + 6, "→ next block", 16, MUTE)
# cond into block, x into block
arrow([(900, 946), (900, 972), (341, 972), (341, by + 42)])
arrow([(634, 678), (1290, 678), (1290, 988), (40, 988), (40, cy), (46, cy)])
# output
oy = 1420
bar(48, oy - 60, 30, 120, "512", lines=12); text(63, oy + 84, "x after block 4", 16, INK, "middle")
arrow([(78, oy), (150, oy)])
op(152, oy - 26, 70, 52, "LN")
arrow([(222, oy), (262, oy)])
op(264, oy - 26, 130, 52, "W_out", "512→232, zero init"); num(329, oy - 40, 9)
arrow([(394, oy), (440, oy)])
bar(442, oy - 46, 26, 92, "232", lines=8)
arrow([(468, oy), (512, oy)])
op(514, oy - 26, 110, 52, "reshape")
arrow([(624, oy), (672, oy)])
grid(674, oy - 38, 124, 76, 8, 6, "#ffffff", T, "8 × 29", "velocity v")
# sampling strip
sx = 900
rect(sx - 20, 1350, 440, 190, "#ffffff", T, 2, 12)
text(sx, 1378, "Sampling an action chunk", 18, T, weight="700"); num(sx + 388, 1372, 10)
grid(sx, 1416, 96, 60, 8, 6, "#f3f4f6", "#6b7280", "8 × 29", "ε")
arrow([(sx + 96, 1446), (sx + 132, 1446)])
op(sx + 134, 1420, 152, 52, "a ← a + v / K", "τ = k / K, k = 0 … K−1")
arrow([(sx + 286, 1446), (sx + 318, 1446)])
grid(sx + 320, 1416, 96, 60, 8, 6, GR, GRS, "8 × 29", "joint targets")

# ---------------- legend
n_diagram = len(o)
lx = 1460
def para(y, n, lines):
    num(lx, y - 6, n)
    for i, l in enumerate(lines):
        text(lx + 26, y + i * 26, l, 17)
    return y + len(lines) * 26 + 20
text(lx - 14, 38, "Each stage, in order:", 20, T, weight="700")
y = 80
y = para(y, 1, ["Each depth image (depth scaled to [0, 1], and its", "validity mask) goes through its own CNN: four", "stride-2 convolutions, then one linear layer to a", "64-vector. The two cameras do not share weights."])
y = para(y, 2, ["Proprioception: the last 3 frames of joint angles q", "and of the policy's own targets q*, 3 × 2 × 29 = 174."])
y = para(y, 3, ["The text command, pick or place, as a one-hot."])
y = para(y, 4, ["Concatenate to 416 and mix with an MLP: the", "observation vector c, computed once per chunk."])
text(lx - 14, y + 10, "Per flow step:", 20, T, weight="700"); y += 50
y = para(y, 5, ["The noised chunk a_τ = τ·a + (1 − τ)·ε is flattened", "(8 × 29 = 232) and projected to x, 512."])
y = para(y, 6, ["τ gets a sinusoidal embedding and an MLP.", "cond = [c | τ], 640, conditions every block."])
y = para(y, 7, ["FiLM: one linear layer maps cond to a scale γ and a", "shift β for each of the 512 channels of LN(x)."])
y = para(y, 8, ["x ← x + W₂ GELU(W₁ FiLM(LN(x))), 512 → 1024 → 512."])
y = para(y, 9, ["LN and a zero-initialised linear layer give the", "velocity, reshaped to 8 × 29. A fresh policy outputs 0."])
y = para(y, 10, ["Euler steps from τ = 0 to 1. Training rollouts:", "ε ~ N(0, I), K = 10. Deployment: ε = 0, K = 5 or 10.", "The encoder runs once; steps 5–9 run K times."])
text(lx - 14, y + 10, "Training target:", 20, T, weight="700"); y += 44
for i, l in enumerate(["v should equal a − ε (conditional flow matching).", "The loss is the squared error summed over the 232", "numbers of a chunk; FPO++ compares it before and", "after an update on the same (τ, ε)."]):
    text(lx - 14, y + i * 26, l, 17)
y += 4 * 26 + 24
text(lx - 14, y, "Colours:", 20, T, weight="700"); y += 32
for fill, stroke, lab in ((OR, ORS, "wrist camera"), (BL, BLS, "exo camera"), (YE, YES, "proprioception"), (GR, GRS, "command, c and the output")):
    rect(lx - 14, y - 15, 22, 18, fill, stroke, 2, 3); text(lx + 18, y, lab, 17); y += 28

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">\n'
       f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{T}"/></marker></defs>\n'
       f'<rect width="{W}" height="{H}" fill="#f3fbf8"/>\n<g transform="translate(64,0)">\n' + "\n".join(o[:n_diagram]) + "\n</g>\n" + "\n".join(o[n_diagram:]) + "\n</svg>\n")
open(sys.argv[1], "w").write(svg)
