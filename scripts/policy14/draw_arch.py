"""Draws antzhang-notes/policy_arch.svg, the tensor-flow diagram of scripts/policy14/model.py (the flow transformer).

  python3 scripts/policy14/draw_arch.py antzhang-notes/policy_arch.svg      (stdlib only)

Shapes and parameter counts are written in by hand from model.py's printout; redraw after changing the model.
Matrices are drawn with one column per token (width 256 down the page), as in the user's reference diagram.
"""

import sys

W, H = 2150, 1780
T, INK, MUTE = "#2f6f66", "#1f3f3a", "#4b6b66"
YE, YES = "#fbf6dc", "#a8862a"      # proprioception
GR, GRS = "#e3f4ee", "#2f6f66"      # command, flow time, output
OR, ORS = "#fdf1e7", "#c0532c"      # wrist camera
BL, BLS = "#e9eefc", "#3b4fd0"      # exo camera
PU, PUS = "#f3e8fb", "#7b3fb0"      # action tokens
o = []


def rect(x, y, w, h, fill="#ffffff", stroke=T, sw=2, rx=0, dash=None):
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" rx="{rx}"'
             + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")


def text(x, y, s, size=17, fill=INK, anchor="start", weight="400", mono=False, italic=False):
    fam = "ui-monospace, Menlo, Consolas, monospace" if mono else "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
    o.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}" '
             f'font-family="{fam}"' + (' font-style="italic"' if italic else "") + f">{s}</text>")


def line(pts):
    o.append('<path d="M' + " L".join(f"{x},{y}" for x, y in pts) + f'" fill="none" stroke="{T}" stroke-width="2.2"/>')


def arrow(pts):
    o.append('<path d="M' + " L".join(f"{x},{y}" for x, y in pts) + f'" fill="none" stroke="{T}" stroke-width="2.2" marker-end="url(#ah)"/>')


def op(x, y, w, h, label, sub=None, size=17):
    rect(x, y, w, h, rx=9, sw=2.4)
    text(x + w / 2, y + h / 2 + 6, label, size, anchor="middle", weight="700")
    if sub:
        text(x + w / 2, y + h + 19, sub, 14, MUTE, "middle", mono=True)


def num(x, y, n):
    o.append(f'<circle cx="{x}" cy="{y}" r="13" fill="{T}"/>')
    text(x, y + 6, str(n), 16, "#ffffff", "middle", "700")


def head(x, y, s):
    text(x, y, s, 21, T, weight="700")


def plus(x, y):
    o.append(f'<circle cx="{x}" cy="{y}" r="15" fill="#ffffff" stroke="{T}" stroke-width="2.4"/>')
    text(x, y + 7, "+", 22, T, "middle", "700")


def times(x, y):
    o.append(f'<circle cx="{x}" cy="{y}" r="15" fill="#ffffff" stroke="{T}" stroke-width="2.4"/>')
    text(x, y + 7, "×", 20, T, "middle", "700")


def tokens(x, y, h, segs, shape, name=None, rows=0, colw=4.0):
    """A matrix drawn as columns of tokens: segs = [(n_tokens, fill, stroke), ...]; rows = dashed head rows."""
    cx = x
    for n, fill, stroke in segs:
        w = n * colw
        rect(cx, y, w, h, fill, stroke, 2.0, 1)
        for k in range(1, n):
            xx = cx + k * colw
            o.append(f'<line x1="{xx}" y1="{y}" x2="{xx}" y2="{y + h}" stroke="{stroke}" stroke-width="0.5" opacity="0.5"/>')
        cx += w
    for k in range(1, rows):
        yy = y + k * h / rows
        o.append(f'<line x1="{x}" y1="{yy}" x2="{cx}" y2="{yy}" stroke="{MUTE}" stroke-width="0.9" stroke-dasharray="4 3" opacity="0.8"/>')
    text((x + cx) / 2, y - 9, shape, 15, MUTE, "middle", mono=True)
    if name:
        text((x + cx) / 2, y + h + 20, name, 16, INK, "middle")
    return cx


def trap(x, c, w, h_in, h_out, label, sub):
    o.append(f'<path d="M{x},{c - h_in / 2} L{x + w},{c - h_out / 2} L{x + w},{c + h_out / 2} L{x},{c + h_in / 2} Z" '
             f'fill="#ffffff" stroke="{T}" stroke-width="2.4"/>')
    text(x + w / 2, c + 6, label, 18, anchor="middle", weight="700", italic=True)
    text(x + w / 2, c + max(h_in, h_out) / 2 + 20, sub, 14, MUTE, "middle", mono=True)


OBS = [(3, YE, YES), (1, GR, GRS), (28, OR, ORS), (28, BL, BLS)]
SEQ = OBS + [(8, PU, PUS)]

# ---------------- A: observation tokens
head(20, 36, "Observation tokens, once per action chunk (tokenizer 0.54 M; the two CNNs 0.24 M each)")
for k, (y, fill, stroke, name) in enumerate(((70, OR, ORS, "wrist depth + mask"), (200, BL, BLS, "exo depth + mask"))):
    rect(20, y, 124, 70, fill, stroke, 2.2, 2)
    text(82, y - 9, "2 × 60 × 106", 15, MUTE, "middle", mono=True)
    text(82, y + 90, name, 16, INK, "middle")
    arrow([(144, y + 35), (190, y + 35)])
    op(192, y + 9, 320, 52, "CNN: 4 × (conv s2, GroupNorm, GELU)", "→ 128 × 4 × 7", 14)
    arrow([(512, y + 35), (556, y + 35)])
    tokens(558, y + 5, 60, [(28, fill, stroke)], "128 × 28", "28 patch tokens", colw=3.2)
    arrow([(648, y + 35), (694, y + 35)])
    op(696, y + 9, 200, 52, "W_p + pos + cam", "128→256, shared", 15)
    line([(896, y + 35), (930, y + 35)])
num(352, 62, 1)
rect(20, 340, 26, 82, YE, YES, 2.2, 2); text(33, 330, "174", 15, MUTE, "middle", mono=True); text(33, 446, "proprio", 16, INK, "middle")
arrow([(46, 381), (190, 381)])
op(192, 355, 320, 52, "regroup per frame [q | q*]", "3 × 58", 15)
num(352, 347, 2)
arrow([(512, 381), (694, 381)])
op(696, 355, 200, 52, "W_q + frame emb.", "58→256", 15)
line([(896, 381), (930, 381)])
rect(20, 486, 26, 22, GR, GRS, 2.2, 2); text(33, 478, "2", 15, MUTE, "middle", mono=True); text(33, 532, "command", 16, INK, "middle")
arrow([(46, 497), (694, 497)])
op(696, 471, 200, 52, "W_c", "2→256", 15)
num(796, 463, 3)
line([(896, 497), (930, 497)])
line([(930, 105), (930, 497)])
arrow([(930, 285), (960, 285)])
x_end = tokens(962, 170, 230, OBS, "256 × 60", None, colw=7.0)
text((962 + x_end) / 2, 425, "O = [proprio ×3 | command | wrist ×28 | exo ×28]", 16, INK, "middle")
num(946, 150, 4)

# ---------------- B: flow transformer input
y0 = 590
head(20, y0, "Flow transformer v(a_τ, τ; O), once per flow step (7.35 M)")
rect(20, y0 + 40, 124, 76, "#ffffff", PUS, 2.2, 2)
text(82, y0 + 31, "8 × 29", 15, MUTE, "middle", mono=True); text(82, y0 + 136, "noised chunk a_τ", 16, INK, "middle")
arrow([(144, y0 + 78), (190, y0 + 78)])
op(192, y0 + 52, 250, 52, "W_a + step emb.", "29→256 per step", 15)
num(317, y0 + 44, 5)
arrow([(442, y0 + 78), (490, y0 + 78)])
tokens(492, y0 + 40, 76, [(8, PU, PUS)], "256 × 8", "action tokens A", colw=7.0)
arrow([(548, y0 + 78), (612, y0 + 78)])
x_seq = tokens(614, y0 + 20, 120, SEQ, "256 × 68", None, colw=7.0)
text((614 + x_seq) / 2, y0 + 162, "X = [O | A]", 17, INK, "middle", "700")
arrow([((962 + x_end) / 2, 436), ((962 + x_end) / 2, 528), (824, 528), (824, y0 + 8)])
text((962 + x_end) / 2 + 12, 500, "O", 18, INK, weight="700", italic=True)
# time
o.append(f'<circle cx="{x_seq + 90}" cy="{y0 + 80}" r="15" fill="#ffffff" stroke="{T}" stroke-width="2.2"/>')
text(x_seq + 90, y0 + 86, "τ", 18, INK, "middle", italic=True)
arrow([(x_seq + 105, y0 + 80), (x_seq + 150, y0 + 80)])
op(x_seq + 152, y0 + 54, 230, 52, "sin/cos → MLP", "128 → 256 → 256", 15)
num(x_seq + 267, y0 + 46, 6)
arrow([(x_seq + 382, y0 + 80), (x_seq + 430, y0 + 80)])
rect(x_seq + 432, y0 + 40, 26, 80, GR, GRS, 2.2, 2)
text(x_seq + 445, y0 + 31, "256", 15, MUTE, "middle", mono=True)
text(x_seq + 445, y0 + 145, "t", 20, INK, "middle", "700", italic=True)

# ---------------- C: block
by = y0 + 210
rect(20, by, 1560, 690, "none", T, 2, 12, "8 6")
text(36, by + 30, "Block × 6 (1.18 M each): every token attends to every other; all of them modulated by t (adaLN-Zero)", 18, T, weight="700")
# adaLN generator
op(36, by + 56, 330, 50, "W_ada: SiLU(t) → 6 × 256", "256 → 1536", 15)
text(380, by + 86, "shift b, scale s, gate g for the attention (a) and the MLP (m)", 15, MUTE, italic=True)
arrow([(x_seq + 445, y0 + 120), (x_seq + 445, by + 40), (366, by + 40), (366, by + 56)])
num(201, by + 48, 7)
cy = by + 280
xx = tokens(36, cy - 50, 100, SEQ, "256 × 68", None, colw=3.0)
text((36 + xx) / 2, cy + 74, "X", 18, INK, "middle", "700", italic=True)
arrow([(xx, cy), (xx + 30, cy)])
op(xx + 32, cy - 26, 150, 52, "LN · (1+s_a) + b_a", None, 14)
arrow([(xx + 182, cy), (xx + 215, cy)])
qx = xx + 217
for k, (lab, yy) in enumerate((("W_q", cy - 120), ("W_k", cy), ("W_v", cy + 120))):
    arrow([(xx + 215, cy), (xx + 215, yy), (qx, yy)])
    op(qx, yy - 26, 80, 52, lab, None, 16)
    arrow([(qx + 80, yy), (qx + 118, yy)])
    tokens(qx + 120, yy - 40, 80, SEQ, "256 × 68" if k == 0 else "", ["Q", "K", "V"][k], rows=8, colw=2.6)
text(qx + 40, cy + 178, "256→256 each", 14, MUTE, "middle", mono=True)
num(qx + 40, cy - 156, 8)
# attention per head
ax = qx + 330
text(ax, cy - 120, "each head h = 1 … 8 (32 rows):", 17, T, weight="700")
rect(ax, cy - 95, 150, 150, "#fdf8f2", T, 2.0, 2)
for k in range(1, 8):
    p = ax + k * 150 / 8
    o.append(f'<line x1="{p}" y1="{cy - 95}" x2="{p}" y2="{cy + 55}" stroke="{T}" stroke-width="0.5" opacity="0.4"/>')
    p = cy - 95 + k * 150 / 8
    o.append(f'<line x1="{ax}" y1="{p}" x2="{ax + 150}" y2="{p}" stroke="{T}" stroke-width="0.5" opacity="0.4"/>')
rect(ax + 150 * 60 / 68, cy - 95, 150 * 8 / 68, 150, PU, PUS, 1.2, 0)
text(ax + 75, cy + 78, "A_h = softmax(Q_hᵀ K_h / √32)", 15, INK, "middle")
text(ax + 75, cy - 104, "68 × 68", 14, MUTE, "middle", mono=True)
num(ax - 20, cy - 126, 9)
arrow([(qx + 300, cy - 120), (ax - 4, cy - 60)])
arrow([(qx + 300, cy), (ax - 4, cy - 10)])
times(ax + 200, cy - 20)
arrow([(ax + 150, cy - 20), (ax + 185, cy - 20)])
arrow([(qx + 300, cy + 120), (ax + 200, cy + 120), (ax + 200, cy - 5)])
arrow([(ax + 215, cy - 20), (ax + 250, cy - 20)])
tokens(ax + 252, cy - 40, 40, SEQ, "32 × 68", "O_h = V_h A_hᵀ", colw=2.0)
# merge + MLP
mx = ax + 400
arrow([(ax + 388, cy - 20), (mx, cy - 20)])
op(mx + 2, cy - 46, 120, 52, "stack, W_o", "256→256", 15)
arrow([(mx + 122, cy - 20), (mx + 150, cy - 20)])
times(mx + 165, cy - 20); text(mx + 165, cy - 48, "g_a", 15, MUTE, "middle", italic=True)
arrow([(mx + 180, cy - 20), (mx + 210, cy - 20)])
plus(mx + 225, cy - 20)
arrow([(56, cy - 52), (56, by + 206), (mx + 225, by + 206), (mx + 225, cy - 35)])
text(mx + 120, by + 199, "skip: X", 14, MUTE, "middle", italic=True)
num(mx + 260, cy - 50, 10)
# MLP row below
my = cy + 300
arrow([(mx + 225, cy - 5), (mx + 225, my - 90), (40, my - 90), (40, my), (70, my)])
op(72, my - 26, 170, 52, "LN · (1+s_m) + b_m", None, 14)
arrow([(242, my), (280, my)])
trap(282, my, 96, 80, 150, "W₁", "256→1024")
arrow([(378, my), (412, my)])
op(414, my - 26, 90, 52, "GELU")
arrow([(504, my), (540, my)])
trap(542, my, 96, 150, 80, "W₂", "1024→256")
arrow([(638, my), (676, my)])
times(691, my); text(691, my - 28, "g_m", 15, MUTE, "middle", italic=True)
arrow([(706, my), (740, my)])
plus(755, my)
arrow([(755, my - 90), (755, my - 15)])
text(470, my - 98, "skip: X after attention", 14, MUTE, "middle", italic=True)
arrow([(770, my), (810, my)])
xe = tokens(812, my - 50, 100, SEQ, "256 × 68", "X → next block", colw=3.0)
num(726, my - 52, 11)

# ---------------- D: output
oy = by + 740
head(20, oy, "Output and sampling")
tokens(36, oy + 40, 76, [(8, PU, PUS)], "256 × 8", "X[:, 60:68] after block 6", colw=7.0)
arrow([(92, oy + 78), (150, oy + 78)])
op(152, oy + 52, 230, 52, "LN · (1+s) + b, from t", None, 15)
arrow([(382, oy + 78), (410, oy + 78)])
op(412, oy + 52, 170, 52, "W_out", "256→29, zero init", 15)
num(497, oy + 44, 12)
arrow([(582, oy + 78), (630, oy + 78)])
rect(632, oy + 40, 124, 76, GR, GRS, 2.2, 2)
text(694, oy + 31, "8 × 29", 15, MUTE, "middle", mono=True); text(694, oy + 136, "velocity v", 16, INK, "middle")
sx = 860
rect(sx - 20, oy + 10, 800, 150, "#ffffff", T, 2, 12)
text(sx, oy + 40, "Sampling an action chunk", 18, T, weight="700"); num(sx + 750, oy + 34, 13)
rect(sx, oy + 66, 96, 60, "#f3f4f6", "#6b7280", 2, 2); text(sx + 48, oy + 58, "8 × 29", 14, MUTE, "middle", mono=True)
text(sx + 48, oy + 148, "ε", 16, INK, "middle", italic=True)
arrow([(sx + 96, oy + 96), (sx + 132, oy + 96)])
op(sx + 134, oy + 70, 260, 52, "a ← a + v(a, k/K; O) / K", "k = 0 … K−1", 15)
arrow([(sx + 394, oy + 96), (sx + 430, oy + 96)])
rect(sx + 432, oy + 66, 96, 60, GR, GRS, 2, 2); text(sx + 480, oy + 58, "8 × 29", 14, MUTE, "middle", mono=True)
text(sx + 480, oy + 148, "joint targets", 15, INK, "middle")
text(sx + 560, oy + 82, "training: ε ~ N(0, I), K = 10", 14, MUTE)
text(sx + 560, oy + 104, "deployed: ε = 0, K = 5 or 10", 14, MUTE)

# ---------------- legend
n_diagram = len(o)
lx = 1690
text(lx - 14, 36, "Each stage, in order:", 20, T, weight="700")


def para(y, n, lines):
    num(lx, y - 6, n)
    for i, l in enumerate(lines):
        text(lx + 26, y + i * 25, l, 16)
    return y + len(lines) * 25 + 16


y = 76
y = para(y, 1, ["Each depth image (depth in [0, 1], validity", "mask) through its own CNN: a 4 × 7 map,", "28 patch tokens of 128."])
y = para(y, 2, ["Proprioception regrouped per history frame:", "[q | q*], 3 tokens of 58."])
y = para(y, 3, ["The command, pick or place, as one token."])
y = para(y, 4, ["All projected to 256 with position, camera", "and frame embeddings: O, 60 tokens."])
text(lx - 14, y + 8, "Per flow step:", 20, T, weight="700"); y += 44
y = para(y, 5, ["The noised chunk a_τ = τ·a + (1−τ)·ε:", "8 action tokens with step embeddings."])
y = para(y, 6, ["τ → sinusoids → MLP: t, 256."])
y = para(y, 7, ["Each block maps t to shift, scale and gate", "for its attention and its MLP (adaLN-Zero:", "gates start at 0, the block as identity)."])
y = para(y, 8, ["Q, K, V for all 68 tokens; 8 heads of 32."])
y = para(y, 9, ["Each head attends over all 68 tokens, so", "actions see each other and every image patch."])
y = para(y, 10, ["X ← X + g_a · W_o [O_1; …; O_8]."])
y = para(y, 11, ["X ← X + g_m · W₂ GELU(W₁ mod(LN(X)))."])
y = para(y, 12, ["The 8 action tokens → v, 8 × 29.", "A fresh policy outputs 0."])
y = para(y, 13, ["Euler steps from τ = 0 to 1. The CNNs run", "once per chunk; steps 5–12 run K times."])
text(lx - 14, y + 8, "Training target:", 20, T, weight="700"); y += 40
for i, l in enumerate(["v = a − ε (conditional flow matching);", "squared error summed over the chunk.", "FPO++ compares it before and after an", "update on the same (τ, ε)."]):
    text(lx - 14, y + i * 25, l, 16)
y += 4 * 25 + 22
text(lx - 14, y, "Columns are tokens:", 20, T, weight="700"); y += 32
for fill, stroke, lab in ((YE, YES, "proprioception ×3"), (GR, GRS, "command ×1, t, output"), (OR, ORS, "wrist patches ×28"),
                          (BL, BLS, "exo patches ×28"), (PU, PUS, "action tokens ×8")):
    rect(lx - 14, y - 15, 22, 18, fill, stroke, 2, 3); text(lx + 18, y, lab, 16); y += 27

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">\n'
       f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{T}"/></marker></defs>\n'
       f'<rect width="{W}" height="{H}" fill="#f3fbf8"/>\n<g transform="translate(36,0)">\n' + "\n".join(o[:n_diagram])
       + "\n</g>\n" + "\n".join(o[n_diagram:]) + "\n</svg>\n")
open(sys.argv[1], "w").write(svg)
