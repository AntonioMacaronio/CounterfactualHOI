"""Physical properties of a clip's object from a VLM (Qwen3.8-27B, scripts/physprops/serve_qwen.sh): material, how
it is built (solid / hollow / filled), mass with a range, and sliding friction and restitution against a table.

  python3 scripts/physprops/estimate.py VID [VID ...] [--refine refine_r10]          (stdlib + numpy + cv2: sam3d env)

Per clip the model sees 4 frames of the video: two with the object at rest and two while it is held (by the
fprecon run's SAM3 masks: largest object area, alone and with the hand), each as a crop around the object with
its outline in green, plus the whole first frame. The text gives the object's name (the generation prompt's
object for generated clips, here the real clip's name) and its reconstructed size: the extents along its long, mid
and short axes and its convex-hull volume, from the refinement's export (the size the simulator uses).
The answer is constrained to a JSON schema (vLLM structured output); the model reasons in "reasoning" first.
--samples answers (default 7, temperature 0.7) are drawn and the numbers are their medians: single answers for the
same object in two clips differed by up to 2.7x (the tape roll: 120 g and 45 g), and 3 samples of one clip by up to
2.9x. mass_spread (the samples' interquartile range over the median) flags the uncertain ones.

Writes $PROJECT_STORAGE_ROOT/runs/physprops/VID/physprops.json (the answer, the inputs, the model) and the image it
sent (inputs.jpg). scripts/physprops/apply_mass.py writes the mass into a FlashDexRetarget dataset;
scripts/mjwarp/trials.py samples the trials around it (mass +-30%).
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

STORAGE = Path(os.environ.get("PROJECT_STORAGE_ROOT", "/data/sky/projects/CounterfactualHOI"))
CHOIR = Path(os.environ.get("CHOIR_ROOT", STORAGE / "build" / "CHOIR"))
NAMES = {1: "small metal device (a RealSense-style depth camera)", 2: "blue tape roll", 3: "black spatula",
         4: "gray mug", 5: "dish soap bottle"}
SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "maxLength": 2000, "description": "What the object is, what it is made of, how it is built "
                      "(solid, hollow, a container and how full), and how that and its size give its mass."},
        "object": {"type": "string"},
        "materials": {"type": "array", "items": {"type": "string"}},
        "construction": {"type": "string", "enum": ["solid", "hollow shell", "container, empty", "container, partly full",
                                                    "container, full", "assembly of parts"]},
        "mass_g": {"type": "number"},
        "mass_low_g": {"type": "number"},
        "mass_high_g": {"type": "number"},
        "friction_table": {"type": "number", "description": "sliding friction coefficient against a wooden or laminate tabletop"},
        "friction_low": {"type": "number"},
        "friction_high": {"type": "number"},
        "restitution": {"type": "number", "description": "coefficient of restitution when dropped on the table, 0-1"},
        "center_of_mass": {"type": "string", "description": "where the centre of mass sits relative to the geometric centre, and why"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["reasoning", "object", "materials", "construction", "mass_g", "mass_low_g", "mass_high_g",
                 "friction_table", "friction_low", "friction_high", "restitution", "center_of_mass", "confidence"],
}


def frames_and_masks(vid):
    """RGB frames (CHOIR's stage-1 frames) and SAM3 object / hand masks (the fprecon run)."""
    fr = CHOIR / "output" / vid / "inputs" / "frames" / "rgb"
    ms = STORAGE / "runs" / "fprecon" / vid / "masks_sam3"
    n = len(list(fr.glob("*.png")))
    return n, (lambda i: cv2.imread(str(fr / f"{i}.png"))), (lambda i, k: cv2.imread(str(ms / k / f"{i}.png"), 0))


def pick_frames(vid, n, mask):
    """Two frames with the object alone (largest object area with no hand on it) and two held (hand touching)."""
    meta = json.load(open(STORAGE / "runs" / "fprecon" / vid / "masks_sam3" / "masks.json"))["per_frame"]
    alone, held = [], []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    for i in range(0, n, max(1, n // 40)):
        om, hm = mask(i, "object"), mask(i, "hand")
        if om is None or (om > 127).sum() < 400:
            continue
        touch = hm is not None and ((cv2.dilate((hm > 127).astype(np.uint8), k) > 0) & (om > 127)).any()
        (held if touch else alone).append((meta[i]["object_area"] if i < len(meta) else (om > 127).mean(), i))
    alone, held = sorted(alone, reverse=True), sorted(held, reverse=True)
    pick = [i for _, i in alone[:2]] + [i for _, i in held[:2]]
    return sorted(set(pick)) or [0]


def crop(img, om, side=448):
    ys, xs = np.nonzero(om > 127)
    over = img.copy()
    cnts, _ = cv2.findContours((om > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, cnts, -1, (0, 255, 0), 2)
    if len(xs) == 0:
        return cv2.resize(over, (side, side))
    cx, cy, r = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, 0.75 * max(xs.ptp(), ys.ptp()) + 20
    x0, y0 = int(max(0, cx - r)), int(max(0, cy - r))
    x1, y1 = int(min(img.shape[1], cx + r)), int(min(img.shape[0], cy + r))
    return cv2.resize(over[y0:y1, x0:x1], (side, side))


def size_of(vid, refine):
    """(extents long/mid/short in cm, convex-hull volume in cm^3) of the mesh the simulator uses."""
    ex = STORAGE / "runs" / refine / vid / "export"
    shape = json.load(open(ex / "object_shape.json")) if (ex / "object_shape.json").exists() else {}
    obj = ex / "object_mesh.obj" if (ex / "object_mesh.obj").exists() else STORAGE / "runs" / "fprecon" / vid / "object_mesh.obj"
    V = np.array([[float(t) for t in l.split()[1:4]] for l in open(obj) if l.startswith("v ")])
    if shape.get("mesh_scale"):
        V = V * float(shape["mesh_scale"]) if not (ex / "object_mesh.obj").exists() else V
    V = V - V.mean(0)
    w, U = np.linalg.eigh(np.cov(V.T))
    ext = np.ptp(V @ U, axis=0)[::-1] * 100
    try:
        from scipy.spatial import ConvexHull
        vol = ConvexHull(V).volume * 1e6
    except Exception:  # noqa: BLE001
        vol = float(np.prod(ext)) * 0.5
    return [round(float(e), 1) for e in ext], round(float(vol), 0)


def ask(url, model, text, images, timeout=600, temperature=0.2):
    content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(
        cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()).decode()}} for im in images]
    content.append({"type": "text", "text": text})
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 6000, "temperature": temperature,
            "response_format": {"type": "json_schema", "json_schema": {"name": "physprops", "schema": SCHEMA}},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    for attempt in range(3):                      # a sampled answer can run past max_tokens: ask again
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.load(r)
        try:
            return json.loads(out["choices"][0]["message"]["content"]), out.get("usage", {})
        except json.JSONDecodeError:
            print(f"   unparseable answer (finish_reason {out['choices'][0].get('finish_reason')}), retrying", flush=True)
    raise RuntimeError("no parseable answer in 3 attempts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vids", nargs="+")
    ap.add_argument("--refine", default="refine_r10", help="the refine round whose exported mesh the simulator uses")
    ap.add_argument("--url", default=os.environ.get("QWEN_URL", "http://127.0.0.1:30021"))
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--name", default=None, help="the object's name (default: the real clips' names by suffix)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--samples", type=int, default=7, help="answers per clip (temperature 0.7); the numbers are their medians")
    a = ap.parse_args()
    for vid in a.vids:
        out = STORAGE / "runs" / "physprops" / vid
        if (out / "physprops.json").exists() and not a.force:
            print(f"{vid}: done, skipping")
            continue
        out.mkdir(parents=True, exist_ok=True)
        n, rgb, mask = frames_and_masks(vid)
        fr = pick_frames(vid, n, mask)
        ims = [crop(rgb(i), mask(i, "object")) for i in fr]
        first = rgb(0)
        ims.append(cv2.resize(first, (896, int(896 * first.shape[0] / first.shape[1]))))
        ext, vol = size_of(vid, a.refine)
        name = a.name or NAMES.get(int(vid.rsplit("_", 1)[-1]), "object")
        text = (f"The images show a right hand manipulating one object on a table: {name}. The first {len(fr)} images are "
                f"crops around the object (its outline in green) from frames {fr} of the video; the last one is the "
                f"whole first frame. A 3D reconstruction of the object measures {ext[0]} x {ext[1]} x {ext[2]} cm along "
                f"its long, mid and short axes, with a convex-hull volume of {vol:.0f} cm^3 (handles and cavities "
                "included). Estimate its physical properties for a physics simulation: what it is made of and how it is "
                "built (solid, a hollow shell, or a container and how full it is), its mass in grams with a plausible "
                "range, its sliding friction coefficient against a wooden or laminate tabletop with a range, its "
                "restitution when dropped on the table, and where its centre of mass is. Use the reconstructed size; "
                "do not assume the object is solid if it is not. Reason first, in the reasoning field.")
        t0 = time.time()
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(min(4, a.samples)) as ex:
            outs = list(ex.map(lambda _: ask(a.url, a.model, text, ims, temperature=0.2 if a.samples == 1 else 0.7),
                               range(a.samples)))
        answers = [o[0] for o in outs]
        usage = [o[1] for o in outs]
        med = lambda k: float(np.median([float(x[k]) for x in answers]))
        # the reported answer: the sample whose mass is the median, with the numeric fields replaced by the medians
        ans = dict(min(answers, key=lambda x: abs(float(x["mass_g"]) - med("mass_g"))))
        for k in ("mass_g", "mass_low_g", "mass_high_g", "friction_table", "friction_low", "friction_high", "restitution"):
            ans[k] = med(k)
        q1, q3 = np.percentile([float(x["mass_g"]) for x in answers], [25, 75])
        ans["mass_spread"] = float((q3 - q1) / max(ans["mass_g"], 1e-6))
        sheet = np.concatenate([cv2.resize(im, (448, 448)) for im in ims[:-1]], 1)
        cv2.imwrite(str(out / "inputs.jpg"), sheet)
        rec = {"vid": vid, "answer": ans, "samples": answers, "inputs": {"name": name, "frames": fr, "extent_cm_long_mid_short": ext,
               "hull_volume_cm3": vol, "refine": a.refine}, "model": a.model, "seconds": round(time.time() - t0, 1),
               "usage": usage, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(rec, open(out / "physprops.json", "w"), indent=1)
        print(f"{vid}: {ans['object']}; {ans['construction']}; {ans['mass_g']:.0f} g ({ans['mass_low_g']:.0f}-"
              f"{ans['mass_high_g']:.0f}); friction {ans['friction_table']:.2f}; restitution {ans['restitution']:.2f}; "
              f"{ans['confidence']} confidence; spread {ans['mass_spread']:.2f}; samples {[round(float(x['mass_g'])) for x in answers]} g ({rec['seconds']:.0f} s)",
              flush=True)


if __name__ == "__main__":
    sys.exit(main())
