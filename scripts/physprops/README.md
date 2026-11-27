# Object physical properties from a VLM (Qwen3.8-27B)

| file | what |
|---|---|
| `serve_qwen.sh [GPU]` | vLLM server for `Qwen/Qwen3.8-27B-FP8` (image + text): 27% of a GPU (~37 GB), 16k context, eager mode, so it can share a GPU. Stop it when done |
| `estimate.py VID ...` | per clip: material, construction, mass with a range, sliding friction against a table, restitution, centre of mass -> `runs/physprops/VID/physprops.json` |
| `apply_mass.py DATASET` | the mass into a FlashDexRetarget dataset: `obj_right_mass` in `human_demo/*.npz` and `right_object_masses` in its packs |

Setup (both on /data):

```sh
$PROJECT_STORAGE_ROOT/venvs/hf-dl/bin/hf download Qwen/Qwen3.8-27B-FP8        # 29 GB, HF cache
uv venv --python 3.12 $PROJECT_STORAGE_ROOT/venvs/vllm && uv pip install --python $PROJECT_STORAGE_ROOT/venvs/vllm/bin/python vllm
```

**What the model sees.**
- Images: 4 crops of the object with its SAM3 outline (two frames at rest, two held), plus the whole first frame.
- Text: the object's name (for generated videos, the generation prompt's object), its reconstructed extents along
  its long, mid and short axes, and its convex-hull volume (the mesh the simulator uses).
- Output: a JSON schema (vLLM structured output), with the reasoning first.

**Sampling.**
- Seven answers per clip, at temperature 0.7.
- The numbers are their medians, and `mass_spread` (IQR / median) flags uncertain ones.
- Single answers for one object disagreed by up to 2.7× (the tape roll: 120 and 45 g), so one sample is not enough.
- Each clip takes 40–55 s on a shared H200.

Round 10, the 10 real clips (median, range of the median answer, spread):

| object | pp | prp | FDR's density estimate (800 kg/m³) |
|---|---|---|---|
| small metal device | 110 g (90–120), 0.25 | 150 g (120–200), 0.23 | 128 / 101 g |
| blue tape roll | 50 g (40–65), **1.10** | 120 g (90–160), 0.35 | 161 / 169 g |
| black spatula | 100 g (80–120), 0.47 | 120 g (90–150), 0.49 | 103 / 93 g |
| gray mug | 400 g (350–450), 0.03 | 400 g (300–500), 0.03 | 188 / 235 g |
| dish soap bottle | 450 g (400–500), 0.51 | 500 g (470–530), 0.24 | 500 / 426 g |

How far to trust these:
- The mug is consistent between the two clips and with a ceramic mug. FDR's density estimate had it at half that.
- The tape roll is uncertain: 50 g against 120 g between the two clips, with a spread of 1.1.
- Weighing the five real objects would settle these numbers and give a check of the VLM before it is used on
  generated videos.
- The trials vary every mass by ±30% (`scripts/mjwarp/trials.py`).
