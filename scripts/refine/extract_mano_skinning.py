"""Extract MANO's skinning weights, kinematic tree, joint regressor and template (numpy only) for the finger tools.

  python -I scripts/refine/extract_mano_skinning.py      (any env with numpy + scipy; run by scripts/fprecon/setup.sh)

Reads MANO_{RIGHT,LEFT}.pkl from Dyn-HaMR's data dir in the CHOIR build (MANO_DIR to override) and writes
$PROJECT_STORAGE_ROOT/models/mano/skinning_{right,left}.npz (weights 778x16, parents 16, J 16x778, v_template 778x3).
The pkls hold chumpy arrays; a stand-in module unpickles them without installing chumpy.
"""

import os
import pickle
import sys
import types

import numpy as np

STORAGE = os.environ.get("PROJECT_STORAGE_ROOT", "/data/sky/projects/CounterfactualHOI")
SRC = os.environ.get("MANO_DIR", os.path.join(STORAGE, "build", "CHOIR", "stage1_preprocess", "Dyn_HaMR_new", "_DATA",
                                              "data", "mano"))


class _Ch:
    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"x": state})


for name in ("chumpy", "chumpy.ch", "chumpy.reordering"):
    m = types.ModuleType(name)
    m.Ch = m.Select = m.transpose = m.Transpose = _Ch
    sys.modules[name] = m


def arr(x):
    if isinstance(x, _Ch):
        for k in ("x", "a", "r"):
            if k in x.__dict__:
                return arr(x.__dict__[k])
        raise ValueError(f"unknown chumpy payload {list(x.__dict__)}")
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def main():
    out = os.path.join(STORAGE, "models", "mano")
    os.makedirs(out, exist_ok=True)
    for side, f in (("right", "MANO_RIGHT.pkl"), ("left", "MANO_LEFT.pkl")):
        d = pickle.load(open(os.path.join(SRC, f), "rb"), encoding="latin1")
        parents = arr(d["kintree_table"]).astype(np.int64)[0]
        parents[0] = -1
        np.savez(os.path.join(out, f"skinning_{side}.npz"), weights=arr(d["weights"]).astype(np.float32), parents=parents,
                 J=arr(d["J_regressor"]).astype(np.float32), v_template=arr(d["v_template"]).astype(np.float32))
        print(f"wrote {out}/skinning_{side}.npz")


if __name__ == "__main__":
    main()
