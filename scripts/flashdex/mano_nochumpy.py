"""Write chumpy-free copies of MANO_{RIGHT,LEFT}.pkl (numpy arrays instead of chumpy objects) for smplx without chumpy.

  python -I scripts/flashdex/mano_nochumpy.py SRC_DIR DST_DIR
"""
import os, pickle, sys, types
import numpy as np


class _Ch:
    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"x": state})


for name in ("chumpy", "chumpy.ch", "chumpy.reordering"):
    m = types.ModuleType(name)
    m.Ch = m.Select = m.transpose = m.Transpose = _Ch
    sys.modules[name] = m


def plain(x):
    if isinstance(x, _Ch):
        for k in ("x", "a", "r"):
            if k in x.__dict__:
                return plain(x.__dict__[k])
        raise ValueError(list(x.__dict__))
    return x


src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for f in ("MANO_RIGHT.pkl", "MANO_LEFT.pkl"):
    d = pickle.load(open(os.path.join(src, f), "rb"), encoding="latin1")
    out = {k: (np.asarray(plain(v)) if isinstance(v, _Ch) else v) for k, v in d.items()}
    p = os.path.join(dst, f)
    if os.path.islink(p):
        os.remove(p)
    pickle.dump(out, open(p, "wb"))
    print("wrote", p, {k: getattr(v, "shape", type(v).__name__) for k, v in out.items() if k in ("v_template", "shapedirs", "posedirs", "J_regressor", "f")})
