"""utils3d compatibility hook for the choir env, installed by scripts/choir/setup.sh as site-packages/
utils3d_compat.py plus utils3d_compat.pth (so every choir-env process, incl. multiprocessing workers, loads it).

CHOIR's env has two users of utils3d with incompatible APIs: its MoGe-2 pins utils3d @3fab839 (v1.3, the
new utils3d.np / utils3d.pt API), while SAM3D's mesh post-processing (sam3d_objects/.../postprocessing_utils.py)
still calls six old utils3d.torch functions that v1.3 removed. requirements-choir.txt pins neither.

When utils3d.torch is first imported, fill in just those missing names from the old utils3d (@3913c65, the
version Do-as-I-Do's sam3d env uses), installed as `utils3d_legacy` under $UTILS3D_LEGACY_DIR by
scripts/choir/setup.sh. Lazy, so processes that never touch utils3d don't import torch here.
"""

import importlib.abc
import importlib.util
import os
import sys

_MISSING = ("compute_connected_components", "remove_unreferenced_vertices", "depth_to_points",
            "compute_edges", "compute_edge_connected_components", "compute_dual_graph")
_LEGACY_DIR = os.environ.get("UTILS3D_LEGACY_DIR", "@UTILS3D_LEGACY_DIR@")   # default filled in at install


class _PatchUtils3dTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "utils3d.torch" or not _LEGACY_DIR:
            return None
        sys.meta_path.remove(self)                  # find the real module with the normal finders
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        real_exec = spec.loader.exec_module

        def exec_module(module):
            real_exec(module)
            if _LEGACY_DIR not in sys.path:
                sys.path.append(_LEGACY_DIR)
            import utils3d_legacy.torch as legacy
            for n in _MISSING:
                if not hasattr(module, n) and hasattr(legacy, n):
                    setattr(module, n, getattr(legacy, n))

        spec.loader.exec_module = exec_module
        return spec


if _LEGACY_DIR:
    sys.meta_path.insert(0, _PatchUtils3dTorch())
