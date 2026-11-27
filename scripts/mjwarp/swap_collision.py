"""FlashDexRetarget's one-object "swap" mode on MuJoCo Warp: the module FDR's command imports as
simulator.mujoco.swap_collision (envs/commands/motion_tracking/objects/multi_object.py), which its release lacks.
scripts/mjwarp/create.py registers this file under that name.

Swap mode (FDR_OBJ_SWAP=1; the default of scripts/mjwarp/create.py) gives every world ONE object entity instead of every clip's object (multi-object mode). The
entity's collision geoms are the first object's convex parts, padded with 1 mm tetrahedra up to the largest part
count; every object's parts are compiled into the model as unused ("orphan") meshes; at reset each world points its
geoms at its own object's meshes (per-world geom_dataid) and gets that object's geom poses, bounds and mass rows
(envs/scene/_object_swap.py, multi_object.py).

What is missing without this module: MuJoCo compiles an orphan mesh's vertices but not its convex-hull graph
(mesh_graphadr = -1) or its polygons (mesh_polynum = 0), because no geom uses it. MuJoCo Warp's convex collision uses
the graph for the GJK / EPA support function (hill climbing; without it, a scan of every vertex) and the polygons with
the vertex-to-polygon map for multi-contact. The import also fails without the module. Measured on the 10 round-10
objects (scripts/mjwarp/swap_check.py): without the hull data, object paths agree with the injected run to 0.01 cm
and the contact count drops 3% (194 against 200 per step). So the injection gives the swapped objects the same
collision path as an object used by a geom, but the fallback was close.

  mesh_collision_blob(m, mesh_id)    one mesh's hull graph, polygons and vertex->polygon map, from a model in which
                                     a geom uses it (FDR's harvest compiles each object on its own); mesh-local indices
  inject_swap_collision(sim, dataid, nhull, hulls)
                                     appends those blobs for every orphan mesh of the scene to the MuJoCo Warp model's
                                     arrays and points the meshes' addresses at them; also raises nmaxpolygon /
                                     nmaxmeshdeg, which size the multi-contact buffers. Runs before the CUDA graphs
                                     are recaptured (FDR calls it just before expand_model_fields).

  patch_assist_inertia()             makes FDR's object assist use each world's own object inertia in swap mode (it
                                     read world 0's; see the function). register() applies it.

Index conventions (checked on MuJoCo 3.8 with two meshes in one model): mesh_graph blocks, mesh_polyvert and
mesh_polymap hold mesh-local indices; only mesh_graphadr, mesh_polyadr, mesh_polyvertadr and mesh_polymapadr are
global addresses. So a blob is appended unchanged and only its start addresses are re-based.
"""

from __future__ import annotations

import numpy as np

_FIELDS = ("mesh_graph", "mesh_graphadr", "mesh_polynum", "mesh_polyadr", "mesh_polynormal", "mesh_polyvertadr",
           "mesh_polyvertnum", "mesh_polyvert", "mesh_polymapadr", "mesh_polymapnum", "mesh_polymap")


def _graph_block(m, mesh_id: int) -> np.ndarray:
    ga = int(m.mesh_graphadr[mesh_id])
    if ga < 0:
        raise ValueError(f"mesh {mesh_id} has no convex-hull graph in this model (no geom uses it)")
    starts = sorted(int(a) for a in m.mesh_graphadr if a > ga)
    end = starts[0] if starts else len(m.mesh_graph)
    return np.asarray(m.mesh_graph[ga:end], dtype=np.int32).copy()


def mesh_collision_blob(m, mesh_id: int) -> dict:
    """Hull graph, polygons and vertex->polygon map of mesh `mesh_id` in compiled MjModel `m` (mesh-local indices)."""
    mesh_id = int(mesh_id)
    va, vn = int(m.mesh_vertadr[mesh_id]), int(m.mesh_vertnum[mesh_id])
    pa, pn = int(m.mesh_polyadr[mesh_id]), int(m.mesh_polynum[mesh_id])
    if pn <= 0:
        raise ValueError(f"mesh {mesh_id} has no polygons in this model")
    pv_adr = np.asarray(m.mesh_polyvertadr[pa:pa + pn], np.int64)
    pv_num = np.asarray(m.mesh_polyvertnum[pa:pa + pn], np.int64)
    pv0, pv1 = int(pv_adr[0]), int(pv_adr[-1] + pv_num[-1])
    pm_adr = np.asarray(m.mesh_polymapadr[va:va + vn], np.int64)
    pm_num = np.asarray(m.mesh_polymapnum[va:va + vn], np.int64)
    pm0, pm1 = int(pm_adr[0]), int(pm_adr[-1] + pm_num[-1])
    return {
        "vert": np.asarray(m.mesh_vert[va:va + vn], np.float64).copy(),
        "graph": _graph_block(m, mesh_id),
        "polynormal": np.asarray(m.mesh_polynormal[pa:pa + pn], np.float64).copy(),
        "polyvertadr": (pv_adr - pv0).astype(np.int32),
        "polyvertnum": pv_num.astype(np.int32),
        "polyvert": np.asarray(m.mesh_polyvert[pv0:pv1], np.int32).copy(),
        "polymapadr": (pm_adr - pm0).astype(np.int32),
        "polymapnum": pm_num.astype(np.int32),
        "polymap": np.asarray(m.mesh_polymap[pm0:pm1], np.int32).copy(),
    }


def inject_swap_collision(sim, dataid, nhull, hulls, vert_tol: float = 1e-5) -> dict:
    """Give every orphan swap mesh of sim's MuJoCo Warp model the hull data in `hulls` (hulls[i][k] = blob of object
    i's part k, from mesh_collision_blob on that object's own compile; dataid[i, k] = the scene's mesh id for it).
    Meshes that already have hull data in the scene (the template object's, used by the entity's geoms) are left as
    they are. Returns a summary. Raises if an orphan's vertices differ from its blob's (a different processing of
    the same file would make the graph point at the wrong vertices)."""
    import warp as wp

    mjm, wm = sim.mj_model, sim.wp_model
    cur = {f: np.asarray(getattr(mjm, f)).copy() for f in _FIELDS}
    graph = [cur["mesh_graph"]]
    normal, pvadr, pvnum, pvert = [cur["mesh_polynormal"]], [cur["mesh_polyvertadr"]], [cur["mesh_polyvertnum"]], [cur["mesh_polyvert"]]
    pmap = [cur["mesh_polymap"]]
    graphadr, polyadr, polynum = cur["mesh_graphadr"], cur["mesh_polyadr"], cur["mesh_polynum"]
    pmadr, pmnum = cur["mesh_polymapadr"], cur["mesh_polymapnum"]
    n_graph, n_poly, n_pvert, n_pmap = len(cur["mesh_graph"]), len(cur["mesh_polynormal"]), len(cur["mesh_polyvert"]), len(cur["mesh_polymap"])
    done, skipped, max_err = 0, 0, 0.0
    seen = set()
    for i in range(len(hulls)):
        for k in range(int(nhull[i])):
            mid = int(dataid[i][k])
            if mid in seen:
                continue
            seen.add(mid)
            if graphadr[mid] >= 0 and polynum[mid] > 0:
                skipped += 1
                continue
            b = hulls[i][k]
            va, vn = int(mjm.mesh_vertadr[mid]), int(mjm.mesh_vertnum[mid])
            if vn != len(b["vert"]):
                raise ValueError(f"mesh {mid}: {vn} vertices in the scene, {len(b['vert'])} in its own compile")
            err = float(np.abs(np.asarray(mjm.mesh_vert[va:va + vn]) - b["vert"]).max())
            max_err = max(max_err, err)
            if err > vert_tol:
                raise ValueError(f"mesh {mid}: vertices differ from its own compile by {err:.2e}")
            graphadr[mid] = n_graph
            graph.append(b["graph"])
            n_graph += len(b["graph"])
            polyadr[mid], polynum[mid] = n_poly, len(b["polynormal"])
            normal.append(b["polynormal"])
            pvadr.append(b["polyvertadr"] + n_pvert)
            pvnum.append(b["polyvertnum"])
            pvert.append(b["polyvert"])
            n_poly += len(b["polynormal"])
            n_pvert += len(b["polyvert"])
            pmadr[va:va + vn] = b["polymapadr"] + n_pmap
            pmnum[va:va + vn] = b["polymapnum"]
            pmap.append(b["polymap"])
            n_pmap += len(b["polymap"])
            done += 1
    new = {"mesh_graph": np.concatenate(graph), "mesh_graphadr": graphadr, "mesh_polynum": polynum,
           "mesh_polyadr": polyadr, "mesh_polynormal": np.concatenate(normal), "mesh_polyvertadr": np.concatenate(pvadr),
           "mesh_polyvertnum": np.concatenate(pvnum), "mesh_polyvert": np.concatenate(pvert),
           "mesh_polymapadr": pmadr, "mesh_polymapnum": pmnum, "mesh_polymap": np.concatenate(pmap)}
    for f, v in new.items():
        old = getattr(wm, f)
        setattr(wm, f, wp.array(v.astype(old.numpy().dtype) if old.dtype != wp.vec3 else v.astype(np.float32),
                                dtype=old.dtype, device=old.device))
    wm.nmaxpolygon = int(max(int(wm.nmaxpolygon), int(new["mesh_polyvertnum"].max(initial=0))))
    wm.nmaxmeshdeg = int(max(int(wm.nmaxmeshdeg), int(new["mesh_polymapnum"].max(initial=0))))
    if hasattr(sim, "_model_bridge"):
        sim._model_bridge.clear_cache()
    out = {"injected": done, "already_had_hulls": skipped, "max_vert_err": max_err, "nmeshgraph": int(n_graph),
           "nmeshpoly": int(n_poly), "nmaxpolygon": int(wm.nmaxpolygon), "nmaxmeshdeg": int(wm.nmaxmeshdeg)}
    print(f"[swap_collision] hull data for {done} orphan meshes ({skipped} already had it); vertices match to "
          f"{max_err:.1e}; nmaxpolygon {out['nmaxpolygon']}, nmaxmeshdeg {out['nmaxmeshdeg']}", flush=True)
    return out


def patch_assist_inertia():
    """FDR's object assist (envs/commands/motion_tracking/objects/assist.py) maps its rotational PD to a torque through
    the object's inertia, which ObjectAssistMixin._obj_inertia_body reads from world 0's model row. That is right in
    multi-object mode (one body per object) but not in swap mode, where the one body holds a different object in
    every world: every world then got world 0's inertia. For an object much less inertial than world 0's about some
    axis the explicit PD overshoots and the object spins up within a few steps (the round-10 metal case, prp_1, on the
    device's inertia: 131 deg and 256 rad/s in 4 control steps, in every world). In swap mode this returns the
    per-object table instead, (n_objects, 3, 3) in the body frame, which the caller indexes by each env's active
    object, as it does in multi-object mode. Other modes keep FDR's code."""
    import torch
    from envs.commands.motion_tracking.objects.assist import ObjectAssistMixin

    orig = ObjectAssistMixin._obj_inertia_body
    if getattr(orig, "_cfhoi_swap_aware", False):
        return

    def _obj_inertia_body(self, side: str) -> torch.Tensor:
        tables = getattr(self, "_swap_tables", None)
        if tables is None or "binertia" not in tables:
            return orig(self, side)
        cached = self._obj_I_body_cache.get(side)
        if cached is not None:
            return cached
        from mjlab.utils.lab_api.math import matrix_from_quat

        diag = tables["binertia"].to(self.device, dtype=torch.float32)       # (S, 3) principal moments
        R = matrix_from_quat(tables["biquat"].to(self.device, dtype=torch.float32))   # principal -> body
        out = R @ torch.diag_embed(diag) @ R.transpose(-1, -2)
        out = out[0] if out.shape[0] == 1 else out
        self._obj_I_body_cache[side] = out
        return out

    _obj_inertia_body._cfhoi_swap_aware = True
    ObjectAssistMixin._obj_inertia_body = _obj_inertia_body


def register():
    """Make this module importable as simulator.mujoco.swap_collision (FDR's import in multi_object.py), and make
    FDR's object assist swap-aware (patch_assist_inertia). Needs FDR's src on sys.path."""
    import sys
    import types

    pkg = sys.modules.get("simulator.mujoco")
    if pkg is None:
        pkg = types.ModuleType("simulator.mujoco")
        pkg.__path__ = []
        sys.modules["simulator.mujoco"] = pkg
    sys.modules["simulator.mujoco.swap_collision"] = sys.modules[__name__]
    pkg.swap_collision = sys.modules[__name__]
    patch_assist_inertia()
