"""Table-plane fit shared by the viewers (view_choir.py, view_fprecon.py) and the refinement agent (numpy only)."""

import numpy as np


def fit_table(points, up, rng, iters=600, tol=0.01, max_deg=25.0):
    """Largest plane with normal within max_deg of up (RANSAC on a subsample), or None."""
    p = points[rng.choice(len(points), min(len(points), 30000), replace=False)]
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        n = n if n @ up > 0 else -n
        if np.degrees(np.arccos(np.clip(n @ up, -1, 1))) > max_deg:
            continue
        cnt = int((np.abs((p - a) @ n) < tol).sum())
        if cnt > best_n:
            best, best_n = (n, a), cnt
    if best is None:
        return None
    n, a = best
    inl = p[np.abs((p - a) @ n) < tol]
    c = inl.mean(0)
    n = np.linalg.svd(inl - c, full_matrices=False)[2][-1]   # least-squares refit on the inliers
    n = n if n @ up > 0 else -n
    return dict(normal=n, point=c, inlier_frac=len(inl) / len(p),
                tilt_deg=float(np.degrees(np.arccos(np.clip(n @ up, -1, 1)))))
