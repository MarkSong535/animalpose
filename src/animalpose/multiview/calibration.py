"""Camera calibration.

* :func:`calibrate_dlt` — full projection matrix from >= 6 known 3D-2D
  correspondences (e.g. arena corners, a calibration object), decomposed
  into K, R, t.
* :func:`calibrate_intrinsics_checkerboard` — OpenCV checkerboard intrinsics.
* :func:`solve_extrinsics` — PnP for a camera with known intrinsics.
* :func:`bundle_adjust` — joint refinement of all extrinsics (and optionally
  3D points) from multi-view 2D observations such as matched animal
  keypoints, in the spirit of Anipose's animal-based calibration refinement.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from animalpose.multiview.camera import Camera, rodrigues, rotation_to_rvec


def _normalize_2d(x):
    c = x.mean(0)
    s = np.sqrt(2) / np.mean(np.linalg.norm(x - c, axis=1))
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    return T


def _normalize_3d(X):
    c = X.mean(0)
    s = np.sqrt(3) / np.mean(np.linalg.norm(X - c, axis=1))
    T = np.diag([s, s, s, 1.0])
    T[:3, 3] = -s * c
    return T


def calibrate_dlt(X: np.ndarray, x: np.ndarray, size=(640, 480), name: str = "cam") -> Camera:
    """Normalised DLT + RQ decomposition. ``X (N, 3)``, ``x (N, 2)``, N >= 6, non-coplanar."""
    X, x = np.asarray(X, float), np.asarray(x, float)
    if len(X) < 6:
        raise ValueError("DLT needs at least 6 correspondences")
    T2, T3 = _normalize_2d(x), _normalize_3d(X)
    xn = (np.c_[x, np.ones(len(x))] @ T2.T)[:, :2]
    Xn = np.c_[X, np.ones(len(X))] @ T3.T
    A = []
    for (u, v), P in zip(xn, Xn):
        A.append(np.r_[np.zeros(4), -P, v * P])
        A.append(np.r_[P, np.zeros(4), -u * P])
    _, _, vt = np.linalg.svd(np.asarray(A))
    Pn = vt[-1].reshape(3, 4)
    P = np.linalg.inv(T2) @ Pn @ T3
    M = P[:, :3]
    # RQ decomposition via QR of the flipped matrix.
    flip = np.flipud(np.eye(3))
    q, r = np.linalg.qr((flip @ M).T)
    K = flip @ r.T @ flip
    R = flip @ q.T
    D = np.diag(np.sign(np.diag(K)))
    K, R = K @ D, D @ R
    t = np.linalg.solve(K, P[:, 3])
    if np.linalg.det(R) < 0:
        R, t = -R, -t
    K = K / K[2, 2]
    return Camera(K, R, t, np.zeros(5), tuple(size), name)


def calibrate_intrinsics_checkerboard(images, pattern=(9, 6), square: float = 1.0):
    """OpenCV checkerboard calibration. Returns ``(K, dist, rms)``."""
    import cv2

    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square
    obj_pts, img_pts, shape = [], [], None
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        shape = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, pattern)
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
            obj_pts.append(objp)
            img_pts.append(corners)
    if len(obj_pts) < 3:
        raise ValueError(f"checkerboard found in only {len(obj_pts)} images; need >= 3")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, shape, None, None)
    return K, dist.ravel(), rms


def solve_extrinsics(K, dist, X, x, size=(640, 480), name="cam") -> Camera:
    import cv2

    ok, rvec, tvec = cv2.solvePnP(np.asarray(X, np.float64), np.asarray(x, np.float64), np.asarray(K, np.float64),
                                  np.asarray(dist, np.float64), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    return Camera(K, rodrigues(rvec.ravel()), tvec.ravel(), dist, tuple(size), name)


def reprojection_errors(cameras: list[Camera], X: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """``obs (V, N, 2)`` with NaN for missing -> per-observation pixel error ``(V, N)``."""
    return np.stack([np.linalg.norm(c.project(X) - o, axis=-1) for c, o in zip(cameras, obs)])


def bundle_adjust(cameras: list[Camera], X: np.ndarray, obs: np.ndarray, fix_first: bool = True,
                  refine_points: bool = True, loss: str = "huber", f_scale: float = 2.0,
                  max_nfev: int = 200) -> tuple[list[Camera], np.ndarray, dict]:
    """Refine extrinsics (and points) to minimise reprojection error.

    ``obs (V, N, 2)`` pixel observations with NaN where a point is not seen.
    The first camera fixes the gauge when ``fix_first``; the overall scale is
    fixed by keeping the 3D points' initial scale if points are not refined,
    otherwise by the remaining cameras' initial baseline (a known
    world-frame target should be included for metric scale).
    """
    v, n = obs.shape[:2]
    seen = np.isfinite(obs).all(-1)
    cam_idx, pt_idx = np.nonzero(seen)
    free = list(range(1 if fix_first else 0, v))

    def pack(cams, pts):
        p = [np.r_[rotation_to_rvec(cams[i].R), cams[i].t] for i in free]
        if refine_points:
            p.append(pts.ravel())
        return np.concatenate(p) if p else np.zeros(0)

    def unpack(p):
        cams = [Camera(c.K, c.R, c.t, c.dist, c.size, c.name) for c in cameras]
        for j, i in enumerate(free):
            cams[i].R = rodrigues(p[6 * j:6 * j + 3])
            cams[i].t = p[6 * j + 3:6 * j + 6]
        pts = p[6 * len(free):].reshape(n, 3) if refine_points else X
        return cams, pts

    def residuals(p):
        cams, pts = unpack(p)
        res = np.empty((len(cam_idx), 2))
        for i, cam in enumerate(cams):
            sel = cam_idx == i
            res[sel] = cam.project(pts[pt_idx[sel]]) - obs[i, pt_idx[sel]]
        return res.ravel()

    n_par = 6 * len(free) + (3 * n if refine_points else 0)
    sparsity = lil_matrix((2 * len(cam_idx), n_par), dtype=int)
    rows = np.arange(len(cam_idx))
    for j, i in enumerate(free):
        sel = rows[cam_idx == i]
        for k in range(6):
            sparsity[2 * sel, 6 * j + k] = 1
            sparsity[2 * sel + 1, 6 * j + k] = 1
    if refine_points:
        for k in range(3):
            sparsity[2 * rows, 6 * len(free) + 3 * pt_idx + k] = 1
            sparsity[2 * rows + 1, 6 * len(free) + 3 * pt_idx + k] = 1
    p0 = pack(cameras, X)
    before = np.sqrt(np.mean(residuals(p0).reshape(-1, 2) ** 2) * 2)
    sol = least_squares(residuals, p0, jac_sparsity=sparsity, loss=loss, f_scale=f_scale, max_nfev=max_nfev,
                        x_scale="jac", method="trf")
    cams, pts = unpack(sol.x)
    after = np.sqrt(np.mean(sol.fun.reshape(-1, 2) ** 2) * 2)
    return cams, pts, {"rmse_before": float(before), "rmse_after": float(after), "nfev": int(sol.nfev)}
