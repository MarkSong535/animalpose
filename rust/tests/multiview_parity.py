"""Reference numbers for `rust/tests/multiview_parity.rs`.

Runs the Python `animalpose.multiview` modules on fixed, closed-form inputs (no random
generator: noise is ``amp * sin(1.7 * i + seed)``) and writes `multiview_parity.json`, which the
Rust test reads. The Rust side rebuilds the same scenes with the same formulas
(`data/synthetic.py:body_pose` and `camera_from_view` are bit-exact in Rust).

    PYTHONPATH=src python rust/tests/multiview_parity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from animalpose.data.synthetic import VIEW_PRESETS, body_pose, camera_from_view
from animalpose.multiview.calibration import bundle_adjust, calibrate_dlt, reprojection_errors, solve_extrinsics
from animalpose.multiview.camera import Camera, essential_matrix, rodrigues, rotation_to_rvec
from animalpose.multiview.fusion import ViewTrack, epipolar_cost, fuse_identities, triangulate_identities
from animalpose.multiview.symmetry import (
    F_SHARED_X,
    epipolar_distance,
    fundamental_from_cameras,
    resolve_left_right_cameras,
)
from animalpose.multiview.triangulation import triangulate_dlt, triangulate_robust
from animalpose.skeleton import CANONICAL

ARENA = 100.0
K = 25


def cameras(views=("top", "side", "angled"), size=(640, 480), distorted=True):
    cams = [camera_from_view(ARENA, *VIEW_PRESETS[v], size=size, name=v) for v in views]
    if distorted and len(cams) > 1:
        cams[1].dist = np.array([-0.1, 0.01, 0.001, -0.002, 0.0])
    return cams


def pose(a: int, t: int) -> np.ndarray:
    """One animal's 25 world keypoints (cm) at frame ``t``: a closed-form walk around the arena."""
    params = np.array([0.2, 0.1, 1.0, 0.0, 0.0, 0.0])
    body = body_pose(params, 0.4 * t, 0.3 * t, 0.1 * np.sin(0.1 * t), 0.2 * t) * 20.0
    h = 0.05 * t + np.pi * a
    c, s = np.cos(h), np.sin(h)
    px, py = 25.0 * np.cos(0.03 * t + 2.0 * a), 25.0 * np.sin(0.03 * t + 2.0 * a)
    out = np.empty((K, 3))
    out[:, 0] = c * body[:, 0] - s * body[:, 1] + px
    out[:, 1] = s * body[:, 0] + c * body[:, 1] + py
    out[:, 2] = body[:, 2]
    return out


def noise(shape, amp: float, seed: float) -> np.ndarray:
    n = int(np.prod(shape))
    return (amp * np.sin(1.7 * np.arange(n) + seed)).reshape(shape)


def tolist(x):
    return np.asarray(x, dtype=np.float64).tolist()


def main() -> None:
    out: dict = {}
    cams = cameras()
    out["cameras"] = [c.to_dict() for c in cams]
    out["camera_json"] = json.dumps([cams[1].to_dict()], indent=2)

    # ---------------------------------------------------------------- camera
    X = pose(0, 7)
    uv, z = cams[1].project(X, return_depth=True)
    out["project"] = {"X": tolist(X), "uv": tolist(uv), "z": tolist(z),
                      "undistorted": tolist(cams[1].undistort_points(uv)),
                      "P": tolist(cams[2].P), "center": tolist(cams[2].center),
                      "E01": tolist(essential_matrix(cams[0], cams[1]))}
    rv = np.array([0.3, -0.7, 1.1])
    half = np.array([0.6, 0.0, 0.8]) * np.pi
    out["rodrigues"] = {"rvec": tolist(rv), "R": tolist(rodrigues(rv)),
                        "back": tolist(rotation_to_rvec(rodrigues(rv))),
                        "half": tolist(half), "half_back": tolist(rotation_to_rvec(rodrigues(half)))}
    look = Camera.look_at([50.0, 40.0, 60.0], [0.0, 0.0, 0.0], focal=600.0, size=(320, 240),
                          dist=[0.05, -0.01, 0.0, 0.0, 0.002], name="look")
    out["look_at"] = look.to_dict()

    # ---------------------------------------------------------------- triangulation
    X = pose(1, 11)
    uv = np.stack([c.project(X) for c in cams]) + noise((3, K, 2), 0.4, 0.5)
    w = 0.5 + 0.5 * np.abs(noise((3, K), 1.0, 2.0))
    uv[2, 3] = np.nan  # one point seen by two views only
    uv[1, 4] = np.nan
    uv[2, 4] = np.nan  # one point seen by a single view: NaN
    out["dlt"] = {"uv": np.nan_to_num(uv, nan=-1e9).tolist(), "w": tolist(w), "X": np.nan_to_num(
        triangulate_dlt(cams, uv, w), nan=-1e9).tolist(), "X_unweighted": np.nan_to_num(
        triangulate_dlt(cams, uv), nan=-1e9).tolist()}

    four = cams + [Camera.look_at([ARENA, 0, ARENA], [0, 0, 0], focal=500, size=(640, 480), name="extra")]
    uv4 = np.stack([c.project(X) for c in four]) + noise((4, K, 2), 0.3, 1.0)
    uv4[1, :5] += 60.0  # gross error in one view for some points
    conf4 = 0.4 + 0.6 * np.abs(noise((4, K), 1.0, 3.0))
    conf4[3, 7] = 0.1  # below conf_thresh
    Xr, err, used = triangulate_robust(four, uv4, conf4)
    out["robust"] = {"extra": four[3].to_dict(), "uv": tolist(uv4), "conf": tolist(conf4), "X": tolist(Xr),
                     "err": tolist(err), "used": used.astype(int).tolist()}

    # ---------------------------------------------------------------- calibration
    pts = np.concatenate([pose(a, t) for a in range(2) for t in (0, 20, 40)])[::5].copy()  # 30 points (contiguous: cv2 needs it)
    obs = cams[2].project(pts) + noise((len(pts), 2), 0.2, 4.0)
    est = calibrate_dlt(pts, obs, cams[2].size, "angled")
    exact = calibrate_dlt(pts, cams[2].project(pts), cams[2].size, "angled")
    out["dlt_calibration"] = {"X": tolist(pts), "x": tolist(obs), "K": tolist(est.K), "R": tolist(est.R),
                              "t": tolist(est.t), "err": tolist(reprojection_errors([est], pts, obs[None])[0]),
                              "exact_K": tolist(exact.K), "exact_R": tolist(exact.R), "exact_t": tolist(exact.t)}
    obs1 = cams[1].project(pts) + noise((len(pts), 2), 0.2, 5.0)
    pnp = solve_extrinsics(cams[1].K, cams[1].dist, pts, obs1, cams[1].size, "side")
    planar = pts.copy()
    planar[:, 2] = 0.0
    obs_planar = cams[0].project(planar) + noise((len(pts), 2), 0.1, 6.0)
    pnp_planar = solve_extrinsics(cams[0].K, cams[0].dist, planar, obs_planar, cams[0].size, "top")
    out["pnp"] = {"x": tolist(obs1), "R": tolist(pnp.R), "t": tolist(pnp.t),
                  "planar_X": tolist(planar), "planar_x": tolist(obs_planar),
                  "planar_R": tolist(pnp_planar.R), "planar_t": tolist(pnp_planar.t)}

    # cv2.calibrateCamera on board corners (the solver behind calibrate_intrinsics_checkerboard).
    import cv2

    board = Camera.look_at([0, 0, 1], [0, 0, 0], focal=600.0, size=(520, 400),
                           dist=[-0.08, 0.02, 0.0005, -0.0004, 0.0])
    objp = np.zeros((7 * 5, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2) * 3.0
    obj_pts, img_pts = [], []
    for i, (az, el) in enumerate(((-25, 45), (20, 58), (-5, 35), (35, 68), (-40, 62), (8, 80))):
        az, el = np.radians(az), np.radians(el)
        center = np.array([9.0, 6.0, 0.0])
        eye = center + 40.0 * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        view = Camera.look_at(eye, center, focal=600.0, size=(520, 400), dist=board.dist)
        uv = view.project(objp.astype(np.float64)) + noise((35, 2), 0.05, 20.0 + i)
        obj_pts.append(objp)
        img_pts.append(uv.astype(np.float32)[:, None])
    rms, Kc, distc, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (520, 400), None, None)
    out["intrinsics"] = {"obj": [o.astype(np.float64).tolist() for o in obj_pts],
                         "img": [i[:, 0].astype(np.float64).tolist() for i in img_pts],
                         "K": tolist(Kc), "dist": tolist(distc.ravel()), "rms": float(rms)}

    Xb = np.concatenate([pose(a, t) for a in range(2) for t in range(0, 60, 3)])  # 1000 points
    obs = np.stack([c.project(Xb) for c in cams]) + noise((3, len(Xb), 2), 0.3, 7.0)
    obs[0, :40] = np.nan  # some points unseen by one view
    noisy = [Camera(c.K, rodrigues(noise((3,), 0.02, 8.0 + i)) @ c.R, c.t + 0.5, c.dist, c.size, c.name)
             if i else c for i, c in enumerate(cams)]
    X0 = triangulate_dlt(noisy, obs)
    refined, Xr, info = bundle_adjust(noisy, X0, obs)
    fixed_pts, _, info_fixed = bundle_adjust(noisy, Xb, obs, refine_points=False)
    out["bundle"] = {"noisy": [c.to_dict() for c in noisy], "info": info, "info_fixed_points": info_fixed,
                     "refined": [c.to_dict() for c in refined], "fixed_points": [c.to_dict() for c in fixed_pts],
                     "X0_head": tolist(X0[:5])}

    # ---------------------------------------------------------------- fusion
    n = 60
    t = np.arange(n)
    world = np.stack([np.stack([pose(a, f) for f in range(n)]) for a in range(2)])  # (2, T, K, 3)
    tracks = []
    for v, cam in enumerate(cams):
        for a in range(2):
            kp = cam.project(world[a]) + noise((n, K, 2), 0.5, 10.0 * v + a)
            conf = np.ones((n, K))
            if v == 1 and a == 0:  # animal 0 fragments into two tracks in view 1
                tracks.append(ViewTrack(v, 100, t[:25], kp[:25], conf[:25]))
                tracks.append(ViewTrack(v, 101, t[30:], kp[30:], conf[30:]))
            else:
                tracks.append(ViewTrack(v, (a + v) % 2 + 10 * v, t, kp, conf))
    ids = fuse_identities(tracks, cams)
    costs = [[epipolar_cost(a, b, cams[a.view], cams[b.view]) if a.view != b.view else None for b in tracks]
             for a in tracks]
    tri = triangulate_identities(tracks, cams, ids, n)
    out["fusion"] = {"tracks": [[tr.view, tr.track_id] for tr in tracks],
                     "ids": [[k[0], k[1], g] for k, g in ids.items()],
                     "costs": [[-1.0 if c is None else (1e300 if np.isinf(c) else c) for c in row] for row in costs],
                     "animals": {str(g): {"max_err_cm": float(np.nanmax(np.linalg.norm(
                         a["keypoints3d"] - world[[0 if ids[(0, 0)] == g else 1][0]], axis=-1))),
                         "mean_reproj": float(np.nanmean(a["reproj_error"])),
                         "n_views_sum": int(a["n_views"].sum()),
                         "frame33": tolist(a["keypoints3d"][33]),
                         "frame27": np.nan_to_num(a["keypoints3d"][27], nan=-1e9).tolist()}
                         for g, a in tri.items()}}
    ids_max = fuse_identities(tracks, cams, max_cost=0.0, max_animals=2)
    out["fusion"]["ids_forced"] = [[k[0], k[1], g] for k, g in ids_max.items()]

    # ---------------------------------------------------------------- symmetry
    X = pose(0, 40)
    kp = np.stack([c.project(X) for c in cams])
    conf = np.ones((3, K))
    conf[2] *= 0.5  # the swapped view is the least confident one
    bad = kp.copy()
    bad[2] = bad[2][CANONICAL.flip_index]
    fixed, conf_out, swapped = resolve_left_right_cameras(bad, conf, cams)
    F01 = fundamental_from_cameras(cams[0], cams[1])
    pa = kp[0, :6]
    pb = kp[1, :6] + noise((6, 2), 2.0, 11.0)
    out["symmetry"] = {"bad": tolist(bad), "conf": tolist(conf), "fixed": tolist(fixed),
                       "swapped": swapped.astype(int).tolist(), "F01": tolist(F01),
                       "pa": tolist(pa), "pb": tolist(pb), "dist": tolist(epipolar_distance(F01, pa, pb)),
                       "shared_x": tolist(epipolar_distance(F_SHARED_X, np.array([[5.0, 1.0], [3.0, 2.0]]),
                                                            np.array([[5.0, 90.0], [7.0, 1.0]])))}

    two = cameras(("top", "side"), distorted=False)
    n = 24
    uv = np.stack([np.stack([c.project(pose(0, f)) for f in range(n)]) for c in two])
    uv[1] = uv[1][:, CANONICAL.flip_index]
    conf = np.ones((2, n, K))
    conf[1] *= 0.6
    tracks = [ViewTrack(v, 0, np.arange(n), uv[v], conf[v]) for v in range(2)]
    ids = {(v, 0): 0 for v in range(2)}
    kw = dict(conf_thresh=0.3, reproj_thresh=1e9)
    truth = np.stack([pose(0, f) for f in range(n)])
    naive = triangulate_identities(tracks, two, ids, n, **kw)[0]["keypoints3d"]
    fixed = triangulate_identities(tracks, two, ids, n, resolve_left_right=True, **kw)[0]["keypoints3d"]
    out["mirror"] = {"naive_max": float(np.nanmax(np.linalg.norm(naive - truth, axis=-1))),
                     "fixed_max": float(np.nanmax(np.linalg.norm(fixed - truth, axis=-1))),
                     "naive_frame5": tolist(naive[5])}

    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
