import numpy as np

from animalpose.data.synthetic import make_cameras, simulate
from animalpose.multiview.calibration import bundle_adjust, calibrate_dlt, reprojection_errors
from animalpose.multiview.camera import Camera, essential_matrix, rodrigues
from animalpose.multiview.fusion import ViewTrack, fuse_identities, triangulate_identities
from animalpose.multiview.triangulation import triangulate_dlt, triangulate_robust


def _scene():
    sim = simulate(2, 60, "rat", seed=3)
    cams = make_cameras(sim.arena_cm, ("top", "side", "angled"), size=(640, 480))
    cams[1].dist = np.array([-0.1, 0.01, 0, 0, 0])  # one distorted lens
    return sim, cams


def test_distortion_roundtrip():
    _, cams = _scene()
    X = np.random.default_rng(0).uniform(-20, 20, (50, 3))
    uv = cams[1].project(X)
    xn = cams[1].undistort_points(uv)
    Xc = cams[1].to_camera(X)
    np.testing.assert_allclose(xn, Xc[:, :2] / Xc[:, 2:], atol=1e-6)


def test_dlt_calibration_recovers_camera():
    sim, cams = _scene()
    cam = cams[2]
    X = np.random.default_rng(1).uniform(-sim.arena_cm / 2, sim.arena_cm / 2, (30, 3)) * [1, 1, 0.3]
    est = calibrate_dlt(X, cam.project(X), cam.size)
    np.testing.assert_allclose(est.K, cam.K, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(est.center, cam.center, atol=1e-3)
    assert reprojection_errors([est], X, cam.project(X)[None]).max() < 1e-4


def test_epipolar_constraint():
    _, cams = _scene()
    X = np.random.default_rng(2).uniform(-10, 10, (20, 3))
    E = essential_matrix(cams[0], cams[2])
    xa = np.c_[cams[0].undistort_points(cams[0].project(X)), np.ones(20)]
    xb = np.c_[cams[2].undistort_points(cams[2].project(X)), np.ones(20)]
    assert np.abs(np.sum(xb * (xa @ E.T), axis=1)).max() < 1e-9


def test_triangulation_exact_and_robust_to_an_outlier_view():
    sim, cams = _scene()
    X = sim.kpts3d[0, 30]
    uv = np.stack([c.project(X) for c in cams])
    np.testing.assert_allclose(triangulate_dlt(cams, uv), X, atol=1e-6)
    four = cams + [Camera.look_at([sim.arena_cm, 0, sim.arena_cm], [0, 0, 0], focal=500, size=(640, 480))]
    uv4 = np.stack([c.project(X) for c in four])
    uv4[1, :5] += 60.0  # gross error in one view for some points
    Xr, err, used = triangulate_robust(four, uv4)
    np.testing.assert_allclose(Xr, X, atol=1e-4)
    assert not used[1, :5].any() and used[1, 5:].all()


def test_bundle_adjust_recovers_perturbed_extrinsics():
    sim, cams = _scene()
    X = sim.kpts3d[:, ::3].reshape(-1, 3)
    obs = np.stack([c.project(X) for c in cams]) + np.random.default_rng(0).normal(0, 0.3, (3, len(X), 2))
    noisy = [Camera(c.K, rodrigues(np.random.default_rng(i).normal(0, 0.02, 3)) @ c.R, c.t + 0.5, c.dist, c.size)
             if i else c for i, c in enumerate(cams)]
    X0 = triangulate_dlt(noisy, obs)
    refined, Xr, info = bundle_adjust(noisy, X0, obs)
    assert info["rmse_after"] < 1.0 < info["rmse_before"]


def test_cross_view_identity_fusion():
    sim, cams = _scene()
    t = np.arange(sim.n_frames)
    tracks = []
    for v, cam in enumerate(cams):
        perm = np.roll(np.arange(sim.n_animals), v)  # per-view track ids differ from animal ids
        for a in range(sim.n_animals):
            kp = cam.project(sim.kpts3d[a]) + np.random.default_rng(v).normal(0, 0.5, (sim.n_frames, 25, 2))
            tracks.append(ViewTrack(v, int(perm[a]) + 10 * v, t, kp, np.ones((sim.n_frames, 25))))
    ids = fuse_identities(tracks, cams)
    for a in range(sim.n_animals):
        assert len({ids[(v, int(np.roll(np.arange(2), v)[a]) + 10 * v)] for v in range(3)}) == 1
    assert len(set(ids.values())) == sim.n_animals
    out = triangulate_identities(tracks, cams, ids, sim.n_frames)
    # In view 0 the track id equals the animal index.
    errs = [np.nanmax(np.linalg.norm(out[ids[(0, a)]]["keypoints3d"] - sim.kpts3d[a], axis=-1))
            for a in range(sim.n_animals)]
    assert max(errs) < 0.5  # cm


def test_fusion_stitches_a_fragmented_single_view_track():
    sim, cams = _scene()
    t = np.arange(sim.n_frames)
    tracks = []
    for v, cam in enumerate(cams):
        for a in range(sim.n_animals):
            kp = cam.project(sim.kpts3d[a])
            if v == 1 and a == 0:  # animal 0 fragments into two tracks in view 1
                tracks.append(ViewTrack(v, 100, t[:25], kp[:25], np.ones((25, 25))))
                tracks.append(ViewTrack(v, 101, t[30:], kp[30:], np.ones((sim.n_frames - 30, 25))))
            else:
                tracks.append(ViewTrack(v, a, t, kp, np.ones((sim.n_frames, 25))))
    ids = fuse_identities(tracks, cams)
    assert len(set(ids.values())) == sim.n_animals
    assert ids[(1, 100)] == ids[(1, 101)] == ids[(0, 0)] != ids[(0, 1)]


def test_left_right_resolution_restores_swapped_view():
    from animalpose.multiview.symmetry import F_SHARED_X, epipolar_distance, resolve_left_right_cameras
    from animalpose.skeleton import CANONICAL

    sim, cams = _scene()
    X = sim.kpts3d[0, 40]
    kp = np.stack([c.project(X) for c in cams])
    conf = np.ones((3, 25))
    conf[2] *= 0.5  # the swapped view is the least confident one
    bad = kp.copy()
    bad[2] = bad[2][CANONICAL.flip_index]
    fixed, _, swapped = resolve_left_right_cameras(bad, conf, cams)
    np.testing.assert_allclose(fixed, kp, atol=1e-6)
    assert swapped[2].any() and not swapped[:2].any()
    # Shared-x geometry (mirror rig): equal x means zero epipolar distance.
    assert epipolar_distance(F_SHARED_X, np.array([[5.0, 1.0]]), np.array([[5.0, 90.0]]))[0] < 1e-9
