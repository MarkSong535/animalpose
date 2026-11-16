import numpy as np

from animalpose.data.synthetic import make_cameras, random_camera, render, render_video, simulate
from animalpose.labels import BEHAVIORS


def test_simulation_shapes_and_behaviors():
    sim = simulate(3, 400, "rat", seed=1)
    assert sim.kpts3d.shape == (3, 400, 25, 3)
    assert np.isfinite(sim.kpts3d).all()
    assert (sim.kpts3d[..., 2] >= 0).all()
    assert len(np.unique(sim.behavior)) >= 3
    assert sim.behavior.max() < len(BEHAVIORS)
    # Animals stay inside the arena.
    assert np.abs(sim.kpts3d[..., :2]).max() < sim.arena_cm


def test_render_views_visibility_and_boxes():
    sim = simulate(2, 30, "mouse", seed=2)
    for cam in make_cameras(sim.arena_cm) + [random_camera(sim.arena_cm, np.random.default_rng(0))]:
        f = render(sim, cam, 25, 0)
        h, w = f.image.shape[:2]
        assert (w, h) == cam.size
        for a in np.nonzero(f.present)[0]:
            x0, y0, x1, y1 = f.boxes[a]
            vis = f.keypoints[a][f.visible[a]]
            assert ((vis[:, 0] >= x0 - 2) & (vis[:, 0] <= x1 + 2) & (vis[:, 1] >= y0 - 2) & (vis[:, 1] <= y1 + 2)).all()


def test_overlapping_animals_occlude_each_other():
    sim = simulate(2, 5, "mouse", seed=3)
    sim.kpts3d[1] = sim.kpts3d[0] + np.array([0.5, 0.0, 0.5])  # same place, slightly above
    f = render(sim, make_cameras(sim.arena_cm, ("top",))[0], 2, 0)
    assert f.visible[0].sum() < f.visible[1].sum()


def test_render_video_static_background():
    sim = simulate(1, 4, "mouse", seed=4)
    frames = render_video(sim, make_cameras(sim.arena_cm, ("angled",))[0])
    assert len(frames) == 4 and frames[0].image.dtype == np.uint8
