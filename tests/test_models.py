"""Shape, decoding and invariance tests for the four networks."""

import numpy as np
import pytest
import torch

from animalpose.behavior.model import BehaviorTCN, EgocentricFeatures, to_bouts
from animalpose.detector.model import CenterDetector, DetectorExport, build_targets, decode, detector_loss
from animalpose.pose.model import PoseExport, PoseNet, decode_pairs, pose_loss, supcon_loss
from animalpose.pose.model import build_targets as pose_targets
from animalpose.pose.model import decode as pose_decode
from animalpose.temporal.model import TemporalExport, TemporalLSTM, smooth_sequence, temporal_loss


def _logit(p):
    p = p.clamp(1e-6, 1 - 1e-6)
    return torch.log(p / (1 - p))


def test_detector_targets_decode_roundtrip():
    boxes = [torch.tensor([[10.3, 20.7, 60.1, 50.9], [100.0, 30.0, 140.0, 90.0]])]
    tgt = build_targets(boxes, (32, 40), 4)
    heat = torch.where(tgt["heat"] == 1, 10.0, -10.0)
    out_boxes, scores, _ = decode(heat, tgt["size"], tgt["offset"], 4, max_det=5)
    found = out_boxes[0, scores[0] > 0.5]
    assert found.shape[0] == 2
    for b in boxes[0]:
        assert torch.min((found - b).abs().max(-1).values) < 1e-3



@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple Metal (MPS)")
def test_the_export_ranking_runs_on_metal_like_on_the_cpu():
    """MPS has no float64: the tie-breaking ranking moves the scores to the CPU before widening them."""
    boxes = [torch.tensor([[10.3, 20.7, 60.1, 50.9], [100.0, 30.0, 140.0, 90.0]])]
    tgt = build_targets(boxes, (32, 40), 4)
    heat = torch.where(tgt["heat"] == 1, 10.0, -10.0)
    cpu = decode(heat, tgt["size"], tgt["offset"], 4, max_det=5, for_export=True)
    mps = decode(heat.to("mps"), tgt["size"].to("mps"), tgt["offset"].to("mps"), 4, max_det=5, for_export=True)
    assert mps[0].device.type == "mps"
    for a, b in zip(cpu, mps):
        assert torch.allclose(a.float(), b.cpu().float(), atol=1e-4)

def test_detector_forward_loss_backward(tiny_backbone):
    model = CenterDetector(tiny_backbone, num_species=2)
    x = torch.rand(2, 3, 64, 96)
    out = model(x)
    assert out["heat"].shape == (2, 1, 16, 24)
    tgt = build_targets([torch.tensor([[5.0, 5.0, 30.0, 25.0]]), torch.zeros(0, 4)], (16, 24), 4,
                        [torch.tensor([1]), torch.zeros(0)])
    loss = detector_loss(out, tgt)
    loss["total"].backward()
    assert torch.isfinite(loss["total"])
    boxes, scores = DetectorExport(model.eval(), 7)(x)
    assert boxes.shape == (2, 7, 4) and scores.shape == (2, 7)


def test_pose_targets_decode_roundtrip():
    kp = torch.tensor([[[13.3, 40.2], [50.9, 7.6], [0.0, 0.0]]])
    state = torch.tensor([[2, 2, 0]])
    tgt = pose_targets(kp, state, (16, 16), 4)
    assert tgt["weight"][0, 2].item() == 0.0
    pred, conf = pose_decode(_logit(tgt["heat"]), tgt["offset"], 4)
    np.testing.assert_allclose(pred[0, :2].numpy(), kp[0, :2].numpy(), atol=1e-4)
    assert conf[0, 0] > 0.99


def _pair_heat(l_at_b: float, r_at_b: float):
    """Left/right heatmaps that both peak at cell A=(2,2); cell B=(10,10) carries the second paw."""
    p = torch.full((1, 2, 16, 16), 1e-4)
    p[0, 0, 2, 2], p[0, 1, 2, 2] = 0.9, 0.8
    p[0, 0, 10, 10], p[0, 1, 10, 10] = l_at_b, r_at_b
    return _logit(p), torch.zeros(1, 4, 16, 16)


def test_decode_pairs_splits_collapsed_pair_in_preferred_order():
    heat, off = _pair_heat(0.5, 0.6)
    pairs = torch.tensor([[0, 1]])
    kp, _ = pose_decode(heat, off, 4)
    assert torch.equal(kp[0, 0], kp[0, 1])  # plain argmax: both on paw A
    kp, conf = decode_pairs(heat, off, 4, pairs)
    # L(A) + R(B) = 1.5 beats L(B) + R(A) = 1.3: left stays on A, right moves to B.
    assert kp[0, 0].tolist() == [8.0, 8.0] and kp[0, 1].tolist() == [40.0, 40.0]
    assert abs(conf[0, 1].item() - 0.6) < 1e-4
    heat, off = _pair_heat(0.7, 0.2)  # now L(B) + R(A) = 1.5 > 1.1
    kp, _ = decode_pairs(heat, off, 4, pairs)
    assert kp[0, 0].tolist() == [40.0, 40.0] and kp[0, 1].tolist() == [8.0, 8.0]


def test_decode_pairs_keeps_pair_without_second_peak():
    heat, off = _pair_heat(0.02, 0.03)
    kp, _ = decode_pairs(heat, off, 4, torch.tensor([[0, 1]]), min_peak=0.1)
    assert torch.equal(kp[0, 0], kp[0, 1])


def test_pose_context_starts_as_identity(tiny_backbone):
    torch.manual_seed(0)
    plain = PoseNet(25, tiny_backbone, embed_dim=16).eval()
    ctx = PoseNet(25, tiny_backbone, embed_dim=16, context=True).eval()
    ctx.load_state_dict(plain.state_dict(), strict=False)
    x = torch.rand(2, 3, 64, 64)
    assert torch.allclose(plain(x)["heat"], ctx(x)["heat"])


def test_pose_forward_loss_and_vocabulary_extension(tiny_backbone):
    model = PoseNet(25, tiny_backbone, embed_dim=16)
    crops = torch.rand(4, 3, 64, 64)
    out = model(crops)
    assert out["heat"].shape == (4, 25, 16, 16) and out["offset"].shape == (4, 50, 16, 16)
    assert torch.allclose(out["embedding"].norm(dim=1), torch.ones(4), atol=1e-5)
    kp = torch.rand(4, 25, 2) * 63
    state = torch.randint(0, 3, (4, 25))
    loss = pose_loss(out, pose_targets(kp, state, (16, 16), 4), torch.tensor([0, 0, 1, -1]))
    loss["total"].backward()
    assert torch.isfinite(loss["total"])
    model.add_keypoints(3, init_from=[0, 1, 2])
    kps, conf, emb = PoseExport(model.eval())(crops)
    assert kps.shape == (4, 28, 2) and conf.shape == (4, 28)


def test_supcon_prefers_matching_ids():
    ids = torch.tensor([0, 0, 1, 1])
    good = torch.nn.functional.normalize(torch.tensor([[1.0, 0], [1, 0.1], [0, 1], [0.1, 1]]), dim=1)
    bad = good[[0, 2, 1, 3]]
    assert supcon_loss(good, ids) < supcon_loss(bad, ids)


def test_temporal_is_translation_equivariant_and_flags_imputation():
    model = TemporalLSTM(25, 2, hidden=32).eval()
    x = torch.rand(2, 20, 25, 2) * 50
    conf = torch.rand(2, 20, 25)
    obs = (torch.rand(2, 20, 25) > 0.3).float()
    with torch.no_grad():
        a = model(x, conf, obs)["coords"]
        b = model(x + torch.tensor([100.0, -40.0]), conf, obs)["coords"]
        coords, confidence, imputed = TemporalExport(model)(x, conf, obs)
    torch.testing.assert_close(b, a + torch.tensor([100.0, -40.0]), atol=1e-3, rtol=1e-4)
    assert torch.equal(imputed, 1 - obs)
    assert ((confidence > 0) & (confidence <= 1)).all()
    out = model(x, conf, obs)
    temporal_loss(out, x, torch.ones_like(obs))["total"].backward()


@pytest.mark.parametrize("bidirectional", [False, True])
def test_kernel_prior_is_one_sided_for_causal_models(bidirectional):
    # Online use runs windows that end at the current frame; within a window the
    # causal prior and LSTM must not read later frames.
    model = TemporalLSTM(25, 2, hidden=16, bidirectional=bidirectional, prior_bandwidth=2.0)
    xn, obs = torch.rand(1, 30, 25, 2), (torch.rand(1, 30, 25) > 0.3).float()
    xn2, obs2 = xn.clone(), obs.clone()
    xn2[:, 20:] += 5.0
    obs2[:, 20:] = 1.0
    a, _ = model.kernel_prior(xn, obs)
    b, _ = model.kernel_prior(xn2, obs2)
    assert torch.allclose(a[:, :20], b[:, :20]) != bidirectional


def test_kernel_prior_matches_observed_mean():
    model = TemporalLSTM(25, 2, hidden=16, prior_bandwidth=1e6)
    xn = torch.rand(1, 10, 25, 2)
    obs = (torch.rand(1, 10, 25) > 0.3).float()
    prior, support = model.kernel_prior(xn, obs)
    expected = (xn * obs.unsqueeze(-1)).sum(1) / obs.sum(1).clamp(min=1).unsqueeze(-1)
    has = obs.sum(1) > 0
    torch.testing.assert_close(prior[:, 0][has], expected[has], atol=1e-4, rtol=1e-4)


def test_smooth_sequence_handles_long_and_short_tracks():
    model = TemporalLSTM(25, 3, hidden=16)
    for t in (10, 150):
        x = np.random.rand(t, 25, 3).astype(np.float32)
        obs = np.random.rand(t, 25) > 0.2
        c, conf, imp = smooth_sequence(model, x, np.ones((t, 25)), obs, window=64, hop=32)
        assert c.shape == (t, 25, 3) and conf.shape == imp.shape == (t, 25)
        assert np.isfinite(c).all()


def _rigid(x, angle, shift):
    c, s = np.cos(angle), np.sin(angle)
    r = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
    return x @ r.T + torch.tensor(shift, dtype=torch.float32)


def test_behavior_features_are_rigid_and_scale_invariant():
    from animalpose.data.synthetic import simulate

    sim = simulate(1, 40, "mouse", seed=0)
    x = torch.from_numpy(sim.kpts3d[0, :, :, :2]).float()[None]
    mask = torch.ones(1, 40, 25)
    feats = EgocentricFeatures(25, 2)
    a = feats(x, mask)
    b = feats(_rigid(x, 1.1, [30.0, -12.0]) * 2.5, mask)
    torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)


def test_behavior_tcn_any_length(tiny_backbone):
    model = BehaviorTCN(25, 3, hidden=16).eval()
    for t in (7, 130):
        assert model(torch.rand(2, t, 25, 3), torch.ones(2, t, 25)).shape == (2, t, 5)


def test_to_bouts_merges_short_runs():
    labels = [0] * 10 + [1] * 2 + [0] * 10 + [2] * 8
    assert to_bouts(labels, min_len=5) == [(0, 0, 22), (2, 22, 30)]


def test_linear_prior_matches_numpy_interpolation():
    from animalpose.eval.metrics import linear_interpolate

    model = TemporalLSTM(25, 2, hidden=16, prior_bandwidth=2.0, prior="linear")
    x = torch.rand(2, 30, 25, 2)
    obs = (torch.rand(2, 30, 25) > 0.5).float()
    got = model.linear_prior(x * obs[..., None], obs).numpy()
    ref = np.nan_to_num(np.stack([linear_interpolate((x[i] * obs[i, ..., None]).numpy(), obs[i].numpy() > 0)
                                  for i in range(2)]))
    np.testing.assert_allclose(got, ref, atol=1e-5)


def test_residual_guard_falls_back_to_prior():
    model = TemporalLSTM(25, 2, hidden=16, prior_bandwidth=2.0, prior="hybrid").eval()
    with torch.no_grad():
        model.head[-1].bias.fill_(100.0)  # force absurd corrections
        x = torch.rand(1, 20, 25, 2) * 50
        obs = (torch.rand(1, 20, 25) > 0.3).float()
        out = model(x, torch.ones(1, 20, 25), obs)
    center, scale = model.normalizer(torch.nan_to_num(x) * obs[..., None], obs)
    xn = (x - center) / scale * obs[..., None]
    k, _ = model.kernel_prior(xn, obs)
    prior = obs[..., None] * k + (1 - obs[..., None]) * model.linear_prior(xn, obs)
    torch.testing.assert_close(out["coords_norm"], prior)
    assert (out["log_sigma"] == 3.0).all()
