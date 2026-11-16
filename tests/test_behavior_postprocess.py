import numpy as np

from animalpose.behavior.postprocess import adapt_prior, fit_temperature, median_smooth, moving_median, temperature_scale


def _onehot(labels, n_classes):
    p = np.full((len(labels), n_classes), 0.02)
    p[np.arange(len(labels)), labels] = 0.94
    return p / p.sum(1, keepdims=True)


def test_moving_median_removes_a_single_frame_flicker_but_keeps_bout_edges():
    labels = np.r_[np.zeros(30, int), np.ones(30, int)]
    p = _onehot(labels, 2)
    p[12] = [0.5, 0.5]
    p[40] = [0.3, 0.7]  # one flickery frame inside the second bout
    sm = median_smooth(p, 5)
    assert sm.shape == p.shape and np.allclose(sm.sum(1), 1.0)
    pred = sm.argmax(1)
    assert np.array_equal(pred, labels)  # flicker gone, boundary still at frame 30
    assert np.allclose(median_smooth(p, 1), p)


def test_moving_median_is_identity_on_a_constant_sequence_and_short_inputs():
    p = _onehot(np.zeros(9, int), 3)
    assert np.allclose(moving_median(p, 5), p)
    short = _onehot(np.zeros(2, int), 3)
    assert np.allclose(moving_median(short, 5), short)


def test_adapt_prior_moves_marginals_towards_the_recordings_own_class_mix():
    # A model trained on a 50/50 mix, applied to a recording that is 85% class 0.
    rng = np.random.default_rng(0)
    true = (rng.random(600) > 0.85).astype(int)
    p = _onehot(true, 2) * 0.9 + 0.05
    p = p / p.sum(1, keepdims=True)
    src = np.array([0.5, 0.5])
    adj, prior = adapt_prior(p, prior_source=src)
    assert adj.shape == p.shape and np.allclose(adj.sum(1), 1.0, atol=1e-5)
    assert prior[0] > prior[1] and abs(prior[0] - 0.85) < 0.15
    assert abs(adj.mean(0)[0] - prior[0]) < 0.05  # the EM fixed point: marginals match the prior
    assert np.abs(np.log(adj / p)).max() <= np.log(16.0) + 1e-3  # capped by max_ratio (renormalization doubles the span)

    soft = _onehot(true, 2) * 0.2 + 0.4  # the same labels, a much less certain model
    soft = soft / soft.sum(1, keepdims=True)
    loose, _ = adapt_prior(soft, prior_source=src, max_ratio=8.0)
    assert (loose.argmax(1) == 0).mean() > (soft.argmax(1) == 0).mean()


def test_adapt_prior_respects_max_ratio_and_is_a_no_op_when_balanced():
    rng = np.random.default_rng(1)
    p = _onehot((rng.random(400) > 0.05).astype(int), 2)
    _, prior = adapt_prior(p, prior_source=np.array([0.5, 0.5]), max_ratio=1.2)
    assert (prior <= 1.2 * np.array([0.5, 0.5]) + 1e-6).all()
    assert (prior >= np.array([0.5, 0.5]) / 1.2 - 1e-6).all()
    balanced = _onehot((rng.random(400) > 0.5).astype(int), 2)
    same, prior = adapt_prior(balanced, prior_source=balanced.mean(0), iters=30)
    assert np.allclose(prior, balanced.mean(0), atol=0.05)
    assert np.abs(same.argmax(1) - balanced.argmax(1)).sum() < 0.02 * len(balanced)


def test_adapt_prior_handles_batches_and_degenerate_inputs():
    p = _onehot(np.r_[np.zeros(20, int), np.ones(20, int)], 2)
    batch = np.stack([p, p])
    adj, prior = adapt_prior(batch)
    assert adj.shape == batch.shape and prior.shape == (2,)
    assert np.allclose(adapt_prior(p[:, :1])[0], p[:, :1])
    empty, prior = adapt_prior(np.zeros((0, 3)))
    assert empty.shape == (0, 3) and np.allclose(prior, 1 / 3)


def test_temperature_scaling_flattens_confident_logits_and_lowers_nll():
    labels = np.r_[np.zeros(40, int), np.ones(40, int)]
    p = _onehot(labels, 2) * 0.6 + 0.2  # underconfident
    t = fit_temperature(p, labels)
    assert t < 1.0  # sharpening helps when the model is underconfident
    sharp = temperature_scale(p, 10.0)
    assert np.allclose(sharp.sum(1), 1.0)
    assert np.allclose(sharp.mean(0), [0.5, 0.5], atol=0.02)  # T -> inf tends to uniform
    assert np.array_equal(sharp.argmax(1), labels)
    over = _onehot(labels, 2)
    over[::5] = 1.0 - over[::5]  # 20 % of frames are confidently wrong
    assert fit_temperature(over, labels) > 1.0
