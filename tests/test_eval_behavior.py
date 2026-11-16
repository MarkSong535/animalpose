"""Session-wise behavior scoring and post-processing variants (:mod:`animalpose.eval.behavior`).

Checks the contract of the module: one forward pass per session, every smoothing/prior variant scored
from it, smoothing applied inside a session and never across its boundary, and ``protocol_metrics``
restricted to labelled frames of the candidate classes -- all against confusion matrices and macro-F1
values worked out by hand (a scripted classifier makes those exact). Not checked: whether smoothing or
prior adaptation actually helps on real recordings, which needs the downloaded labelled sessions.
"""

import numpy as np
import pytest
import torch
from torch import nn

from animalpose.behavior.postprocess import adapt_prior, median_smooth
from animalpose.data.behavior_sets import remap_labels
from animalpose.data.synthetic import make_cameras, simulate
from animalpose.data.tracks import IGNORE, Session
from animalpose.eval.behavior import (
    PROTOCOLS,
    _softmax,
    behavior_report,
    protocol_metrics,
    protocol_scores,
    session_logits,
)
from animalpose.labels import BEHAVIORS, BEHAVIORS_EXT, LABELLED, SHARED
from animalpose.train.checkpoint import build_model
from animalpose.train.loops import train_behavior

PAIR = ["walking", "rearing"]
# Scripted bout: walking with a two-frame flicker at 10-11, rearing from 15; the second session is
# walking with the model confidently wrong from its second frame on.
PRED = {30: np.r_[np.zeros(10, int), [1, 1], np.zeros(3, int), np.ones(15, int)],
        6: np.r_[[0], np.ones(5, int)]}
GT = {30: np.r_[np.zeros(15, int), np.ones(15, int)], 6: np.zeros(6, int)}
RAW_CM, SMOOTH_CM = [[14, 7], [0, 15]], [[16, 5], [0, 15]]


class _Scripted(nn.Module):
    """Returns fixed logits per session length and counts how often it was run."""

    def __init__(self, classes):
        super().__init__()
        self.classes, self.n_calls = tuple(classes), 0
        z = 10.0 * (2.0 * np.eye(len(classes)) - 1.0)  # +10 on the scripted class, -10 on the rest
        self.logits = {t: z[np.asarray(p)] for t, p in PRED.items()}
        self.probe = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x, mask):
        self.n_calls += 1
        return torch.from_numpy(self.logits[int(x.shape[1])])[None].to(x.device, x.dtype)


def _stub_sessions():
    sim = simulate(2, 30, "mouse", seed=11)
    cam = make_cameras(sim.arena_cm, ("top",))[0]
    sessions = []
    for t, labels in GT.items():
        xy = cam.project(sim.kpts3d[0, : len(labels)]).astype(np.float64)
        sessions.append(Session(f"s{t}", "top", xy, np.full(xy.shape[:2], 0.9), labels, 30.0))
    return sessions


def _onehot(pred, n_classes: int) -> np.ndarray:
    p = np.full((len(pred), n_classes), 0.05)
    p[np.arange(len(pred)), np.asarray(pred)] = 0.9
    return p


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    out = tmp_path_factory.mktemp("behavior-eval")
    cfg = {"device": "cpu", "output": str(out / "behavior.pt"), "model": {"dims": 2, "hidden": 16},
           "data": {"window": 32, "pool_sims": 2, "pool_frames": 120},
           "train": {"steps": 60, "batch_size": 4, "log_every": 0, "workers": 0}}
    trained, history = train_behavior(cfg)
    assert np.isfinite([h["total"] for h in history]).all()
    return trained


@pytest.fixture(scope="module")
def sessions():
    """Real tracks of the simulator, projected through one camera, with the simulator's own labels."""
    sim = simulate(3, 200, "mouse", seed=7)
    cam = make_cameras(sim.arena_cm, ("angled",))[0]
    out = []
    for a in range(sim.n_animals):
        xy = cam.project(sim.kpts3d[a]).astype(np.float64)
        out.append(Session(f"m{a}", "angled", xy, np.full(xy.shape[:2], 0.9), sim.behavior[a].copy(), 30.0))
    return out


def test_session_logits_are_one_row_per_frame_and_respect_the_confidence_mask(model, sessions):
    logits = session_logits(model, sessions)
    assert len(logits) == len(sessions)
    for s, z in zip(sessions, logits):
        assert z.shape == (len(s.labels), len(model.classes)) and np.isfinite(z).all()
    # conf_thresh 1.5 leaves no keypoint observed, which the classifier has to notice.
    assert not np.allclose(session_logits(model, sessions, conf_thresh=1.5)[0], logits[0])


def test_softmax_is_shift_invariant_and_cannot_overflow():
    z = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    assert np.allclose(_softmax(z), torch.softmax(torch.tensor(z), -1).numpy(), atol=1e-7)
    assert np.allclose(_softmax(z), _softmax(z + 500.0)) and np.allclose(_softmax(z).sum(-1), 1.0)
    assert np.allclose(_softmax(np.full((3, 4), 500.0)), 0.25)


def test_protocol_metrics_scores_labelled_candidate_classes_only_with_hand_checked_arithmetic():
    probs = _onehot([1, 2, 2, 1, 4, 0], len(BEHAVIORS))
    probs[:, 0] = 0.9  # the model's own argmax is "resting", which no protocol scores
    gt = np.array([1, 1, 2, 2, 3, 0])  # frame 5 is "resting": labelled, but outside SHARED
    r = protocol_metrics(probs, gt, BEHAVIORS, SHARED)
    assert r["n_frames"] == 5 and sorted(r["per_class_f1"]) == sorted(SHARED)
    assert r["confusion"] == [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
    assert r["per_class_f1"] == {"walking": 0.5, "rearing": 0.5, "grooming": 0.0, "eating": 0.0}
    assert r["macro_f1"] == pytest.approx(1 / 3) and r["accuracy"] == pytest.approx(0.4)
    empty = protocol_metrics(probs, np.full(6, IGNORE), BEHAVIORS, SHARED)
    assert np.isnan([empty["macro_f1"], empty["accuracy"]]).all()
    assert empty["n_frames"] == 0 and empty["per_class_f1"] == {}


def test_frames_outside_the_label_vocabulary_drop_out_of_the_scored_set():
    probs = _onehot([1, 2, 2, 1, 4, 0, 0, 0], len(BEHAVIORS_EXT))
    probs[:, 0] = 0.9
    gt = np.array([1, 1, 2, 2, 3, 0])
    ext = np.r_[remap_labels(gt, BEHAVIORS, BEHAVIORS_EXT), 5, IGNORE]  # one frame coded "other"
    assert protocol_metrics(probs, ext, BEHAVIORS, SHARED, label_classes=BEHAVIORS_EXT)["n_frames"] == 5
    wide = protocol_metrics(probs, ext, BEHAVIORS_EXT, LABELLED, label_classes=BEHAVIORS_EXT)
    assert wide["n_frames"] == 6 and wide["confusion"][LABELLED.index("other")][0] == 1  # predicted walking
    assert wide["per_class_f1"] == {"walking": 0.4, "rearing": 0.5, "grooming": 0.0, "eating": 0.0, "other": 0.0}
    assert wide["macro_f1"] == pytest.approx((0.4 + 0.5) / 4)  # "eating" is absent from gt, so not averaged


def test_report_scores_every_variant_from_one_forward_pass_per_session():
    stub, sessions = _Scripted(PAIR), _stub_sessions()
    rep = behavior_report(stub, sessions, {"both": PAIR}, smoothing=(1, 3, 5, 15), prior_adapt=True)
    assert stub.n_calls == len(sessions)  # the whole grid of variants costs a single pass per session
    assert set(rep) == {v + s for v in ("raw", "smooth3", "smooth5", "smooth15") for s in ("", "+prior")}
    assert {k: rep[k]["both"]["confusion"] for k in ("raw", "smooth3", "smooth5", "smooth15")} == {
        "raw": RAW_CM, "smooth3": RAW_CM, "smooth5": SMOOTH_CM, "smooth15": RAW_CM}
    assert rep["raw"]["both"]["n_frames"] == 36 and rep["raw"]["both"]["accuracy"] == pytest.approx(29 / 36)
    assert rep["raw"]["both"]["macro_f1"] == pytest.approx((4 / 5 + 30 / 37) / 2)  # prec 1/rec 2/3, 30/37
    assert rep["smooth5"]["both"]["accuracy"] == pytest.approx(31 / 36)
    assert rep["smooth5"]["both"]["macro_f1"] > rep["raw"]["both"]["macro_f1"]
    # a 15-frame median removes the flicker too, but now swallows two frames of the walking bout
    assert behavior_report(stub, sessions, {"both": PAIR}, smoothing=(1, 5), prior_adapt=False) == {
        k: rep[k] for k in ("raw", "smooth5")}


def test_bout_smoothing_stays_inside_each_session():
    stub, sessions = _Scripted(PAIR), _stub_sessions()
    labels = np.concatenate([s.labels for s in sessions])
    probs = np.concatenate([_softmax(z) for z in session_logits(stub, sessions)])
    across = median_smooth(probs, 5)  # the wiring the module rules out: sessions are different animals
    assert protocol_metrics(across, labels, stub.classes, PAIR)["confusion"] == [[15, 6], [0, 15]]
    per_session = np.concatenate([median_smooth(_softmax(z), 5) for z in session_logits(stub, sessions)])
    assert protocol_metrics(per_session, labels, stub.classes, PAIR)["confusion"] == SMOOTH_CM
    rep = behavior_report(stub, sessions, {"both": PAIR}, smoothing=(5,), prior_adapt=True)
    assert rep["smooth5"]["both"]["confusion"] == SMOOTH_CM
    smoothed = [median_smooth(_softmax(z), 5) for z in session_logits(stub, sessions)]
    adapted = np.concatenate([adapt_prior(p, max_ratio=4.0)[0] for p in smoothed])
    assert rep["smooth5+prior"]["both"]["confusion"] == protocol_metrics(adapted, labels, stub.classes, PAIR)["confusion"]
    capped = behavior_report(stub, sessions, {"both": PAIR}, smoothing=(5,), prior_max_ratio=1.0)
    assert capped["smooth5+prior"]["both"] == capped["smooth5"]["both"]  # no prior may move at all


def test_report_of_a_trained_classifier_matches_an_independent_recomputation(model, sessions):
    rep = behavior_report(model, sessions, {"shared4": SHARED}, smoothing=(1, 5), prior_adapt=True,
                          label_classes=BEHAVIORS_EXT)
    logits, labels = session_logits(model, sessions), np.concatenate([np.asarray(s.labels) for s in sessions])
    expect = protocol_metrics(np.concatenate([_softmax(z) for z in logits]), labels, model.classes, SHARED,
                              label_classes=BEHAVIORS_EXT)
    assert rep["raw"]["shared4"] == expect
    assert rep["smooth5"]["shared4"]["n_frames"] == expect["n_frames"]
    assert np.isfinite(rep["smooth5"]["shared4"]["macro_f1"])


def test_protocol_scores_drops_protocols_the_model_cannot_be_scored_on(model, sessions):
    assert PROTOCOLS == {"shared4": SHARED, "labelled5": LABELLED}
    scored = protocol_scores(model, sessions)  # labels come in the extended vocabulary by default
    assert set(scored) == {"shared4"}  # this classifier has no "other" class
    assert set(scored["shared4"]) == {"macro_f1", "accuracy", "per_class_f1", "n_frames"}
    rep = behavior_report(model, sessions, {"shared4": SHARED}, smoothing=(1,), prior_adapt=False)
    assert scored["shared4"]["macro_f1"] == pytest.approx(rep["raw"]["shared4"]["macro_f1"])
    wide = build_model("behavior", {"dims": 2, "hidden": 16, "classes": BEHAVIORS_EXT})
    assert set(protocol_scores(wide, sessions)) == set(PROTOCOLS)
    subset = protocol_scores(model, sessions, protocols={"present": ["walking", "eating"]}, label_classes=BEHAVIORS)
    assert list(subset["present"]["per_class_f1"]) == ["walking", "eating"]
