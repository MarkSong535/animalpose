"""Rule-based cluster naming, checked against clusters a person named by eye on one recording."""

from animalpose.behavior.catalog import canonical
from animalpose.behavior.clusters import name_clusters, naming_agreement

# Median kinematics per cluster, from one recording's clustering.
# A person named six of the eight by watching montage clips; the other two are tracking jumps.
NAMED = {
    "0": {"share_of_train": 0.390, "speed": 0.164, "length": 1.108, "turn": 0.265, "posture": 0.194,
          "head_motion": 0.134, "bend": 0.131, "speed_sd": 0.324, "length_sd": 0.037, "posture_sd": 0.236},
    "4": {"share_of_train": 0.295, "speed": 0.148, "length": 0.835, "turn": 0.374, "posture": 0.235,
          "head_motion": 0.154, "bend": 0.317, "speed_sd": 0.236, "length_sd": 0.041, "posture_sd": 0.300},
    "5": {"share_of_train": 0.172, "speed": 0.662, "length": 1.083, "turn": 0.794, "posture": 0.509,
          "head_motion": 0.329, "bend": 0.160, "speed_sd": 0.677, "length_sd": 0.154, "posture_sd": 0.412},
    "3": {"share_of_train": 0.097, "speed": 0.567, "length": 0.762, "turn": 2.018, "posture": 0.789,
          "head_motion": 0.639, "bend": 0.351, "speed_sd": 0.526, "length_sd": 0.126, "posture_sd": 0.522},
    "2": {"share_of_train": 0.024, "speed": 0.615, "length": 0.388, "turn": 2.215, "posture": 0.839,
          "head_motion": 1.237, "bend": 1.475, "speed_sd": 0.522, "length_sd": 0.229, "posture_sd": 0.570},
    "7": {"share_of_train": 0.016, "speed": 0.898, "length": 0.510, "turn": 7.608, "posture": 1.211,
          "head_motion": 0.821, "bend": 0.499, "speed_sd": 0.622, "length_sd": 0.248, "posture_sd": 0.747},
    "1": {"share_of_train": 0.004, "speed": 37.584, "length": 0.730, "turn": 0.387, "posture": 38.162,
          "head_motion": 0.688, "bend": 0.939, "speed_sd": 94.428, "length_sd": 0.190, "posture_sd": 94.000},
    "6": {"share_of_train": 0.002, "speed": 0.676, "length": 0.708, "turn": 3.442, "posture": 1.206,
          "head_motion": 0.759, "bend": 0.368, "speed_sd": 69.425, "length_sd": 0.194, "posture_sd": 69.384},
}
HUMAN = {"0": "sniffing", "2": "grooming", "3": "rearing", "4": "resting", "5": "walking", "7": "turning"}


def test_rule_naming_reproduces_the_hand_named_clusters():
    naming = name_clusters(NAMED)
    agree = naming_agreement(naming, HUMAN)
    assert agree["n_named_clusters"] == 6
    assert agree["weighted_agreement"] >= 0.99, agree["mismatches"]


def test_rule_naming_rejects_tracking_jumps_and_only_uses_catalogue_names():
    naming = name_clusters(NAMED)
    assert naming["1"]["name"] == "other" and naming["6"]["name"] == "other"  # 38 body lengths/s is a jump
    assert all(canonical(n["name"]) for n in naming.values())
    assert all(n["name"] in {"resting", "walking", "running", "sniffing", "grooming", "rearing", "turning",
                             "other"} for n in naming.values())


def test_naming_agreement_reports_nothing_to_compare():
    assert naming_agreement({}, {})["n_named_clusters"] == 0
