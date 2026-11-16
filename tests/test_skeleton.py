import numpy as np
import pytest

from animalpose.skeleton import CANONICAL, DATASET_MAPPINGS, register_mapping, remap_keypoints
from animalpose.skeleton.species import CALMS21_ORDER


def test_vocabulary_is_consistent():
    names = CANONICAL.names
    assert len(names) == len(set(names)) == CANONICAL.num_keypoints == 25
    assert CANONICAL.edge_index.max() < CANONICAL.num_keypoints


def test_flip_index_is_an_involution_that_swaps_sides():
    flip = CANONICAL.flip_index
    assert np.array_equal(flip[flip], np.arange(CANONICAL.num_keypoints))
    for i, kp in enumerate(CANONICAL.keypoints):
        other = CANONICAL.keypoints[flip[i]]
        assert other.region == kp.region
        assert (kp.side == "mid") == (flip[i] == i)


@pytest.mark.parametrize("dataset", sorted(DATASET_MAPPINGS))
def test_mappings_target_canonical_names(dataset):
    mapping = DATASET_MAPPINGS[dataset]
    assert set(mapping.values()) <= set(CANONICAL.names)
    assert len(set(mapping.values())) == len(mapping), "two dataset points map to one canonical point"


def test_remap_places_points_and_masks_unlabeled():
    kp = np.arange(len(CALMS21_ORDER) * 3, dtype=np.float32).reshape(-1, 3)
    out, labeled = remap_keypoints(CALMS21_ORDER, kp, "calms21")
    assert labeled.sum() == 7
    np.testing.assert_array_equal(out[CANONICAL.index("neck")], kp[3])
    assert np.isnan(out[CANONICAL.index("left_front_paw")]).all()


def test_register_mapping_rejects_unknown_targets():
    with pytest.raises(ValueError):
        register_mapping("bad", {"snout": "snoot"})
